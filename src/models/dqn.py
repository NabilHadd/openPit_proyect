"""DQN (Mnih et al., 2015) over the cone view of the mine environment, with a small network
that generalizes BETWEEN BLOCKS.

Actions are those of MineEnv.cone_actions: extract the pending cone of a block (every plan
is feasible), advance the period, or stop. There is no cursor and no forced start.

Q(s, a) is not a table: a small multilayer perceptron (scikit-learn's MLPRegressor, trained
online with partial_fit) receives a FEATURE VECTOR of the pair (state, action) and returns
its value. The features describe the action's cone in that state (src/models/dqn.py,
ConeFeatures), so what is learned about one block applies to every block that looks alike,
and the network has the same size for 100 or 3000 blocks:

    0 reward of the action: VAN of the pending cone, discounted at the current period
    1 change of the shared-waste potential (src/training/potential.py) that the action causes
    2 value of the ore in the pending cone
    3 pending blocks of the cone, over the reachable blocks
    4 depth of the block (0 = surface, 1 = deepest level)
    5 capacity of the period used after the action (0 without capacity)
    6 period of the state after the action, over the number of periods
    7 fraction of the reachable blocks already extracted
    8 shared-waste potential of the state
    9 1 for advance_period, 0 for an extraction
Values (0, 1, 2, 8) are divided by the positive value of the reachable blocks, and so are
the rewards the network learns from, so its targets are of order 1.

STOP is never learned: its value is exactly 0 (the episode ends and nothing else is
earned), so the greedy choice compares the best Q of the other actions with 0, and the
target of a transition is reward + gamma * max(0, max Q(next)). A random exploration move
never chooses STOP, as in the tabular models.

Training (the usual DQN): epsilon-greedy episodes with the schedule of the YAML; every
transition goes to a replay buffer (replay_size, the oldest is dropped) together with the
features of the actions of its next state, so a target needs no recomputation; after
`warmup` transitions of pure random exploration, every `train_every` transitions one
minibatch of batch_size transitions updates the network towards targets computed with a
TARGET network, a copy of the network refreshed every target_update transitions.

Budget and cost: transitions = cone actions taken (cost_unit). One transition extracts a
whole cone, not one block as the cursor of the tabular models does, so transitions are not
comparable across the two families. Delivered plan: the best plan observed while training,
as in the tabular models; policy rollouts are diagnostics.
"""

from collections import Counter, deque
import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from statistics import mean
from typing import ClassVar
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_is_fitted

from configs import integer
from src.data.instance import MineInstance
from src.envs.mine_env import ADVANCE, STOP, MineEnv, State
from src.evaluation.metrics import Extraction, Plan, block_usage, checked_evaluation, same_value
from src.evaluation.plots import plot_history
from src.evaluation.reporter import write_json, write_plan_csv
from src.models.base import MineEstimator
from src.models.exploration import SCHEDULES, epsilon_at
from src.models.register import register
from src.models.tabular import EpisodeRecord, ObservedPlan, PolicyRollout
from src.training.potential import SharedWastePotential

N_FEATURES = 10


@dataclass(frozen=True)
class DQNParams:
    """gamma, epsilon, epsilon_schedule, transitions, policy_rollouts: as in the tabular
    models (transitions counts cone actions). Network: hidden_layers layers of hidden_units
    ReLU units, Adam with learning_rate. Replay: replay_size transitions, minibatches of
    batch_size every train_every transitions after warmup random ones; the target network is
    refreshed every target_update transitions."""
    gamma: float = 1.0
    epsilon: float = 0.3
    epsilon_schedule: str = "constant"
    transitions: int = 20_000
    policy_rollouts: int = 20
    hidden_units: int = 32
    hidden_layers: int = 2
    learning_rate: float = 0.001
    batch_size: int = 32
    replay_size: int = 10_000
    warmup: int = 1_000
    train_every: int = 1
    target_update: int = 500

    FIXED: ClassVar[tuple[str, ...]] = ("transitions", "policy_rollouts")
    SMOKE: ClassVar[dict] = {"transitions": 300, "policy_rollouts": 2, "warmup": 50, "replay_size": 300,
                             "target_update": 100}

    def __post_init__(self):
        if self.epsilon_schedule not in SCHEDULES:
            raise ValueError(f"epsilon_schedule debe ser uno de {sorted(SCHEDULES)}")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("gamma y epsilon deben estar en [0, 1]")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate debe estar en (0, 1)")
        for name in ("transitions", "policy_rollouts", "hidden_units", "hidden_layers", "batch_size",
                     "replay_size", "train_every", "target_update"):
            integer(getattr(self, name), name, 1)
        integer(self.warmup, "warmup", 0)
        if self.batch_size > self.replay_size:
            raise ValueError("batch_size no puede superar replay_size")


class ConeFeatures:
    """Actions of a state (as MineEnv.cone_actions, without STOP) and their feature rows,
    computed with matrix products over the cone membership matrix."""

    def __init__(self, env: MineEnv, potential: SharedWastePotential):
        self.env, self.parameters = env, env.parameters
        ids = sorted(env.blocks)
        self.index = {block_id: n for n, block_id in enumerate(ids)}
        self.candidates = [b for b in env.order if env.blocks[b].reachable]  # order of cone_actions
        members = np.zeros((len(self.candidates), len(ids)))
        for row, block_id in enumerate(self.candidates):
            members[row, [self.index[i] for i in env.cones[block_id]]] = 1.0
        self.members = members
        self.value = np.array([env.blocks[i].value for i in ids], dtype=float)
        self.ore_value = np.where(self.value > 0, self.value, 0.0) * np.array(
            [env.blocks[i].reachable for i in ids], dtype=float)
        self.usage = np.array([block_usage(env.blocks[i], self.parameters) for i in ids], dtype=float)
        zs = [env.blocks[b].z for b in self.candidates]
        top, bottom = max(zs), min(zs)
        self.depth = np.array([(top - z) / (top - bottom) if top > bottom else 0.0 for z in zs])
        self.scale = float(self.ore_value.sum()) or 1.0
        self.reachable = float(env.reachable_count)
        # shared-waste potential as matrices: ore rows, their value and the cost each waste block adds
        self.ore_rows = np.array([self.index[o] for o, _, _ in potential.terms], dtype=int)
        self.ore_values = np.array([v for _, v, _ in potential.terms], dtype=float)
        costs = np.zeros((len(potential.terms), len(ids)))
        for row, (_, _, wastes) in enumerate(potential.terms):
            for w, cost in wastes:
                costs[row, self.index[w]] = cost
        self.costs = costs

    def pending(self, state: State) -> np.ndarray:
        mask = np.ones(len(self.index))
        if state.extracted:
            mask[[self.index[i] for i in state.extracted]] = 0.0
        return mask

    def potential(self, pending: np.ndarray) -> np.ndarray:
        """Undiscounted shared-waste potential of one pending mask (1-D) or of several (rows)."""
        pending = np.atleast_2d(pending)
        net = self.ore_values[None, :] - pending @ self.costs.T
        alive = pending[:, self.ore_rows]
        return (np.maximum(net, 0.0) * alive).sum(axis=1)

    def __call__(self, state: State) -> tuple[tuple, np.ndarray]:
        """(actions, features): every action of cone_actions(state) except STOP, one row each."""
        if state.terminal:
            return (), np.zeros((0, N_FEATURES))
        parameters = self.parameters
        pending = self.pending(state)
        cone = self.members * pending[None, :]            # pending blocks of each candidate's cone
        alive = pending[[self.index[b] for b in self.candidates]] > 0
        used = cone @ self.usage
        if parameters.capacity is None:
            fits = np.ones(len(self.candidates), dtype=bool)
        else:
            total = state.used + used
            fits = (total <= parameters.capacity) | np.isclose(total, parameters.capacity, rtol=1e-12, atol=0)
        rows = np.flatnonzero(alive & fits)
        discount = 1.0 / (1.0 + parameters.discount_rate) ** state.period
        phi_now = float(self.potential(pending)[0])
        done = len(state.extracted) / self.reachable
        features = np.zeros((len(rows) + 1, N_FEATURES))
        if len(rows):
            chosen = cone[rows]
            after = self.potential(pending[None, :] - chosen)
            features[:len(rows), 0] = (chosen @ self.value) * discount / self.scale
            features[:len(rows), 1] = (after - phi_now) * discount / self.scale
            features[:len(rows), 2] = (chosen @ self.ore_value) * discount / self.scale
            features[:len(rows), 3] = chosen.sum(axis=1) / self.reachable
            features[:len(rows), 4] = self.depth[rows]
            if parameters.capacity is not None:
                features[:len(rows), 5] = (state.used + used[rows]) / parameters.capacity
            features[:len(rows), 6] = state.period / parameters.periods
            features[:len(rows), 7] = done
            features[:len(rows), 8] = phi_now * discount / self.scale
        actions = [self.candidates[r] for r in rows]
        if state.period + 1 < parameters.periods:
            next_discount = discount / (1.0 + parameters.discount_rate)
            features[len(rows)] = (0, 0, 0, 0, 0, 0, (state.period + 1) / parameters.periods, done,
                                   phi_now * next_discount / self.scale, 1)
            actions.append(ADVANCE)
        else:
            features = features[:len(rows)]
        return tuple(actions), features


@register("dqn", params=DQNParams)
class DQNEstimator(MineEstimator):
    cost_unit = "transitions"

    def __init__(self, gamma=1.0, epsilon=0.3, epsilon_schedule="constant", transitions=20_000,
                 policy_rollouts=20, hidden_units=32, hidden_layers=2, learning_rate=0.001, batch_size=32,
                 replay_size=10_000, warmup=1_000, train_every=1, target_update=500, random_state=0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_schedule = epsilon_schedule
        self.transitions = transitions
        self.policy_rollouts = policy_rollouts
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.warmup = warmup
        self.train_every = train_every
        self.target_update = target_update
        self.random_state = random_state

    # --- contract -------------------------------------------------------------------------

    def fit(self, instance: MineInstance) -> "DQNEstimator":
        self._check_params()
        env = MineEnv(instance.mine, instance.problem, instance.precedences, instance.environment)
        potential = SharedWastePotential(env.blocks, env.cones, instance.problem.discount_rate)
        self.features_ = ConeFeatures(env, potential)
        self.rng_ = random.Random(self.random_state)
        self._replay_rng = np.random.default_rng(self.random_state)
        self.network_ = MLPRegressor(hidden_layer_sizes=(self.hidden_units,) * self.hidden_layers,
                                     activation="relu", solver="adam", learning_rate_init=self.learning_rate,
                                     batch_size=self.batch_size, shuffle=False, random_state=self.random_state)
        self._target = None
        self._fitted = False
        self.n_updates_ = 0
        replay = deque(maxlen=self.replay_size)  # (features of the action, scaled reward, next features | None)
        self._start_trace()
        history, best, transitions = [], None, 0
        while transitions < self.transitions:
            episode = len(history)
            state = env.initial_state()
            actions, rows = self.features_(state)
            extractions, total, steps, end = [], 0.0, 0, None
            while transitions < self.transitions:
                if steps >= instance.environment.max_steps:
                    end = "episode_cap"
                    break
                if not actions:  # only STOP is left
                    step = env.cone_step(state, STOP)
                    transitions += 1
                    steps += 1
                    end = step.reason
                    break
                epsilon = epsilon_at(self.epsilon_schedule, self.epsilon, transitions, self.transitions)
                explore = transitions < self.warmup or self.rng_.random() < epsilon
                choice = self.rng_.randrange(len(actions)) if explore else self._greedy(rows, self.rng_)
                action = STOP if choice is None else actions[choice]
                step = env.cone_step(state, action)
                transitions += 1
                steps += 1
                if step.extracted:
                    extractions.extend(Extraction(i, step.state.period) for i in step.extracted)
                    total += step.reward
                    counters = self._counters(transitions, env)
                    if self._offer(counters, total):
                        best = total, Plan(tuple(extractions)), episode, steps
                if action == STOP:
                    end = step.reason
                    break
                next_actions, next_rows = self.features_(step.state)
                replay.append((rows[choice], step.reward / self.features_.scale,
                               None if step.terminal else next_rows))
                if transitions >= self.warmup and transitions % self.train_every == 0 and len(replay) >= self.batch_size:
                    self._train(replay)
                if self._fitted and transitions % self.target_update == 0:
                    self._target = copy.deepcopy(self.network_)
                state, actions, rows = step.state, next_actions, next_rows
                if step.terminal:
                    end = step.reason
                    break
            if end is None:
                end = "budget"
            plan = Plan(tuple(extractions))
            final = checked_evaluation(instance, plan)
            if not same_value(total, final.npv):
                raise RuntimeError("Las recompensas del episodio no coinciden con el VAN de su plan")
            history.append(EpisodeRecord(episode, steps, final.npv, end))
        if best is None:  # no extraction at all: the empty plan
            best = 0.0, Plan(), 0, 0
            self._offer(self._counters(transitions, env), 0.0)
        best_evaluation = checked_evaluation(instance, best[1])
        if not same_value(best_evaluation.npv, best[0]):
            raise RuntimeError("El mejor plan observado no coincide con su VAN verificado")
        self.instance_ = instance
        self.history_ = tuple(history)
        self.best_observed_ = ObservedPlan(best[1], best_evaluation, best[2], best[3])
        self.trace_ = tuple(self._trace)
        self.counters_ = self._counters(transitions, env)  # before the rollouts: they are not budget
        self.n_transitions_ = transitions
        self.policy_ = tuple(self._policy_rollout(env, instance, r) for r in range(self.policy_rollouts))
        return self

    def predict(self, instance: MineInstance) -> Plan:
        check_is_fitted(self, "best_observed_")
        self._check_instance(instance)
        return self.best_observed_.plan

    def diagnostics(self) -> dict:
        check_is_fitted(self, "best_observed_")
        values = [r.evaluation.npv for r in self.policy_]
        best = self.best_observed_
        loss = getattr(self.network_, "loss_", None)
        return {
            "training": {"transitions": self.n_transitions_, "episodes": len(self.history_),
                         "episode_ends": dict(sorted(Counter(e.end for e in self.history_).items())),
                         "network_updates": self.n_updates_,
                         "last_loss": None if loss is None or not math.isfinite(loss) else float(loss)},
            "best_observed": {"npv": best.evaluation.npv, "episode": best.episode, "step": best.step,
                              "blocks": len(best.plan.extractions)},
            "policy": {"rollouts": len(values), "npv_mean": mean(values), "npv_min": min(values),
                       "npv_max": max(values),
                       "episode_cap_rollouts": sum(r.end == "episode_cap" for r in self.policy_)},
        }

    def save_artifacts(self, folder: Path) -> None:
        """best_observed.csv (the delivered plan), policy_best.csv, training.json (episodes and
        rollouts) and training.svg (VAN of every training episode and best so far)."""
        check_is_fitted(self, "best_observed_")
        folder = Path(folder)
        write_plan_csv(folder / "best_observed.csv", self.best_observed_.plan)
        write_plan_csv(folder / "policy_best.csv", max(self.policy_, key=lambda r: r.evaluation.npv).plan)
        write_json(folder / "training.json", {
            "episodes": [asdict(e) for e in self.history_],
            "policy_rollouts": [{k: v for k, v in asdict(r).items() if k != "plan"} for r in self.policy_]})
        plot_history([e.npv for e in self.history_], folder / "training.svg")

    def compare_reference(self, reference_npv: float) -> dict[str, int]:
        check_is_fitted(self, "best_observed_")
        matches = sum(same_value(r.evaluation.npv, reference_npv) for r in self.policy_)
        return {"policy_matches": matches, "policy_rollouts": len(self.policy_),
                "policy_always_matches": int(matches == len(self.policy_))}

    # --- network --------------------------------------------------------------------------

    def q_values(self, rows: np.ndarray, network=None) -> np.ndarray:
        """Q of every row; 0 before the first update (the network does not exist yet)."""
        network = self.network_ if network is None else network
        if not self._fitted or len(rows) == 0:
            return np.zeros(len(rows))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return network.predict(rows)

    def _greedy(self, rows: np.ndarray, rng: random.Random) -> int | None:
        """Index of the action of highest Q, or None for STOP (value 0); ties at random."""
        values = self.q_values(rows)
        best = max(0.0, float(values.max())) if len(values) else 0.0
        tied = [n for n, v in enumerate(values) if v == best]
        if best == 0.0:
            tied.append(None)  # STOP is worth exactly 0
        return tied[rng.randrange(len(tied))]

    def _train(self, replay: deque):
        picks = self._replay_rng.choice(len(replay), size=self.batch_size, replace=False)
        batch = [replay[int(i)] for i in picks]
        x = np.stack([features for features, _, _ in batch])
        targets = np.array([reward for _, reward, _ in batch])
        if self._fitted and self.gamma > 0:
            target_network = self._target if self._target is not None else self.network_
            following = [(n, rows) for n, (_, _, rows) in enumerate(batch) if rows is not None and len(rows)]
            if following:
                stacked = np.concatenate([rows for _, rows in following])
                values = self.q_values(stacked, target_network)
                start = 0
                for n, rows in following:
                    targets[n] += self.gamma * max(0.0, float(values[start:start + len(rows)].max()))
                    start += len(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.network_.partial_fit(x, targets)
        self._fitted = True
        self.n_updates_ += 1

    def _counters(self, transitions: int, env: MineEnv) -> dict[str, int]:
        return {"transitions": transitions, "simulator_calls": env.calls, "updates": self.n_updates_}

    def _policy_rollout(self, env: MineEnv, instance: MineInstance, rollout: int) -> PolicyRollout:
        rng = random.Random(f"{self.random_state}:policy:{rollout}")  # never touches the training RNG
        state, extractions, steps, ties, end = env.initial_state(), [], 0, 0, "episode_cap"
        while steps < instance.environment.max_steps:
            actions, rows = self.features_(state)
            values = self.q_values(rows)
            best = max(0.0, float(values.max())) if len(values) else 0.0
            ties += sum(1 for v in values if v == best) + (best == 0.0) > 1
            choice = self._greedy(rows, rng) if actions else None
            step = env.cone_step(state, STOP if choice is None else actions[choice])
            steps += 1
            extractions.extend(Extraction(i, step.state.period) for i in step.extracted)
            state = step.state
            if step.terminal:
                end = step.reason
                break
        plan = Plan(tuple(extractions))
        return PolicyRollout(rollout, plan, checked_evaluation(instance, plan), steps, end, ties, 0, 0)
