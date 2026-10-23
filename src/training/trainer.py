"""Episode orchestration, independent plan checks, and a read-only greedy rollout."""

from dataclasses import dataclass
from collections.abc import Callable
from time import perf_counter

from src.envs.mine_env import Action, MineEnv
from src.evaluation.metrics import Evaluation, Plan, evaluate_plan
from src.models.base import BaseModel


@dataclass(frozen=True)
class ObservedPath:
    plan: Plan
    evaluation: Evaluation
    episode: int | None
    phase: str
    steps: int
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    reward_sum: float
    steps: int
    plan: Plan
    evaluation: Evaluation
    actions: tuple[str, ...] = ()
    termination_reason: str = "terminal"
    best_path: ObservedPath | None = None


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[EpisodeRecord, ...]
    best_episode: int
    best_plan: Plan
    best_evaluation: Evaluation
    greedy_plan: Plan
    greedy_evaluation: Evaluation
    greedy_actions: tuple[str, ...] = ()
    greedy_termination_reason: str = "terminal"
    greedy_steps: int = 0
    greedy_best_path: ObservedPath | None = None
    selected_path: ObservedPath | None = None


def run_episode(env: MineEnv, model: BaseModel, *, learn: bool,
                on_transition: Callable[[int, Action], None] | None = None,
                transition_observer: Callable | None = None) -> tuple[float, int]:
    state = env.reset(training=learn)
    reward_sum = 0.0
    steps = 0
    bound = env.episode_step_bound
    while not state.terminal:
        if steps >= bound:
            raise RuntimeError("Episodio excede la cota de transiciones del entorno")
        action = model.select_action(state, env.available_actions(), explore=learn)
        next_state, reward, terminal = env.step(action)
        if learn:
            model.update(state, action, reward, next_state, env.available_actions(), terminal)
        reward_sum += reward
        steps += 1
        if on_transition is not None:
            on_transition(steps, action)
        if transition_observer is not None:
            transition_observer(state, action, reward, next_state)
        state = next_state
    return reward_sum, steps


def checked_evaluation(env: MineEnv) -> Evaluation:
    result = evaluate_plan(env.mine, env.plan, env.parameters, env.precedences)
    if result.feasible is not True:
        raise RuntimeError(f"El entorno produjo un plan inválido: {result.errors}")
    return result


class PathObserver:
    """Observe feasible prefixes without changing actions, rewards or termination."""

    def __init__(self, env: MineEnv, episode: int | None, phase: str):
        self.env = env
        self.episode = episode
        self.phase = phase
        self.actions = []
        self.last_plan = None
        self.best: ObservedPath | None = None

    def observe(self, steps: int, action: Action):
        self.actions.append(action)
        self.finish(steps)

    def finish(self, steps: int):
        if self.env.plan == self.last_plan:
            return
        evaluation = checked_evaluation(self.env)
        self.last_plan = self.env.plan
        if self.best is None or evaluation.npv > self.best.evaluation.npv:
            self.best = ObservedPath(self.env.plan, evaluation, self.episode, self.phase,
                                     steps, tuple(self.actions))


def train(env: MineEnv, model: BaseModel, episodes: int, *, diagnostics=None) -> TrainingResult:
    if episodes < 1:
        raise ValueError("Se requiere al menos un episodio")
    history = []
    best = None
    selected_path = None
    learning_started = perf_counter() if diagnostics is not None else 0.0
    for episode in range(episodes):
        observer = PathObserver(env, episode, "learning")
        trajectory = diagnostics.new_episode("learning", episode) if diagnostics is not None else None
        reward_sum, steps = run_episode(env, model, learn=True, on_transition=observer.observe,
                                       transition_observer=trajectory.observe if trajectory is not None else None)
        if trajectory is not None:
            trajectory.finish(getattr(env, "termination_reason", None), steps)
        observer.finish(steps)
        record = EpisodeRecord(episode, reward_sum, steps, env.plan, checked_evaluation(env),
                               tuple(getattr(env, "action_history", ())),
                               getattr(env, "termination_reason", "terminal"), observer.best)
        history.append(record)
        if best is None or record.evaluation.npv > best.evaluation.npv:
            best = record
        if observer.best is not None and (selected_path is None or observer.best.evaluation.npv > selected_path.evaluation.npv):
            selected_path = observer.best
    if diagnostics is not None:
        diagnostics.learning_seconds = perf_counter() - learning_started
    greedy_started = perf_counter() if diagnostics is not None else 0.0
    observer = PathObserver(env, None, "construction")
    trajectory = diagnostics.new_episode("construction", None) if diagnostics is not None else None
    _, greedy_steps = run_episode(env, model, learn=False, on_transition=observer.observe,
                                 transition_observer=trajectory.observe if trajectory is not None else None)
    if trajectory is not None:
        trajectory.finish(getattr(env, "termination_reason", None), greedy_steps)
    observer.finish(greedy_steps)
    greedy_evaluation = checked_evaluation(env)
    if observer.best is not None and (selected_path is None or observer.best.evaluation.npv > selected_path.evaluation.npv):
        selected_path = observer.best
    if diagnostics is not None:
        diagnostics.greedy_seconds = perf_counter() - greedy_started
    return TrainingResult(tuple(history), best.episode, best.plan, best.evaluation,
                          env.plan, greedy_evaluation, tuple(getattr(env, "action_history", ())),
                          getattr(env, "termination_reason", "terminal"), greedy_steps,
                          observer.best, selected_path)
