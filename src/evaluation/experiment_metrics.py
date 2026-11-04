"""Passive diagnostics for controlled experiments; no reward, policy or stopping rule."""

from collections import Counter
from collections.abc import Callable, Hashable
from dataclasses import asdict, dataclass

from src.envs.spatial_cone_env import DIRECTIONS, SpatialState
from src.models.experimental_q_learning import control_physical_key


def mining_context(state: SpatialState) -> tuple:
    """Navigation context ignores the clock, not extraction, location or capacity."""
    return state.extracted, state.position, state.period, state.used


@dataclass
class TrajectoryStats:
    phase: str
    episode: int | None
    transitions: int = 0
    extraction_transitions: int = 0
    extracted_blocks: int = 0
    no_extraction_transitions: int = 0
    zero_extraction_moves: int = 0
    navigation_cycle_returns: int = 0
    longest_no_extraction_streak: int = 0
    trailing_no_extraction_transitions: int = 0
    unseen_learning_q_state_visits: int = 0
    unseen_learning_state_action_visits: int = 0
    termination_reason: str | None = None


class TrajectoryDiagnostics:
    def __init__(self, owner, phase: str, episode: int | None):
        self.owner = owner
        self.stats = TrajectoryStats(phase, episode)
        self.seen_navigation = set()

    def observe(self, state: SpatialState, action, _reward: float, following: SpatialState):
        self.owner.visit(state, action, self.stats)
        stats = self.stats
        stats.transitions += 1
        new_blocks = following.extracted - state.extracted
        if new_blocks:
            stats.extraction_transitions += 1
            stats.extracted_blocks += len(new_blocks)
            stats.trailing_no_extraction_transitions = 0
        else:
            stats.no_extraction_transitions += 1
            stats.trailing_no_extraction_transitions += 1
            stats.zero_extraction_moves += int(action in DIRECTIONS)
        stats.longest_no_extraction_streak = max(stats.longest_no_extraction_streak,
                                               stats.trailing_no_extraction_transitions)
        next_context = mining_context(following)
        if new_blocks or action not in DIRECTIONS:
            self.seen_navigation = {next_context}
        else:
            self.seen_navigation.add(mining_context(state))
            if next_context in self.seen_navigation:
                stats.navigation_cycle_returns += 1
            self.seen_navigation.add(next_context)

    def finish(self, reason: str | None, steps: int):
        if steps != self.stats.transitions:
            raise RuntimeError("Diagnóstico de transiciones no coincide con el episodio")
        self.stats.termination_reason = reason


class ControlDiagnostics:
    def __init__(self, state_key: Callable[[SpatialState], Hashable]):
        self.state_key = state_key
        self.learning: list[TrajectoryStats] = []
        self.greedy: TrajectoryStats | None = None
        self.q_visits = Counter()
        self.state_action_visits = Counter()
        self.physical_visits = Counter()
        self.first_physical = {}
        self.first_mining_context = {}
        self.aliased_q_rows = set()
        self.mine_aliased_q_rows = set()
        self.learning_seconds = 0.0
        self.greedy_seconds = 0.0

    def new_episode(self, phase: str, episode: int | None) -> TrajectoryDiagnostics:
        if phase not in {"learning", "construction"}:
            raise ValueError("Fase de diagnóstico desconocida")
        trajectory = TrajectoryDiagnostics(self, phase, episode)
        if phase == "learning":
            self.learning.append(trajectory.stats)
        else:
            if self.greedy is not None:
                raise ValueError("Solo se registra un rollout greedy por ejecución")
            self.greedy = trajectory.stats
        return trajectory

    def visit(self, state: SpatialState, action, stats: TrajectoryStats):
        key = self.state_key(state)
        if stats.phase == "construction":
            stats.unseen_learning_q_state_visits += int(key not in self.q_visits)
            stats.unseen_learning_state_action_visits += int((key, action) not in self.state_action_visits)
            return
        physical = control_physical_key(state)
        context = mining_context(state)
        self.q_visits[key] += 1
        self.state_action_visits[key, action] += 1
        self.physical_visits[physical] += 1
        if key in self.first_physical:
            if self.first_physical[key] != physical:
                self.aliased_q_rows.add(key)
            if self.first_mining_context[key] != context:
                self.mine_aliased_q_rows.add(key)
        else:
            self.first_physical[key] = physical
            self.first_mining_context[key] = context

    def payload(self, q_entries: int) -> dict:
        visits = sum(self.q_visits.values())
        unique = len(self.q_visits)
        learning_stats = [asdict(stats) for stats in self.learning]
        totals = {field: sum(row[field] for row in learning_stats) for field in (
            "transitions", "extraction_transitions", "extracted_blocks", "no_extraction_transitions",
            "zero_extraction_moves", "navigation_cycle_returns")}
        totals["episodes_with_navigation_cycles"] = sum(row["navigation_cycle_returns"] > 0 for row in learning_stats)
        totals["termination_counts"] = dict(Counter(row["termination_reason"] for row in learning_stats))
        totals["longest_no_extraction_streak"] = max((row["longest_no_extraction_streak"] for row in learning_stats), default=0)
        totals["no_extraction_fraction"] = totals["no_extraction_transitions"] / visits if visits else None
        unique_actions = len(self.state_action_visits)
        return {
            "learning": totals,
            "state_coverage": {
                "learning_q_state_visits": visits,
                "unique_learning_q_states": unique,
                "q_state_reuse_fraction": (visits - unique) / visits if visits else None,
                "q_states_visited_once_fraction": sum(v == 1 for v in self.q_visits.values()) / unique if unique else None,
                "unique_learning_state_action_pairs": unique_actions,
                "state_action_reuse_fraction": (visits - unique_actions) / visits if visits else None,
                "state_action_pairs_visited_once_fraction": sum(v == 1 for v in self.state_action_visits.values()) / unique_actions if unique_actions else None,
                "unique_learning_physical_states": len(self.physical_visits),
                "q_rows_with_distinct_physical_states": len(self.aliased_q_rows),
                "q_rows_with_distinct_mining_contexts": len(self.mine_aliased_q_rows),
                "q_entries": q_entries,
            },
            "episodes": learning_stats,
            "greedy": asdict(self.greedy) if self.greedy is not None else None,
            "timings": {"learning_seconds": self.learning_seconds, "greedy_seconds": self.greedy_seconds},
            "definitions": {
                "visits": "Una visita al estado previo por transición de aprendizaje; no se cuentan consultas bootstrap",
                "reuse_fraction": "(visitas - estados Q únicos) / visitas; NO prueba de consistencia ni convergencia",
                "once_fraction": "Filas Q visitadas una sola vez / filas Q únicas visitadas",
                "state_action_reuse_fraction": "(actualizaciones - pares estado/acción únicos) / actualizaciones; denominador cero -> null",
                "physical_state": "Estado completo con horizonte; fase normalizada solo bajo dinámica idéntica 6/6",
                "mining_context": "Extraídos, posición, período y uso; ignora reloj y etiqueta de fase",
                "navigation_cycle_returns": "Movimientos que vuelven a un contexto minero ya visto en el episodio; no significa ciclo infinito de la política ni invalidez",
                "no_extraction_transitions": "Incluye stop/avance y cualquier transición sin bloques nuevos",
                "zero_extraction_moves": "Solo movimientos direccionales sin extracción; excluye stop/avance",
                "unseen_learning_q_state_visits": "Visitas greedy a claves sin actualizaciones durante aprendizaje; una Q aprendida igual a cero no es desconocida",
                "unseen_learning_state_action_visits": "Acciones greedy cuyos pares clave/acción nunca fueron actualizados; distinto de clave desconocida",
            },
        }
