"""MetaController and SafetyGate for GAIA v7.3 Self-Evolution Subsystem."""

import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from shared.constants import MAX_ROLLBACK_DEPTH

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    weights: np.ndarray
    architecture: dict
    step: int
    fitness: float
    signature: str = field(default="", init=False)

    def compute_signature(self) -> str:
        payload = json.dumps(
            {
                "w_hash": hashlib.sha256(self.weights.tobytes()).hexdigest(),
                "arch": str(sorted(self.architecture.items())),
                "step": self.step,
                "fitness": self.fitness,
            },
            sort_keys=True,
        )
        self.signature = hashlib.sha256(payload.encode()).hexdigest()
        return self.signature


def _default_mutation_fn(
    state: "ModelState", budget: float, seed: int
) -> "ModelState":
    """Детерминированная мутация весов на основе seed шага."""
    rng = np.random.default_rng(seed)
    scale = min(0.01, budget * 0.001)
    delta = rng.standard_normal(state.weights.shape) * scale
    return ModelState(
        weights=state.weights + delta,
        architecture=state.architecture,
        step=state.step,
        fitness=0.0,
    )


class SafetyGate:
    def __init__(
        self,
        safety_constraints: List[
            Callable[["ModelState"], Tuple[bool, Optional[str]]]
        ],
    ) -> None:
        self.constraints = safety_constraints

    def validate(self, state: "ModelState") -> Tuple[bool, Optional[str]]:
        for constraint in self.constraints:
            ok, msg = constraint(state)
            if not ok:
                return False, msg
        return True, None


def _default_safety_gate() -> SafetyGate:
    return SafetyGate([])


class MetaController:
    def __init__(
        self,
        eval_fn: Callable[["ModelState"], float],
        mutation_fn: Callable[["ModelState", float, int], "ModelState"] = None,
        safety_gate: SafetyGate = None,
        max_rollback_depth: int = MAX_ROLLBACK_DEPTH,
        evolution_budget: float = 100.0,
    ) -> None:
        self.eval_fn = eval_fn
        self.mutation_fn = mutation_fn or _default_mutation_fn
        self.safety = safety_gate or _default_safety_gate()
        self.vault: List[ModelState] = []
        self.max_rollback = max_rollback_depth
        self.budget = evolution_budget

    def step(self, current: ModelState) -> ModelState:
        if self.budget <= 0:
            logger.warning("Evolution budget exhausted — skipping mutation.")
            return current

        seed = current.step
        candidate = self.mutation_fn(current, self.budget, seed)
        candidate.fitness = self.eval_fn(candidate)
        candidate.step = current.step + 1

        ok, reason = self.safety.validate(candidate)

        if ok:
            candidate.compute_signature()
            self.vault.append(candidate)
            if len(self.vault) > self.max_rollback:
                self.vault.pop(0)
            self.budget -= abs(candidate.fitness - current.fitness)
            return candidate

        logger.warning(
            "SafetyGate rejected candidate at step %d: %s — rolling back.",
            candidate.step,
            reason,
        )
        return self._rollback(current)

    def _rollback(self, last_valid: ModelState) -> ModelState:
        depth = min(self.max_rollback, len(self.vault))
        if depth == 0:
            last_valid.compute_signature()
            return last_valid
        state = deepcopy(self.vault[-depth])
        logger.info("Rolled back to step %d.", state.step)
        return state


def reset_agent(mc: "MetaController", agent: "Any") -> bool:
    """Восстанавливает изолированного агента если fitness улучшился.

    Проверяет последние 3 шага vault — если fitness растёт,
    агент получает второй шанс.

    Args:
        mc: MetaController instance.
        agent: BaseAgent instance для восстановления.

    Returns:
        True если агент восстановлен, False иначе.
    """
    if len(mc.vault) < 3:
        logger.info(
            "reset_agent: vault too small (%d < 3) — cannot assess fitness trend.",
            len(mc.vault),
        )
        return False

    last_three = mc.vault[-3:]
    fitness_values = [s.fitness for s in last_three]
    improving = fitness_values[-1] > fitness_values[0]

    logger.info(
        "reset_agent: fitness trend=%s values=%s agent=%s",
        "improving" if improving else "not improving",
        [round(f, 4) for f in fitness_values],
        agent.agent_id,
    )

    if improving:
        result = agent.reset_isolation("meta_controller")
        if result:
            logger.info("reset_agent: agent %s successfully restored.", agent.agent_id)
        return result

    logger.info("reset_agent: fitness not improving — keeping agent isolated.")
    return False
