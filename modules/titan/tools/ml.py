"""Titan ML tool — обучение через MetaController + ModelState versioning.

GAIA обучается БЕЗ Titan — Titan только ускоряет.
Titan принимает TrainingBatch → MetaController.train() → новый ModelState.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from modules.hephaestus.tools.base_tool import BaseTool, ToolResult
from modules.titan.lifecycle import DataLifecycleManager

logger = logging.getLogger(__name__)

_MODEL_VERSION_PREFIX = "titan"


@dataclass
class TrainingBatch:
    """Батч обучения для MetaController.

    Содержит набор примеров (inputs, targets) для одного шага обучения.
    """
    batch_id: str
    inputs: List[List[float]]
    targets: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLTrainTool(BaseTool):
    """Инструмент ML обучения через MetaController (локальный).

    Выполняет один шаг эволюции весов модели.
    Сохраняет ModelState через DataLifecycleManager.
    """

    def __init__(
        self,
        meta_controller=None,
        lifecycle: Optional[DataLifecycleManager] = None,
    ) -> None:
        super().__init__(
            name="ml_train",
            description="MetaController training step with ModelState versioning",
        )
        self._mc = meta_controller
        self._lifecycle = lifecycle or DataLifecycleManager()
        self._version_counter: int = self._load_version_counter()

    def _load_version_counter(self) -> int:
        """Восстанавливает счётчик версий из существующих ModelState."""
        try:
            versions = self._lifecycle.list_model_states() if self._lifecycle else []
            titan_versions = [
                v for v in versions
                if v.startswith(_MODEL_VERSION_PREFIX)
            ]
            if not titan_versions:
                return 0
            last = max(
                int(v.removeprefix(_MODEL_VERSION_PREFIX + "_"))
                for v in titan_versions
                if v.removeprefix(_MODEL_VERSION_PREFIX + "_").isdigit()
            )
            return last + 1
        except Exception:
            return 0

    def validate_params(self, params: Dict[str, Any]):
        if "batch" not in params and "weights_shape" not in params:
            return False, "batch or weights_shape required"
        return True, None

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        valid, err = self.validate_params(params)
        if not valid:
            return ToolResult(success=False, output="", error=err)

        try:
            # Инициализация или получение текущего ModelState
            current_state = self._get_or_init_state(params)
            if current_state is None:
                return ToolResult(
                    success=False, output="",
                    error="MetaController not configured — no state to train"
                )

            # Один шаг эволюции через MetaController
            if self._mc:
                new_state = self._mc.step(current_state)
            else:
                # Fallback: применяем батч градиентным шагом
                new_state = self._gradient_step(current_state, params)

            # Сохраняем ModelState
            version = f"{_MODEL_VERSION_PREFIX}_{self._version_counter}"
            self._version_counter += 1

            state_dict = {
                "step": new_state.step,
                "fitness": new_state.fitness,
                "weights_shape": list(new_state.weights.shape),
                "architecture": new_state.architecture,
                "weights_hash": _hash_weights(new_state.weights),
            }
            path = self._lifecycle.save_model_state(state_dict, version)

            # Лог обучения
            self._lifecycle.write_training_log({
                "version": version,
                "step": new_state.step,
                "fitness": new_state.fitness,
                "batch_id": params.get("batch", {}).get("batch_id", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            result = {
                "version": version,
                "step": new_state.step,
                "fitness": new_state.fitness,
                "model_state_path": path,
                "weights_shape": list(new_state.weights.shape),
            }
            logger.info(
                "MLTrainTool: training step done version=%s step=%d fitness=%.4f",
                version, new_state.step, new_state.fitness,
            )
            return ToolResult(
                success=True,
                output=json.dumps(result),
                metadata=result,
            )

        except Exception as e:
            logger.error("MLTrainTool: error: %s", e)
            return ToolResult(success=False, output="", error=str(e))

    def _get_or_init_state(self, params: Dict[str, Any]):
        """Получает текущий ModelState или создаёт начальный."""
        from core.safety.meta_controller import ModelState

        # Если MetaController передан — берём его текущее состояние
        if self._mc and hasattr(self._mc, "vault") and self._mc.vault:
            return self._mc.vault[-1]

        # Инициализируем по переданной форме весов
        weights_shape = params.get("weights_shape", [32, 32])
        architecture = params.get("architecture", {"type": "dense", "layers": weights_shape})
        rng = np.random.default_rng(42)
        weights = rng.standard_normal(weights_shape).astype(np.float32) * 0.01

        state = ModelState(
            weights=weights,
            architecture=architecture,
            step=0,
            fitness=0.0,
        )
        state.compute_signature()
        return state

    def _gradient_step(self, state, params: Dict[str, Any]):
        """Минимальный gradient step без MetaController (fallback)."""
        from core.safety.meta_controller import ModelState

        batch = params.get("batch", {})
        inputs = batch.get("inputs", [])
        targets = batch.get("targets", [])

        lr = params.get("learning_rate", 0.001)

        new_weights = state.weights.copy()
        fitness = 0.0

        if inputs and targets:
            try:
                X = np.array(inputs, dtype=np.float32)
                y = np.array(targets, dtype=np.float32)
                W = state.weights.reshape(X.shape[1], -1) if state.weights.ndim <= 2 else state.weights
                pred = (X @ W).flatten()[:len(y)]
                y_trimmed = y[:len(pred)]
                err = pred - y_trimmed
                fitness = float(-np.mean(err ** 2))
                grad = X[:len(y_trimmed)].T @ err / max(len(y_trimmed), 1)
                grad_reshaped = grad.reshape(state.weights.shape) if grad.size == state.weights.size else np.zeros_like(state.weights)
                new_weights = state.weights - lr * grad_reshaped
            except Exception as e:
                logger.debug("MLTrainTool: gradient step shape error %s — using perturbation", e)
                rng = np.random.default_rng(state.step)
                new_weights = state.weights + rng.standard_normal(state.weights.shape) * lr

        new_state = ModelState(
            weights=new_weights,
            architecture=state.architecture,
            step=state.step + 1,
            fitness=fitness,
        )
        new_state.compute_signature()
        return new_state


def _hash_weights(weights: np.ndarray) -> str:
    import hashlib
    return hashlib.sha256(weights.tobytes()).hexdigest()[:16]
