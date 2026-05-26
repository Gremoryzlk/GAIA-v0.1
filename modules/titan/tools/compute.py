"""Titan compute tool — тяжёлые numpy/scipy вычисления.

Локальный fallback. Основной режим — через Titan сервер.
"""

import json
import logging
from typing import Any, Dict

from modules.hephaestus.tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_SUPPORTED_OPS = {"matrix_mul", "svd", "fft", "stats", "dot", "norm", "inv"}


class ComputeTool(BaseTool):
    """Numpy-вычисления для Titan (локальный fallback).

    Поддерживаемые операции:
      matrix_mul — перемножение матриц
      svd        — сингулярное разложение
      fft        — быстрое преобразование Фурье
      stats      — описательная статистика
      dot        — скалярное/матричное произведение
      norm       — норма вектора/матрицы
      inv        — обратная матрица
    """

    def __init__(self) -> None:
        super().__init__(
            name="compute",
            description="Heavy numpy computations (local fallback)",
        )

    def validate_params(self, params: Dict[str, Any]):
        op = params.get("operation")
        if not op:
            return False, "operation required"
        if op not in _SUPPORTED_OPS:
            return False, f"unsupported operation '{op}', supported: {sorted(_SUPPORTED_OPS)}"
        if "data" not in params:
            return False, "data required"
        return True, None

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        valid, err = self.validate_params(params)
        if not valid:
            return ToolResult(success=False, output="", error=err)

        try:
            import numpy as np
        except ImportError:
            return ToolResult(
                success=False, output="", error="numpy not installed"
            )

        op = params["operation"]
        data = params["data"]

        try:
            result = self._dispatch(np, op, data, params)
            return ToolResult(
                success=True,
                output=json.dumps(result, default=_json_safe),
                metadata={"operation": op},
            )
        except Exception as e:
            logger.error("ComputeTool: operation=%s error=%s", op, e)
            return ToolResult(success=False, output="", error=str(e))

    def _dispatch(self, np, op: str, data: Any, params: Dict) -> Any:
        if op == "matrix_mul":
            a = np.array(data["a"])
            b = np.array(data["b"])
            return {"result": np.matmul(a, b).tolist()}

        elif op == "svd":
            m = np.array(data)
            U, s, Vt = np.linalg.svd(m, full_matrices=params.get("full_matrices", False))
            return {"U": U.tolist(), "s": s.tolist(), "Vt": Vt.tolist()}

        elif op == "fft":
            arr = np.array(data)
            result = np.fft.fft(arr)
            return {
                "real": result.real.tolist(),
                "imag": result.imag.tolist(),
                "abs": np.abs(result).tolist(),
            }

        elif op == "stats":
            arr = np.array(data, dtype=float)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "shape": list(arr.shape),
            }

        elif op == "dot":
            a = np.array(data["a"])
            b = np.array(data["b"])
            return {"result": np.dot(a, b).tolist()}

        elif op == "norm":
            arr = np.array(data)
            ord_ = params.get("ord")
            return {"norm": float(np.linalg.norm(arr, ord=ord_))}

        elif op == "inv":
            m = np.array(data)
            return {"result": np.linalg.inv(m).tolist()}

        raise ValueError(f"Unknown operation: {op}")


def _json_safe(obj):
    """JSON serializer для numpy типов."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")
