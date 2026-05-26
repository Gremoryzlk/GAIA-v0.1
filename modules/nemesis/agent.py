"""NemesisAgent (L2) — контроль качества решений для GAIA v7.3.

Роли:
- Shadow mode: параллельная проверка решений агентов
- Regression testing: проверка после обучения
- Rogue detection: систематические отклонения → rogue_score
- Confidence calibration: точность уверенности агентов
"""

import json
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple
from uuid import uuid4

from agents.base_agent import BaseAgent
from core.eventbus.event import Event, EventPriority
from modules.hephaestus.tools.base_tool import BaseTool, ToolResult
from shared.constants import SHADOW_DIVERGENCE_THRESHOLD
from shared.tool_pool import ToolPool

logger = logging.getLogger(__name__)

_HISTORY_SIZE = 100          # сколько решений хранить в истории
_REGRESSION_SAMPLE = 20      # сколько задач брать для regression test
_CALIBRATION_WINDOW = 50     # окно для calibration
_ROGUE_SUSPECT_THRESHOLD = 3 # сколько расхождений до подозрения


class _QualityCheckTool(BaseTool):
    """Инструмент оценки качества решения."""

    def __init__(self) -> None:
        super().__init__(
            name="quality_check",
            description="Evaluate decision quality and detect divergence",
        )

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        decision = params.get("decision", {})
        expected = params.get("expected", {})
        context = params.get("context", {})

        score = self._score_decision(decision, expected, context)
        divergence = self._calc_divergence(decision, expected)

        result = {
            "score": score,
            "divergence": divergence,
            "above_threshold": round(divergence, 6) > SHADOW_DIVERGENCE_THRESHOLD,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return ToolResult(
            success=True,
            output=json.dumps(result),
            metadata=result,
        )

    def _score_decision(
        self, decision: Dict, expected: Dict, context: Dict
    ) -> float:
        """Оценка решения от 0.0 до 1.0."""
        if not expected:
            # Без эталона — оцениваем по confidence
            confidence = decision.get("confidence", 0.5)
            escalated = decision.get("escalated", False)
            return confidence * (0.8 if escalated else 1.0)

        # С эталоном — сравниваем decision type
        expected_decision = expected.get("decision", "")
        actual_decision = decision.get("decision", "")
        match = 1.0 if expected_decision == actual_decision else 0.0

        # Учитываем разницу в confidence
        conf_diff = abs(
            decision.get("confidence", 0.5) - expected.get("confidence", 0.5)
        )
        return max(0.0, match - conf_diff * 0.5)

    def _calc_divergence(self, decision: Dict, expected: Dict) -> float:
        """Вычисляет степень расхождения решения от ожидаемого."""
        if not expected:
            return 0.0
        conf_a = decision.get("confidence", 0.5)
        conf_b = expected.get("confidence", 0.5)
        decision_match = decision.get("decision") == expected.get("decision")
        base_divergence = abs(conf_a - conf_b)
        if not decision_match:
            base_divergence = max(base_divergence, 0.3)
        return min(1.0, base_divergence)

    def validate_params(self, params: Dict[str, Any]):
        return True, None


class _CalibrationTool(BaseTool):
    """Инструмент калибровки уверенности агентов."""

    def __init__(self) -> None:
        super().__init__(
            name="calibrate_confidence",
            description="Calibrate agent confidence against actual outcomes",
        )

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        history = params.get("history", [])
        if len(history) < 5:
            return ToolResult(
                success=True,
                output=json.dumps({"calibrated": False, "reason": "insufficient_data"}),
                metadata={"calibrated": False},
            )

        claimed = [h.get("confidence", 0.5) for h in history]
        actual = [1.0 if h.get("success", False) else 0.0 for h in history]

        avg_claimed = sum(claimed) / len(claimed)
        avg_actual = sum(actual) / len(actual)
        bias = avg_claimed - avg_actual  # > 0 = overconfident

        result = {
            "calibrated": True,
            "avg_claimed_confidence": round(avg_claimed, 3),
            "avg_actual_success": round(avg_actual, 3),
            "bias": round(bias, 3),
            "overconfident": bias > 0.15,
            "underconfident": bias < -0.15,
            "sample_size": len(history),
        }
        return ToolResult(
            success=True,
            output=json.dumps(result),
            metadata=result,
        )

    def validate_params(self, params: Dict[str, Any]):
        return True, None


class NemesisAgent(BaseAgent):
    """L2 агент контроля качества."""

    def __init__(self) -> None:
        super().__init__(agent_type="nemesis_l2", level=2)

        # История решений per agent
        self._decision_history: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=_HISTORY_SIZE)
        )
        # Счётчик расхождений per agent
        self._divergence_counts: Dict[str, int] = defaultdict(int)
        # История для regression testing
        self._regression_baseline: List[Dict] = []
        self._lock = threading.Lock()

        # Инструменты
        self._quality_tool = _QualityCheckTool()
        self._calibration_tool = _CalibrationTool()

        pool = ToolPool()
        pool.register(self._quality_tool)
        pool.register(self._calibration_tool)

        logger.info("NemesisAgent %s initialized.", self.agent_id)

    def _subscribe_to_events(self) -> None:
        self.eventbus.subscribe(
            name=f"{self.agent_id}_subscriber",
            event_types=[
                "task.completed",
                "agent.restored",
                "metrics.system",
                "quality.check",
            ],
            callback=self._handle_event,
        )

    def _handle_event(self, event: Event) -> None:
        try:
            if event.event_type == "task.completed":
                self._evaluate_completed_task(event)
            elif event.event_type == "agent.restored":
                self._run_regression_test(event.payload.get("agent_id"))
            elif event.event_type == "quality.check":
                self._shadow_check(event)
        except Exception as e:
            logger.error("NemesisAgent event error: %s", e)

    def _evaluate_completed_task(self, event: Event) -> None:
        """Оценивает завершённую задачу."""
        payload = event.payload
        result = payload.get("result", {})
        source = event.source

        if not result:
            return

        # Записываем в историю
        record = {
            "task_id": payload.get("task_id"),
            "decision": result.get("decision", ""),
            "confidence": result.get("confidence", 0.5),
            "escalated": result.get("escalated", False),
            "latency": result.get("latency", 0),
            "success": not result.get("escalated", False),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }

        with self._lock:
            self._decision_history[source].append(record)

            # Сохраняем в regression baseline если достаточно данных
            if len(self._regression_baseline) < _REGRESSION_SAMPLE:
                self._regression_baseline.append(record)

        # Shadow mode check
        self._check_divergence(source, result)

        # Калибровка каждые N решений
        history = list(self._decision_history[source])
        if len(history) % _CALIBRATION_WINDOW == 0 and len(history) > 0:
            self._calibrate_agent(source, history)

    def _check_divergence(self, agent_id: str, result: Dict) -> None:
        """Проверяет расхождение с историческим baseline."""
        with self._lock:
            history = list(self._decision_history[agent_id])

        if len(history) < 5:
            return

        # Средняя уверенность из истории
        avg_confidence = sum(h["confidence"] for h in history[-10:]) / min(10, len(history))
        current_confidence = result.get("confidence", 0.5)
        divergence = abs(current_confidence - avg_confidence)

        if divergence > SHADOW_DIVERGENCE_THRESHOLD:
            with self._lock:
                self._divergence_counts[agent_id] += 1
                count = self._divergence_counts[agent_id]

            logger.warning(
                "Nemesis: divergence detected agent=%s divergence=%.3f count=%d",
                agent_id, divergence, count,
            )

            # После N расхождений → подозрение на rogue
            if count >= _ROGUE_SUSPECT_THRESHOLD:
                self.publish_event(
                    event_type="agent.rogue_suspect",
                    priority=EventPriority.HIGH,
                    payload={
                        "agent_id": agent_id,
                        "divergence_count": count,
                        "last_divergence": round(divergence, 3),
                        "threshold": SHADOW_DIVERGENCE_THRESHOLD,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                # Сбрасываем счётчик после публикации
                with self._lock:
                    self._divergence_counts[agent_id] = 0

            self.publish_event(
                event_type="quality.alert",
                priority=EventPriority.HIGH,
                payload={
                    "agent_id": agent_id,
                    "divergence": round(divergence, 3),
                    "threshold": SHADOW_DIVERGENCE_THRESHOLD,
                    "current_confidence": current_confidence,
                    "avg_confidence": round(avg_confidence, 3),
                },
            )

    def _shadow_check(self, event: Event) -> None:
        """Параллельная проверка через quality_check tool."""
        payload = event.payload
        result = self._quality_tool.execute({
            "decision": payload.get("decision", {}),
            "expected": payload.get("expected", {}),
            "context": payload.get("context", {}),
        })
        data = json.loads(result.output)

        self.publish_event(
            event_type="quality.report",
            priority=EventPriority.NORMAL,
            payload={
                "task_id": payload.get("task_id"),
                "score": data["score"],
                "divergence": data["divergence"],
                "above_threshold": data["above_threshold"],
                "timestamp": data["timestamp"],
            },
        )

    def _calibrate_agent(self, agent_id: str, history: List[Dict]) -> None:
        """Калибрует уверенность агента."""
        result = self._calibration_tool.execute({"history": history})
        data = json.loads(result.output)

        if not data.get("calibrated"):
            return

        if data.get("overconfident") or data.get("underconfident"):
            logger.warning(
                "Nemesis: calibration alert agent=%s bias=%.3f overconfident=%s",
                agent_id, data["bias"], data.get("overconfident"),
            )
            self.publish_event(
                event_type="quality.calibration_alert",
                priority=EventPriority.NORMAL,
                payload={
                    "agent_id": agent_id,
                    **data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

    def _run_regression_test(self, agent_id: Optional[str] = None) -> None:
        """Запускает regression test после восстановления агента."""
        with self._lock:
            baseline = list(self._regression_baseline)

        if not baseline:
            logger.info("Nemesis: no regression baseline yet.")
            return

        logger.info(
            "Nemesis: running regression test for agent=%s samples=%d",
            agent_id, len(baseline),
        )

        failed = [r for r in baseline if r.get("escalated", False)]
        pass_rate = 1.0 - (len(failed) / len(baseline))

        self.publish_event(
            event_type="quality.regression_result",
            priority=EventPriority.HIGH,
            payload={
                "agent_id": agent_id,
                "pass_rate": round(pass_rate, 3),
                "total": len(baseline),
                "failed": len(failed),
                "passed": len(baseline) - len(failed),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        if pass_rate < 0.7:
            logger.error(
                "Nemesis: regression FAILED agent=%s pass_rate=%.1f%%",
                agent_id, pass_rate * 100,
            )
            self.publish_event(
                event_type="quality.regression_failed",
                priority=EventPriority.CRITICAL,
                payload={
                    "agent_id": agent_id,
                    "pass_rate": round(pass_rate, 3),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Возвращает статистику по агенту."""
        with self._lock:
            history = list(self._decision_history.get(agent_id, []))
            divergences = self._divergence_counts.get(agent_id, 0)

        if not history:
            return {"agent_id": agent_id, "no_data": True}

        return {
            "agent_id": agent_id,
            "total_evaluated": len(history),
            "avg_confidence": round(
                sum(h["confidence"] for h in history) / len(history), 3
            ),
            "avg_latency": round(
                sum(h["latency"] for h in history) / len(history), 3
            ),
            "escalation_rate": round(
                sum(1 for h in history if h["escalated"]) / len(history), 3
            ),
            "divergence_count": divergences,
        }

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.mark_busy()
        try:
            action = task.get("action", "check")

            if action == "check":
                result = self._quality_tool.execute(task)
            elif action == "calibrate":
                agent_id = task.get("agent_id", "")
                with self._lock:
                    history = list(self._decision_history.get(agent_id, []))
                result = self._calibration_tool.execute({"history": history})
            elif action == "stats":
                agent_id = task.get("agent_id", "")
                stats = self.get_agent_stats(agent_id)
                result = ToolResult(
                    success=True,
                    output=json.dumps(stats),
                    metadata=stats,
                )
            elif action == "regression":
                self._run_regression_test(task.get("agent_id"))
                result = ToolResult(
                    success=True,
                    output=json.dumps({"started": True}),
                    metadata={},
                )
            else:
                result = self._quality_tool.execute(task)

            self.tasks_processed += 1
            return {
                "success": result.success,
                "output": result.output,
                "action": action,
                "agent_id": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error("NemesisAgent task error: %s", e)
            raise
        finally:
            self.mark_idle()
