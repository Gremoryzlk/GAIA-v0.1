"""GaiaAgent (L0) for GAIA v7.3."""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agents.base_agent import BaseAgent
from uuid import uuid4
from core.eventbus.event import Event, EventPriority
from core.heuristic.heuristic_core import HeuristicCore
from core.llm.cache import LLMCache
from core.llm.interpreter import LLMInterpreter, TaskContext
from core.llm.prompt_templates import format_task_prompt
from shared.constants import LLM_CACHE_TTL_HOURS
from shared.types import AgentStatus
from shared.tool_pool import ToolPool

logger = logging.getLogger(__name__)


class GaiaAgent(BaseAgent):
    def __init__(self, heuristic_core: Optional[HeuristicCore] = None) -> None:
        super().__init__(agent_type="gaia_l0", level=0)
        self.heuristic_core = heuristic_core or HeuristicCore()
        self._latency_sum: float = 0.0
        self._latency_count: int = 0

        self.use_llm = os.environ.get("USE_LLM_INTERPRETER", "false").lower() == "true"
        self._llm_cache: Optional[LLMCache] = (
            LLMCache(default_ttl_hours=LLM_CACHE_TTL_HOURS) if self.use_llm else None
        )
        self._interpreter: Optional[LLMInterpreter] = (
            LLMInterpreter(cache=self._llm_cache) if self.use_llm else None
        )

        # HephaestusAgent подключается через manifest/модули
        self._hephaestus = None
        try:
            from modules.hephaestus.agent import HephaestusAgent
            self._hephaestus = HephaestusAgent()
            logger.info("HephaestusAgent connected to GaiaAgent.")
        except Exception as e:
            logger.warning("HephaestusAgent not available: %s", e)

        # ToolPool — fallback при недоступности Hephaestus
        self._pool = ToolPool()

        # Hermes — маршрутизатор
        self._hermes = None
        try:
            from modules.hermes.agent import HermesAgent
            self._hermes = HermesAgent()
            logger.info("HermesAgent connected to GaiaAgent.")
        except Exception as e:
            logger.warning("HermesAgent not available: %s", e)

        # Athena — планировщик
        self._athena = None
        try:
            from modules.athena.agent import AthenaAgent
            self._athena = AthenaAgent()
            logger.info("AthenaAgent connected to GaiaAgent.")
        except Exception as e:
            logger.warning("AthenaAgent not available: %s", e)

        # Titan — тяжёлые вычисления (L2, optional)
        self._titan = None
        try:
            from modules.titan.agent import TitanAgent
            self._titan = TitanAgent()
            logger.info("TitanAgent connected to GaiaAgent (enabled=%s).", self._titan._enabled)
        except Exception as e:
            logger.warning("TitanAgent not available: %s", e)

        if self.use_llm:
            logger.info("LLM interpreter enabled for agent %s.", self.agent_id)

    def _subscribe_to_events(self) -> None:
        self.eventbus.subscribe(
            name=f"{self.agent_id}_subscriber",
            event_types=["task.assigned", "task.failed", "compute.request"],
            callback=self._handle_event,
        )

    def _handle_event(self, event: Event) -> None:
        if event.event_type == "task.assigned":
            self.mark_busy()
        elif event.event_type == "task.failed":
            self.increment_rogue_score(0.1)
        elif event.event_type == "compute.request":
            self._route_compute_request(event)

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now(timezone.utc)
        self.mark_busy()

        try:
            # Tool-задачи делегируем HephaestusAgent
            if task.get("task_type") == "tool":
                if self._hephaestus and self._hephaestus.accept_task():
                    return self._hephaestus.process_task(task)
                else:
                    # Hephaestus недоступен — fallback через ToolPool
                    logger.warning(
                        "HephaestusAgent unavailable — using ToolPool fallback."
                    )
                    return self._tool_pool_fallback(task)

            context = {
                "task_id": task.get("task_id"),
                "task_type": task.get("task_type"),
                "payload": task.get("payload", {}),
                "priority": task.get("priority", 0),
                "assigned_agent": task.get("assigned_agent"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Сложные задачи (urgency > 0.7 или "and"/"then" в тексте) → Athena
            task_text = str(task.get("payload", {}).get("text", ""))
            llm_urgency = 0.0
            if task.get("llm_interpretation"):
                llm_urgency = task["llm_interpretation"].get("urgency", 0.0)

            is_complex = (
                llm_urgency > 0.7
                or " and " in task_text.lower()
                or " then " in task_text.lower()
            )

            if is_complex and self._athena and self._athena.accept_task():
                logger.info("Complex task detected — delegating to Athena.")
                return self._athena.process_task({
                    **task,
                    "task_id": task.get("task_id", str(uuid4())),
                    "text": task_text,
                })

            # LLM-интерпретация если включена
            llm_result: Optional[Dict[str, Any]] = None
            if self.use_llm and self._interpreter:
                llm_result = self._llm_interpret(task)
                if llm_result:
                    context["llm_interpretation"] = llm_result

            decision = self.heuristic_core.evaluate(context)

            latency = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._latency_sum += latency
            self._latency_count += 1

            escalated = decision.confidence < 0.75
            if escalated:
                self.publish_event(
                    event_type="routing.escalation",
                    priority=EventPriority.HIGH,
                    payload={
                        "task_id": task.get("task_id"),
                        "confidence": decision.confidence,
                        "reason": "low_confidence",
                        "latency": latency,
                    },
                )

            result = {
                "decision": decision.decision,
                "confidence": decision.confidence,
                "threshold": decision.threshold,
                "latency": latency,
                "escalated": escalated,
            }

            if llm_result:
                result["llm_interpretation"] = llm_result

            # Если decision → hephaestus, публикуем tool.execute
            if (
                decision.decision == "auto_execute"
                and isinstance(decision.trace, list)
                and decision.trace
            ):
                last_trace = decision.trace[-1] if isinstance(decision.trace[-1], dict) else {}
                action = last_trace.get("action", {}) if isinstance(last_trace, dict) else {}
                route_to = action.get("route_to") if isinstance(action, dict) else None

                if route_to == "hephaestus":
                    tool = (
                        (llm_result or {}).get("tool")
                        or (action.get("tool") if isinstance(action, dict) else None)
                    )
                    operation = (
                        (llm_result or {}).get("operation")
                        or (action.get("operation") if isinstance(action, dict) else None)
                    )
                    if tool and operation:
                        self.publish_event(
                            event_type="tool.execute",
                            priority=EventPriority.NORMAL,
                            payload={
                                "tool_name": tool,
                                "params": {
                                    "action": operation,
                                    "path": (llm_result or {}).get("path", ""),
                                    "agent_id": self.agent_id,
                                },
                            },
                        )
                        logger.info(
                            "Dispatched tool.execute: tool=%s operation=%s",
                            tool, operation,
                        )

            self.publish_event(
                event_type="task.completed",
                priority=EventPriority.NORMAL,
                payload={"task_id": task.get("task_id"), "result": result},
            )

            return result

        except Exception as e:
            self.publish_event(
                event_type="task.failed",
                priority=EventPriority.HIGH,
                payload={"task_id": task.get("task_id"), "error": str(e)},
            )
            raise

        finally:
            self.mark_idle()
            self.tasks_processed += 1

    def _llm_interpret(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            ctx: TaskContext = self._interpreter.interpret(task)
            if ctx.parse_error:
                logger.warning("LLM parse error: %s", ctx.parse_error)
                return None
            return {
                "task_type": ctx.task_type,
                "intent": ctx.intent,
                "priority": ctx.priority,
                "entities": ctx.entities,
                "from_cache": ctx.from_cache,
            }
        except Exception as e:
            logger.error("LLM interpret error: %s", e)
            return None

    def _route_compute_request(self, event: Event) -> None:
        """Перехватывает compute.request и диспатчит на TitanAgent."""
        if self._titan and self._titan.accept_task():
            try:
                self._titan._handle_compute_request(event)
            except Exception as e:
                logger.error("GaiaAgent: TitanAgent compute dispatch error: %s", e)
                self.publish_event(
                    event_type="compute.failed",
                    priority=EventPriority.HIGH,
                    payload={
                        "task_id": event.payload.get("task_id"),
                        "error": str(e),
                        "source": event.source,
                    },
                )
        else:
            logger.warning(
                "GaiaAgent: TitanAgent unavailable for compute.request task_id=%s",
                event.payload.get("task_id"),
            )
            self.publish_event(
                event_type="compute.failed",
                priority=EventPriority.NORMAL,
                payload={
                    "task_id": event.payload.get("task_id"),
                    "reason": "titan_unavailable",
                    "source": event.source,
                },
            )

    def _tool_pool_fallback(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Выполняет tool-задачу напрямую через ToolPool если Hephaestus недоступен."""
        tool_name = task.get("tool_name", "")
        params = task.get("params", {})

        if not tool_name:
            return {
                "success": False,
                "error": "tool_name missing in fallback",
                "task_id": task.get("task_id"),
                "fallback": True,
            }

        result = self._pool.execute(tool_name, params)
        logger.info(
            "ToolPool fallback: tool=%s success=%s", tool_name, result.success
        )
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
            "task_id": task.get("task_id"),
            "fallback": True,
        }

    def get_average_latency(self) -> float:
        if self._latency_count == 0:
            return 0.0
        return self._latency_sum / self._latency_count
