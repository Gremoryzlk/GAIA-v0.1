"""HermesAgent (L1) — маршрутизатор между агентами для GAIA v7.3."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from agents.base_agent import BaseAgent
from core.eventbus.event import Event, EventPriority
from modules.hephaestus.tools.base_tool import BaseTool, ToolResult
from shared.tool_pool import ToolPool

logger = logging.getLogger(__name__)

_ROUTING_RULES: Dict[str, str] = {
    "file_read":   "tool.execute",
    "file_write":  "tool.execute",
    "file_list":   "tool.execute",
    "memory_read": "tool.execute",
    "analysis":    "task.assigned",
    "query":       "task.assigned",
}


class _WebSearchTool(BaseTool):
    """Stub web search tool — заглушка до реальной реализации."""

    def __init__(self) -> None:
        super().__init__(name="web_search", description="Search the web for information")

    def execute(self, params: dict) -> ToolResult:
        query = params.get("query", "")
        return ToolResult(
            success=True,
            output="[]",
            metadata={"query": query, "source": "hermes_stub", "results": []},
        )

    def validate_params(self, params: dict):
        return True, None


class _RssFetchTool(BaseTool):
    """Stub RSS fetch tool — заглушка до реальной реализации."""

    def __init__(self) -> None:
        super().__init__(name="rss_fetch", description="Fetch RSS feed items from URL")

    def execute(self, params: dict) -> ToolResult:
        url = params.get("url", "")
        return ToolResult(
            success=True,
            output="[]",
            metadata={"url": url, "source": "hermes_stub", "items": []},
        )

    def validate_params(self, params: dict):
        return True, None


class HermesAgent(BaseAgent):
    """L1 агент-маршрутизатор."""

    def __init__(self) -> None:
        super().__init__(agent_type="hermes_l1", level=1)
        # Регистрируем инструменты в ToolPool
        pool = ToolPool()
        pool.register(_WebSearchTool())
        pool.register(_RssFetchTool())

    def _subscribe_to_events(self) -> None:
        self.eventbus.subscribe(
            name=f"{self.agent_id}_subscriber",
            event_types=["routing.request"],
            callback=self._handle_event,
        )

    def _handle_event(self, event: Event) -> None:
        try:
            self.mark_busy()
            self._route(event)
        except Exception as e:
            logger.error("HermesAgent routing error: %s", e)
            self.increment_rogue_score(0.1)
        finally:
            self.mark_idle()

    def _route(self, event: Event) -> None:
        """Определяет целевой агент и публикует событие."""
        payload = event.payload
        task_type = payload.get("task_type", "unknown")
        route_payload = payload.get("payload", payload)
        task_id = payload.get("task_id", str(uuid4()))
        event_type = _ROUTING_RULES.get(task_type, "task.assigned")

        try:
            if event_type == "tool.execute":
                tool = _task_type_to_tool(task_type)
                operation = _task_type_to_operation(task_type)
                out_payload = {
                    "task_id": task_id,
                    "tool_name": tool,
                    "params": {
                        "action": operation,
                        **({k: v for k, v in route_payload.items()
                            if k not in ("task_type", "task_id")}),
                    },
                }
            else:
                out_payload = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "payload": route_payload,
                    "assigned_agent": "gaia_l0",
                }

            self.publish_event(
                event_type=event_type,
                priority=EventPriority.NORMAL,
                payload=out_payload,
            )
            logger.info(
                "Hermes routed task_type=%s → event=%s task_id=%s",
                task_type, event_type, task_id,
            )

        except Exception as e:
            logger.error(
                "Hermes routing failed for task_type=%s task_id=%s: %s",
                task_type, task_id, e,
            )
            self.publish_event(
                event_type="routing.error",
                priority=EventPriority.HIGH,
                payload={
                    "task_id": task_id,
                    "task_type": task_type,
                    "error": str(e),
                    "attempted_route": event_type,
                },
            )
            self.increment_rogue_score(0.1)

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.mark_busy()
        try:
            task_type = task.get("task_type", "unknown")
            event_type = _ROUTING_RULES.get(task_type, "task.assigned")
            task_id = task.get("task_id", str(uuid4()))
            self.publish_event(
                event_type=event_type,
                priority=EventPriority.NORMAL,
                payload={**task, "task_id": task_id},
            )
            self.tasks_processed += 1
            return {
                "routed_to": event_type,
                "task_type": task_type,
                "task_id": task_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error("HermesAgent process_task error: %s", e)
            raise
        finally:
            self.mark_idle()


def _task_type_to_tool(task_type: str) -> str:
    if task_type.startswith("file"):
        return "file"
    if task_type.startswith("memory"):
        return "memory"
    return "file"


def _task_type_to_operation(task_type: str) -> str:
    return {
        "file_read":   "read",
        "file_write":  "write",
        "file_list":   "list",
        "memory_read": "read",
    }.get(task_type, "read")
