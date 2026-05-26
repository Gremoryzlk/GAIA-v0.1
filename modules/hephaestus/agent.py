"""HephaestusAgent (L1) — агент выполнения инструментов для GAIA v7.3."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from agents.base_agent import BaseAgent
from core.eventbus.event import Event, EventPriority
from core.eventbus.eventbus import EventBus
from core.memory.memory_broker import MemoryBroker
from modules.hephaestus.tools.hephaestus import HephaestusTool
from modules.hephaestus.tools.file_tool import FileTool
from modules.hephaestus.tools.memory_tool import MemoryTool
from shared.tool_pool import ToolPool

logger = logging.getLogger(__name__)


class HephaestusAgent(BaseAgent):
    """L1 агент — выполняет инструменты по заданиям от GaiaAgent."""

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        memory_broker: Optional[MemoryBroker] = None,
    ) -> None:
        super().__init__(agent_type="hephaestus_l1", level=1)
        self._workspace_dir = workspace_dir or "modules/hephaestus/data/workspace"
        self._broker = memory_broker or MemoryBroker()
        self._registry = HephaestusTool()

        # Создаём инструменты
        file_tool = FileTool(self._workspace_dir)
        memory_tool = MemoryTool(self._broker, self.agent_id)

        # Регистрируем в локальном реестре агента
        self._registry.register(file_tool)
        self._registry.register(memory_tool)

        # Регистрируем в глобальном ToolPool — доступны при fallback
        pool = ToolPool()
        pool.register(file_tool)
        pool.register(memory_tool)

        logger.info(
            "HephaestusAgent %s initialized. Tools: %s",
            self.agent_id,
            self._registry.list_tools(),
        )

    def _subscribe_to_events(self) -> None:
        self.eventbus.subscribe(
            name=f"{self.agent_id}_subscriber",
            event_types=["tool.request", "tool.execute"],
            callback=self._handle_event,
        )

    def _handle_event(self, event: Event) -> None:
        try:
            payload = event.payload
            tool_name = payload.get("tool_name")
            params = payload.get("params", {})
            task_id = payload.get("task_id", str(uuid4()))

            if not tool_name:
                self.publish_event(
                    event_type="tool.result",
                    priority=EventPriority.NORMAL,
                    payload={"success": False, "error": "tool_name required", "task_id": task_id},
                )
                return

            self.mark_busy()
            result = self._registry.execute(tool_name, params)
            self.publish_event(
                event_type="tool.result",
                priority=EventPriority.NORMAL,
                payload={
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                    "task_id": task_id,
                    "metadata": result.metadata,
                },
            )
        except Exception as e:
            logger.error("HephaestusAgent event error: %s", e)
        finally:
            self.mark_idle()

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.mark_busy()
        try:
            result = self._registry.process_task(task)
            self.tasks_processed += 1
            outcome = {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "metadata": result.metadata,
                "agent_id": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.publish_event(
                event_type="task.completed" if result.success else "task.failed",
                priority=EventPriority.NORMAL,
                payload={"task_id": task.get("task_id"), "result": outcome},
            )
            return outcome
        except Exception as e:
            logger.error("HephaestusAgent task error: %s", e)
            raise
        finally:
            self.mark_idle()

    def register_tool(self, tool: Any) -> None:
        self._registry.register(tool)

    def list_tools(self) -> list:
        return self._registry.list_tools()

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["workspace_dir"] = self._workspace_dir
        state["registered_tools"] = self._registry.list_tools()
        return state
