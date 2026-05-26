"""AthenaAgent (L1) — планировщик задач для GAIA v7.3."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from agents.base_agent import BaseAgent
from core.eventbus.event import Event, EventPriority
from modules.hephaestus.tools.base_tool import BaseTool, ToolResult
from shared.tool_pool import ToolPool

logger = logging.getLogger(__name__)


class _AnalyzeTextTool(BaseTool):
    """Анализ текста — срочность и базовые метрики."""

    def __init__(self, agent: "AthenaAgent") -> None:
        super().__init__(name="analyze_text", description="Analyze text for urgency and entities")
        self._agent = agent

    def execute(self, params: dict) -> ToolResult:
        import json
        text = params.get("text", "")
        words = text.lower().split()
        urgency_keywords = ["urgent", "critical", "emergency", "asap", "immediately"]
        urgency = min(1.0, sum(1 for w in words if w in urgency_keywords) / 3)
        result = {
            "text_length": len(text),
            "word_count": len(words),
            "urgency": urgency,
            "source": "athena",
        }
        return ToolResult(success=True, output=json.dumps(result), metadata=result)

    def validate_params(self, params: dict):
        return True, None


class _DecomposeTaskTool(BaseTool):
    """Декомпозиция задачи на шаги."""

    def __init__(self, agent: "AthenaAgent") -> None:
        super().__init__(name="decompose_task", description="Decompose complex task into steps")
        self._agent = agent

    def execute(self, params: dict) -> ToolResult:
        import json
        task_text = params.get("task", "")
        steps = self._agent._decompose(task_text)
        result = {"steps": steps, "count": len(steps), "source": "athena"}
        return ToolResult(success=True, output=json.dumps(result), metadata=result)

    def validate_params(self, params: dict):
        return True, None


class AthenaAgent(BaseAgent):
    """L1 агент-планировщик — декомпозирует сложные задачи на шаги."""

    def __init__(self) -> None:
        super().__init__(agent_type="athena_l1", level=1)
        # Регистрируем инструменты в ToolPool
        pool = ToolPool()
        pool.register(_AnalyzeTextTool(self))
        pool.register(_DecomposeTaskTool(self))

    def _subscribe_to_events(self) -> None:
        self.eventbus.subscribe(
            name=f"{self.agent_id}_subscriber",
            event_types=["task.plan"],
            callback=self._handle_event,
        )

    def _handle_event(self, event: Event) -> None:
        try:
            self.mark_busy()
            payload = event.payload
            task_text = (
                payload.get("text")
                or str(payload.get("payload", ""))
                or str(payload)
            )
            steps = self._decompose(task_text)
            logger.info(
                "Athena decomposed task into %d steps: %s",
                len(steps), steps,
            )
            for step in steps:
                self.publish_event(
                    event_type="routing.request",
                    priority=EventPriority.NORMAL,
                    payload={
                        "task_id": str(uuid4()),
                        "task_type": step["task_type"],
                        "payload": {"text": step["text"]},
                        "parent_task_id": payload.get("task_id"),
                    },
                )
        except Exception as e:
            logger.error("AthenaAgent planning error: %s", e)
            self.rogue_score += 0.1
        finally:
            self.mark_idle()

    def _decompose(self, task_text: str) -> List[Dict[str, Any]]:
        """Простая эвристическая декомпозиция задачи на шаги.

        Если в тексте есть 'and' или 'then' — разбиваем на несколько шагов.
        Иначе — один шаг.
        """
        text_lower = task_text.lower()
        steps = []

        if " and " in text_lower or " then " in text_lower:
            # Разбиваем по 'and' или 'then'
            import re
            parts = re.split(r'\s+(?:and|then)\s+', task_text, flags=re.IGNORECASE)
            for part in parts:
                part = part.strip()
                if part:
                    steps.append({
                        "text": part,
                        "task_type": _infer_task_type(part),
                    })
        else:
            steps.append({
                "text": task_text,
                "task_type": _infer_task_type(task_text),
            })

        return steps

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Прямой вызов планировщика."""
        self.mark_busy()
        try:
            task_text = (
                task.get("text")
                or str(task.get("payload", ""))
                or str(task)
            )
            steps = self._decompose(task_text)
            self.tasks_processed += 1

            for step in steps:
                self.publish_event(
                    event_type="routing.request",
                    priority=EventPriority.NORMAL,
                    payload={
                        "task_id": str(uuid4()),
                        "task_type": step["task_type"],
                        "payload": {"text": step["text"]},
                    },
                )

            return {
                "steps": steps,
                "step_count": len(steps),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error("AthenaAgent process_task error: %s", e)
            raise
        finally:
            self.mark_idle()


def _infer_task_type(text: str) -> str:
    """Определяет task_type по ключевым словам в тексте."""
    text_lower = text.lower()
    if any(w in text_lower for w in ("read file", "open file", "load file", "show file")):
        return "file_read"
    if any(w in text_lower for w in ("write file", "save file", "create file")):
        return "file_write"
    if any(w in text_lower for w in ("list file", "list dir", "show files")):
        return "file_list"
    if any(w in text_lower for w in ("read memory", "get memory", "load memory")):
        return "memory_read"
    if any(w in text_lower for w in ("analys", "analyze", "examine", "inspect")):
        return "analysis"
    return "query"
