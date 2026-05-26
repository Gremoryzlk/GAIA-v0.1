"""AegisAgent (L1) — безопасность и валидация для GAIA v7.3."""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from agents.base_agent import BaseAgent
from core.eventbus.event import Event, EventPriority
from modules.hephaestus.tools.base_tool import BaseTool, ToolResult
from shared.tool_pool import ToolPool

logger = logging.getLogger(__name__)

# Паттерны опасного контента
_DANGEROUS_PATTERNS: List[str] = [
    r"\.\./",              # path traversal
    r";\s*rm\s+-",         # shell injection
    r"<script",            # XSS
    r"DROP\s+TABLE",       # SQL injection
    r"__import__",         # Python injection
    r"eval\s*\(",          # eval injection
]

_MAX_PAYLOAD_SIZE = 64 * 1024  # 64 KB


class _ValidateTool(BaseTool):
    """Инструмент валидации входных данных."""

    def __init__(self) -> None:
        super().__init__(
            name="validate_input",
            description="Validate and sanitize input payload",
        )
        self._patterns = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS_PATTERNS]

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        payload = params.get("payload", {})
        payload_str = json.dumps(payload) if isinstance(payload, dict) else str(payload)

        # Проверка размера
        if len(payload_str.encode()) > _MAX_PAYLOAD_SIZE:
            return ToolResult(
                success=False,
                output="",
                error=f"Payload too large: {len(payload_str)} bytes > {_MAX_PAYLOAD_SIZE}",
                metadata={"violation": "size_limit"},
            )

        # Проверка опасных паттернов
        for pattern in self._patterns:
            match = pattern.search(payload_str)
            if match:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Dangerous pattern detected: {match.group()}",
                    metadata={"violation": "dangerous_pattern", "pattern": match.group()},
                )

        return ToolResult(
            success=True,
            output=payload_str,
            metadata={"valid": True, "size_bytes": len(payload_str.encode())},
        )

    def validate_params(self, params: Dict[str, Any]):
        return True, None


class _SignatureTool(BaseTool):
    """Инструмент верификации HMAC-подписей событий."""

    def __init__(self) -> None:
        super().__init__(
            name="verify_signature",
            description="Verify HMAC signature of an event",
        )

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        try:
            from shared.logging import verify_event, sign_event
            from shared.types import Event, EventPriority

            # Если передан event dict — верифицируем
            event_data = params.get("event")
            if not event_data:
                return ToolResult(
                    success=False, output="", error="No event provided"
                )

            event = Event.from_dict(event_data) if isinstance(event_data, dict) else event_data
            valid = verify_event(event)

            return ToolResult(
                success=True,
                output=json.dumps({"valid": valid}),
                metadata={"valid": valid, "event_type": event.event_type},
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def validate_params(self, params: Dict[str, Any]):
        return True, None


class AegisAgent(BaseAgent):
    """L1 агент безопасности — валидация, фильтрация, верификация подписей."""

    def __init__(self) -> None:
        super().__init__(agent_type="aegis_l1", level=1)
        self._violations: List[Dict[str, Any]] = []
        self._validate_tool = _ValidateTool()
        self._signature_tool = _SignatureTool()

        # Регистрируем в ToolPool
        pool = ToolPool()
        pool.register(self._validate_tool)
        pool.register(self._signature_tool)

        logger.info("AegisAgent %s initialized.", self.agent_id)

    def _subscribe_to_events(self) -> None:
        self.eventbus.subscribe(
            name=f"{self.agent_id}_subscriber",
            event_types=["security.validate", "security.verify", "agent.isolated"],
            callback=self._handle_event,
        )

    def _handle_event(self, event: Event) -> None:
        try:
            if event.event_type == "security.validate":
                result = self._validate_tool.execute(event.payload)
                if not result.success:
                    self._record_violation(event, result.error)
                    self.publish_event(
                        event_type="security.violation",
                        priority=EventPriority.HIGH,
                        payload={
                            "source": event.source,
                            "violation": result.metadata.get("violation"),
                            "error": result.error,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )

            elif event.event_type == "security.verify":
                result = self._signature_tool.execute(event.payload)
                if not result.success or not json.loads(result.output).get("valid"):
                    self._record_violation(event, "invalid_signature")
                    self.publish_event(
                        event_type="security.violation",
                        priority=EventPriority.CRITICAL,
                        payload={
                            "source": event.source,
                            "violation": "invalid_signature",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )

            elif event.event_type == "agent.isolated":
                logger.warning(
                    "Aegis: agent isolated — %s reason=rogue_score=%.1f",
                    event.payload.get("agent_id"),
                    event.payload.get("rogue_score", 0),
                )
        except Exception as e:
            logger.error("AegisAgent event error: %s", e)

    def _record_violation(self, event: Event, reason: str) -> None:
        self._violations.append({
            "event_type": event.event_type,
            "source": event.source,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        logger.warning(
            "Aegis violation: source=%s reason=%s", event.source, reason
        )

    def validate(self, payload: Any) -> Tuple[bool, Optional[str]]:
        """Публичный метод валидации — для прямого вызова."""
        result = self._validate_tool.execute({"payload": payload})
        return result.success, result.error

    def get_violations(self) -> List[Dict[str, Any]]:
        """Возвращает историю нарушений."""
        return list(self._violations)

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.mark_busy()
        try:
            action = task.get("action", "validate")
            if action == "validate":
                result = self._validate_tool.execute(task)
            elif action == "verify":
                result = self._signature_tool.execute(task)
            else:
                result = self._validate_tool.execute(task)

            self.tasks_processed += 1
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "action": action,
                "agent_id": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error("AegisAgent task error: %s", e)
            raise
        finally:
            self.mark_idle()
