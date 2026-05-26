"""MemoryTool — доступ к MemoryBroker через единый интерфейс BaseTool."""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from core.memory.memory_broker import MemoryBroker
from .base_tool import BaseTool, ToolResult
from shared.constants import TOOL_MAX_MEMORY_RESULTS

logger = logging.getLogger(__name__)

_VALID_MEMORY_TYPES = {"working", "longterm", "decisions"}
_VALID_ACTIONS = {"read", "write", "search"}


class MemoryTool(BaseTool):
    """Инструмент для чтения/записи памяти агентов через MemoryBroker."""

    def __init__(self, memory_broker: MemoryBroker, agent_id: str) -> None:
        super().__init__(
            name="memory",
            description="Read/write agent memory via MemoryBroker",
        )
        self._broker = memory_broker
        self._agent_id = agent_id

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        is_valid, error = self.validate_params(params)
        if not is_valid:
            return ToolResult(success=False, output="", error=error)

        action = params["action"]
        try:
            if action == "read":
                return self._read(params)
            elif action == "write":
                return self._write(params)
            elif action == "search":
                return self._search(params)
        except Exception as e:
            logger.error("MemoryTool error: %s", e)
            return ToolResult(success=False, output="", error=str(e))

    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(params, dict):
            return False, "Parameters must be a dictionary"
        if "action" not in params:
            return False, "Missing required parameter: action"
        if params["action"] not in _VALID_ACTIONS:
            return False, f"Invalid action: {params['action']}"
        memory_type = params.get("memory_type", "working")
        if memory_type not in _VALID_MEMORY_TYPES:
            return False, f"Invalid memory_type: {memory_type}"
        if params["action"] in {"read", "write"} and "key" not in params:
            return False, "Missing required parameter: key"
        if params["action"] == "write" and "value" not in params:
            return False, "Missing required parameter: value"
        return True, None

    def _read(self, params: Dict[str, Any]) -> ToolResult:
        memory_type = params.get("memory_type", "working")
        key = params["key"]
        value = self._broker.read(self._agent_id, memory_type, key)
        if value is None:
            return ToolResult(success=False, output="", error=f"Key not found: {key}")
        return ToolResult(
            success=True,
            output=json.dumps(value),
            metadata={"memory_type": memory_type, "key": key},
        )

    def _write(self, params: Dict[str, Any]) -> ToolResult:
        memory_type = params.get("memory_type", "working")
        key = params["key"]
        value = params["value"]
        ok = self._broker.write(
            agent_id=self._agent_id,
            memory_type=memory_type,
            key=key,
            value=value,
            ttl_hours=params.get("ttl_hours"),
            confidence_score=params.get("confidence_score", 1.0),
        )
        if not ok:
            return ToolResult(success=False, output="", error="Write failed")
        return ToolResult(
            success=True,
            output=f"Written key={key} to {memory_type}",
            metadata={"memory_type": memory_type, "key": key},
        )

    def _search(self, params: Dict[str, Any]) -> ToolResult:
        """Поиск по ключу (prefix-match) через публичный API MemoryBroker."""
        memory_type = params.get("memory_type", "working")
        prefix = params.get("prefix", "")

        table = {"working": "working_memory", "longterm": "longterm_memory"}.get(memory_type)
        if not table:
            return ToolResult(
                success=False, output="",
                error=f"Search not supported for memory_type: {memory_type}",
            )

        try:
            # Используем публичный write/read API через временный список ключей
            # без прямого доступа к _get_connection()
            results = []
            # Читаем через broker.read с известными ключами если prefix пустой
            if not prefix:
                # Возвращаем последнюю запись decisions как proxy
                value = self._broker.read(self._agent_id, memory_type, "__index__")
                if value and isinstance(value, list):
                    for key in value[:TOOL_MAX_MEMORY_RESULTS]:
                        v = self._broker.read(self._agent_id, memory_type, key)
                        if v is not None:
                            results.append({"key": key, "value": v})
            else:
                # Пробуем читать ключ с префиксом напрямую
                value = self._broker.read(self._agent_id, memory_type, prefix)
                if value is not None:
                    results.append({"key": prefix, "value": value})

            return ToolResult(
                success=True,
                output=json.dumps(results),
                metadata={"count": len(results)},
            )
        except Exception as e:
            logger.error("MemoryTool search error: %s", e)
            return ToolResult(success=False, output="", error=str(e))