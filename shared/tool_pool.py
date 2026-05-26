"""Shared Tool Pool для GAIA v7.3.

Singleton реестр инструментов — инструменты живут независимо от агентов.
При падении агента инструменты остаются доступны через пул.
"""

import logging
import threading
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.hephaestus.tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolPool:
    """Thread-safe singleton реестр инструментов.

    Инструменты регистрируются один раз при старте системы.
    Любой агент или GaiaAgent может получить инструмент по имени.
    Инструменты stateless — можно вызывать параллельно.
    """

    _instance: Optional["ToolPool"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ToolPool":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._tools: Dict[str, "BaseTool"] = {}
        self._tools_lock = threading.Lock()
        self._initialized = True

    def register(self, tool: "BaseTool") -> None:
        """Регистрирует инструмент в пуле."""
        with self._tools_lock:
            if tool.name in self._tools:
                logger.warning(
                    "ToolPool: tool '%s' already registered — overwriting.", tool.name
                )
            self._tools[tool.name] = tool
            logger.info("ToolPool: registered tool '%s'.", tool.name)

    def get(self, name: str) -> Optional["BaseTool"]:
        """Возвращает инструмент по имени или None."""
        with self._tools_lock:
            return self._tools.get(name)

    def execute(self, tool_name: str, params: Dict) -> "ToolResult":
        """Выполняет инструмент напрямую из пула.

        Используется GaiaAgent как fallback когда Hephaestus недоступен.
        """
        from modules.hephaestus.tools.base_tool import ToolResult

        tool = self.get(tool_name)
        if tool is None:
            logger.error("ToolPool: tool '%s' not found.", tool_name)
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' not found in pool.",
            )

        logger.info(
            "ToolPool: executing tool '%s' directly (fallback).", tool_name
        )
        return tool.execute(params)

    def list_tools(self) -> List[str]:
        """Возвращает список зарегистрированных инструментов."""
        with self._tools_lock:
            return sorted(self._tools.keys())

    def unregister(self, name: str) -> None:
        """Удаляет инструмент из пула."""
        with self._tools_lock:
            if name in self._tools:
                del self._tools[name]
                logger.info("ToolPool: unregistered tool '%s'.", name)

    def discover_from_contract(
        self, contract_path: str, agent_callable_map: dict
    ) -> list:
        """Регистрирует инструменты из contract.json.

        contract.json формат:
          {"tools": [{"name": "tool_name", "description": "..."}]}
        agent_callable_map: {"tool_name": BaseTool instance}
        Возвращает список зарегистрированных имён.
        """
        import json
        from pathlib import Path
        registered = []
        try:
            data = json.loads(Path(contract_path).read_text(encoding="utf-8"))
            for tool_def in data.get("tools", []):
                name = tool_def.get("name")
                if name and name in agent_callable_map:
                    self.register(agent_callable_map[name])
                    registered.append(name)
            if registered:
                logger.info(
                    "ToolPool: discovered %d tools from %s: %s",
                    len(registered), contract_path, registered,
                )
        except Exception as e:
            logger.debug("ToolPool: contract discovery skipped (%s): %s", contract_path, e)
        return registered

    @classmethod
    def reset(cls) -> None:
        """Сбрасывает singleton — только для тестов."""
        with cls._lock:
            cls._instance = None
