"""HephaestusTool — реестр инструментов с thread-safe выполнением."""

import logging
import threading
from typing import Dict, List, Optional

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class HephaestusTool:
    """Реестр и диспетчер инструментов для HephaestusAgent."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._lock = threading.Lock()

    def register(self, tool: BaseTool) -> None:
        with self._lock:
            if tool.name in self._tools:
                logger.warning("Tool '%s' already registered — overwriting.", tool.name)
            self._tools[tool.name] = tool
            logger.info("Tool registered: %s", tool.name)

    def unregister(self, name: str) -> None:
        with self._lock:
            if name in self._tools:
                del self._tools[name]

    def execute(self, tool_name: str, params: dict) -> ToolResult:
        with self._lock:
            tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False, output="",
                error=f"Tool not found: '{tool_name}'. Available: {self.list_tools()}",
            )
        logger.debug("Executing tool '%s' with params=%s", tool_name, list(params.keys()))
        return tool.execute(params)

    def process_task(self, task: dict) -> ToolResult:
        tool_name = task.get("tool_name")
        params = task.get("params")
        if not tool_name:
            return ToolResult(success=False, output="", error="Missing tool_name")
        if not isinstance(params, dict):
            return ToolResult(success=False, output="", error="params must be a dict")
        return self.execute(tool_name, params)

    def list_tools(self) -> List[str]:
        with self._lock:
            return sorted(self._tools.keys())

    def get_tool(self, name: str) -> Optional[BaseTool]:
        with self._lock:
            return self._tools.get(name)