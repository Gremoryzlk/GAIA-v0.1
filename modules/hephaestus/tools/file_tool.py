"""FileTool for secure file operations with workspace isolation."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base_tool import BaseTool, ToolResult
from shared.constants import TOOL_MAX_FILE_SIZE_BYTES, TOOL_WORKSPACE_DIR


class FileTool(BaseTool):
    """Secure file tool isolated to workspace directory."""

    def __init__(self, workspace_dir: Optional[str] = None) -> None:
        super().__init__(
            name="file",
            description="Secure file operations: read, write, list, exists",
        )
        self._workspace = Path(
            workspace_dir or TOOL_WORKSPACE_DIR
        ).resolve()
        self._workspace.mkdir(parents=True, exist_ok=True)

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        is_valid, error = self.validate_params(params)
        if not is_valid:
            return ToolResult(success=False, output="", error=error)

        action = params["action"].lower()
        try:
            if action == "read":
                return self._read_file(params)
            elif action == "write":
                return self._write_file(params)
            elif action == "list":
                return self._list_directory(params)
            elif action == "exists":
                return self._check_exists(params)
            else:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except PermissionError as e:
            return ToolResult(success=False, output="", error=f"Permission denied: {e}")
        except FileNotFoundError as e:
            return ToolResult(success=False, output="", error=f"File not found: {e}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Operation failed: {e}")

    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if not isinstance(params, dict):
            return False, "Parameters must be a dictionary"
        if "action" not in params:
            return False, "Missing required parameter: action"
        action = params["action"].lower()
        if action not in {"read", "write", "list", "exists"}:
            return False, f"Invalid action: {action}"
        if "path" not in params:
            return False, "Missing required parameter: path"
        if not isinstance(params["path"], str) or not params["path"].strip():
            return False, "Path must be a non-empty string"
        if not self._is_safe_path(params["path"]):
            return False, "Path not allowed: must be within workspace"
        if action == "write" and "content" not in params:
            return False, "Missing required parameter for write: content"
        return True, None

    def _is_safe_path(self, path: str) -> bool:
        """Проверяет что путь находится внутри workspace_dir."""
        try:
            # Строим абсолютный путь относительно workspace
            candidate = (self._workspace / path).resolve()
            # Путь должен начинаться с workspace
            return str(candidate).startswith(str(self._workspace))
        except (OSError, ValueError):
            return False

    def _resolve(self, path: str) -> Path:
        """Возвращает абсолютный путь внутри workspace."""
        return (self._workspace / path).resolve()

    def _read_file(self, params: Dict[str, Any]) -> ToolResult:
        full_path = self._resolve(params["path"])
        size = full_path.stat().st_size
        if size > TOOL_MAX_FILE_SIZE_BYTES:
            return ToolResult(
                success=False, output="",
                error=f"File too large: {size} bytes (max {TOOL_MAX_FILE_SIZE_BYTES})",
            )
        encoding = params.get("encoding", "utf-8")
        content = full_path.read_text(encoding=encoding)
        return ToolResult(
            success=True,
            output=content,
            metadata={"bytes_read": len(content.encode(encoding))},
        )

    def _write_file(self, params: Dict[str, Any]) -> ToolResult:
        full_path = self._resolve(params["path"])
        full_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = params.get("encoding", "utf-8")
        content = params["content"]
        full_path.write_text(content, encoding=encoding)
        return ToolResult(
            success=True,
            output=f"Written to {params['path']}",
            metadata={"bytes_written": len(content.encode(encoding))},
        )

    def _list_directory(self, params: Dict[str, Any]) -> ToolResult:
        full_path = self._resolve(params["path"])
        if not full_path.is_dir():
            return ToolResult(success=False, output="", error=f"Not a directory: {params['path']}")
        entries = []
        for entry in sorted(full_path.iterdir()):
            entries.append({
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "is_file": entry.is_file(),
                "size": entry.stat().st_size if entry.is_file() else None,
            })
        return ToolResult(
            success=True,
            output=json.dumps(entries),   # JSON вместо str(list)
            metadata={"entry_count": len(entries)},
        )

    def _check_exists(self, params: Dict[str, Any]) -> ToolResult:
        full_path = self._resolve(params["path"])
        exists = full_path.exists()
        result = {
            "exists": exists,
            "is_file": full_path.is_file() if exists else False,
            "is_dir": full_path.is_dir() if exists else False,
        }
        return ToolResult(success=True, output=json.dumps(result), metadata=result)