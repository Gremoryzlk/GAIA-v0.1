from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate parameters before execution.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass