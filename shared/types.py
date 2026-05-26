"""GAIA v7.3 Shared Types."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventPriority(Enum):
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1


class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AgentStatus(Enum):
    IDLE = "IDLE"
    BUSY = "BUSY"
    ERROR = "ERROR"
    ISOLATED = "ISOLATED"


@dataclass
class Event:
    event_type: str
    priority: EventPriority
    source: str
    payload: Dict[str, Any]
    timestamp: Optional[datetime] = None
    message_id: Optional[str] = None
    signature: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def get_priority_value(self) -> int:
        return self.priority.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "priority": self.priority.name,
            "source": self.source,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "message_id": self.message_id,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        priority = (
            EventPriority[data["priority"]]
            if isinstance(data.get("priority"), str)
            else data.get("priority")
        )
        timestamp = (
            datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
        )
        return cls(
            event_type=data["event_type"],
            priority=priority,
            source=data["source"],
            payload=data["payload"],
            timestamp=timestamp,
            message_id=data.get("message_id"),
            signature=data.get("signature"),
        )


@dataclass
class Task:
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_at: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


@dataclass
class AgentContext:
    agent_id: str
    agent_type: str
    level: int
    status: AgentStatus = AgentStatus.IDLE
    tasks_processed: int = 0
    rogue_score: float = 0.0
    last_heartbeat: Optional[datetime] = None
    capabilities: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now(timezone.utc)


@dataclass
class AgentState:
    agent_id: str
    status: str
    tasks_processed: int
    rogue_score: float
    last_heartbeat: datetime


@dataclass
class DecisionTrace:
    decision_id: str
    rule_id: str
    confidence: float
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    timestamp: Optional[datetime] = None
    trace_steps: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class MemoryEntry:
    key: str
    value: Any
    tags: Optional[List[str]] = None
    ttl: Optional[int] = None
    created_at: Optional[datetime] = None
    provenance_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
