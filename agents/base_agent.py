"""BaseAgent — abstract base for all GAIA agents."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from core.eventbus.event import Event, EventPriority
from core.eventbus.eventbus import EventBus
from shared.constants import ROGUE_SCORE_ISOLATE
from shared.types import AgentStatus

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, agent_type: str, level: int) -> None:
        self.agent_id = f"{agent_type}_{uuid4().hex[:8]}"
        self.agent_type = agent_type
        self.level = level
        self.status: AgentStatus = AgentStatus.IDLE
        self.tasks_processed: int = 0
        self.rogue_score: float = 0.0
        self.last_heartbeat: datetime = datetime.now(timezone.utc)
        self._isolation_count: int = 0
        self.eventbus = EventBus()
        self._subscribe_to_events()

    @abstractmethod
    def _subscribe_to_events(self) -> None:
        pass

    @abstractmethod
    def process_task(self, task: Dict[str, Any]) -> Any:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "tasks_processed": self.tasks_processed,
            "rogue_score": self.rogue_score,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "isolation_count": self._isolation_count,
        }

    def is_isolated(self) -> bool:
        return self.status == AgentStatus.ISOLATED

    def mark_busy(self) -> None:
        if self.is_isolated():
            logger.warning("Agent %s is ISOLATED — ignoring mark_busy.", self.agent_id)
            return
        self.status = AgentStatus.BUSY

    def mark_idle(self) -> None:
        if self.is_isolated():
            return
        self.status = AgentStatus.IDLE
        self.last_heartbeat = datetime.now(timezone.utc)

    def increment_rogue_score(self, delta: float = 0.1) -> None:
        self.rogue_score += delta
        if self.rogue_score >= ROGUE_SCORE_ISOLATE and not self.is_isolated():
            self._isolate()

    def _isolate(self) -> None:
        self.status = AgentStatus.ISOLATED
        self._isolation_count += 1
        logger.error(
            "Agent %s ISOLATED — rogue_score=%.1f >= threshold=%d isolation_count=%d",
            self.agent_id, self.rogue_score, ROGUE_SCORE_ISOLATE, self._isolation_count,
        )
        self.publish_event(
            event_type="agent.isolated",
            priority=EventPriority.CRITICAL,
            payload={
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "rogue_score": self.rogue_score,
                "threshold": ROGUE_SCORE_ISOLATE,
                "isolation_count": self._isolation_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def reset_isolation(self, authorized_by: str) -> bool:
        """Восстанавливает агента из ISOLATED.

        Только MetaController может авторизовать восстановление.
        """
        if authorized_by != "meta_controller":
            logger.warning(
                "Agent %s reset_isolation rejected — unauthorized: %s",
                self.agent_id, authorized_by,
            )
            return False

        self.rogue_score = 0.0
        self.status = AgentStatus.IDLE
        self.last_heartbeat = datetime.now(timezone.utc)

        logger.info(
            "Agent %s restored from ISOLATED by %s isolation_count=%d",
            self.agent_id, authorized_by, self._isolation_count,
        )
        self.publish_event(
            event_type="agent.restored",
            priority=EventPriority.CRITICAL,
            payload={
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "authorized_by": authorized_by,
                "isolation_count": self._isolation_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return True

    def accept_task(self) -> bool:
        if self.is_isolated():
            logger.warning(
                "Agent %s rejected task — status=ISOLATED rogue_score=%.1f",
                self.agent_id, self.rogue_score,
            )
            self.publish_event(
                event_type="agent.rejected",
                priority=EventPriority.HIGH,
                payload={
                    "agent_id": self.agent_id,
                    "reason": "isolated",
                    "rogue_score": self.rogue_score,
                },
            )
            return False
        return True

    def publish_event(
        self,
        event_type: str,
        priority: EventPriority,
        payload: Dict[str, Any],
    ) -> None:
        self.eventbus.publish(
            Event(
                event_type=event_type,
                priority=priority,
                source=self.agent_id,
                payload=payload,
            )
        )
