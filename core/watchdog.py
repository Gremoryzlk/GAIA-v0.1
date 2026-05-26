"""AgentWatchdog для GAIA v7.3 — мониторинг heartbeat агентов."""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional, TYPE_CHECKING

from core.safety.meta_controller import reset_agent

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent
    from core.safety.meta_controller import MetaController

logger = logging.getLogger(__name__)

_HEARTBEAT_TIMEOUT_SEC = 60
_MANUAL_INTERVENTION_THRESHOLD = 3


class AgentWatchdog:
    """Фоновый поток мониторинга агентов по heartbeat.

    Каждые check_interval секунд проверяет всех зарегистрированных агентов:
    - heartbeat старше 60 сек → increment_rogue_score
    - isolation_count == 1 → попытка auto-reset через MetaController
    - isolation_count >= 3 → CRITICAL лог, ручное вмешательство
    """

    def __init__(
        self,
        check_interval: int = 30,
        meta_controller: Optional["MetaController"] = None,
    ) -> None:
        self._check_interval = check_interval
        self._meta_controller = meta_controller
        self._agents: List["BaseAgent"] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def register(self, agent: "BaseAgent") -> None:
        with self._lock:
            if agent not in self._agents:
                self._agents.append(agent)
                logger.info("Watchdog registered agent: %s", agent.agent_id)

    def unregister(self, agent: "BaseAgent") -> None:
        with self._lock:
            if agent in self._agents:
                self._agents.remove(agent)

    def start(self) -> None:
        if self._running:
            logger.warning("Watchdog already running.")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._check_loop,
            name="gaia-watchdog",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Watchdog started — interval=%ds heartbeat_timeout=%ds auto_reset=%s",
            self._check_interval,
            _HEARTBEAT_TIMEOUT_SEC,
            self._meta_controller is not None,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Watchdog stopped.")

    def _check_loop(self) -> None:
        while self._running:
            time.sleep(self._check_interval)
            if not self._running:
                break
            self._check_all()

    def _check_all(self) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            agents = list(self._agents)
        for agent in agents:
            try:
                self._check_agent(agent, now)
            except Exception as e:
                logger.error("Watchdog error checking agent %s: %s", agent.agent_id, e)

    def _check_agent(self, agent: "BaseAgent", now: datetime) -> None:
        # Требует ручного вмешательства
        if agent.is_isolated() and agent._isolation_count >= _MANUAL_INTERVENTION_THRESHOLD:
            logger.critical(
                "Agent %s requires MANUAL INTERVENTION — "
                "isolation_count=%d rogue_score=%.1f",
                agent.agent_id, agent._isolation_count, agent.rogue_score,
            )
            return

        # Auto-reset при isolation_count == 1
        if agent.is_isolated() and agent._isolation_count == 1:
            if self._meta_controller is not None:
                try:
                    result = reset_agent(self._meta_controller, agent)
                    if result:
                        logger.info(
                            "Watchdog auto-reset agent %s successfully.",
                            agent.agent_id,
                        )
                    else:
                        logger.warning(
                            "Watchdog auto-reset skipped for %s — fitness not improving.",
                            agent.agent_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Watchdog auto-reset failed for %s: %s",
                        agent.agent_id, e,
                    )
            else:
                logger.warning(
                    "Agent %s isolated (count=1) but no MetaController — cannot auto-reset.",
                    agent.agent_id,
                )
            return

        # Пропускаем остальных изолированных
        if agent.is_isolated():
            return

        # Проверяем heartbeat
        heartbeat_age = (now - agent.last_heartbeat).total_seconds()
        if heartbeat_age > _HEARTBEAT_TIMEOUT_SEC:
            logger.warning(
                "Watchdog: agent %s heartbeat timeout — age=%.0fs",
                agent.agent_id, heartbeat_age,
            )
            agent.increment_rogue_score(0.1)
