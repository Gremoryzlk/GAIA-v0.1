"""PrometheusAgent (L1) — мониторинг системы и метрики для GAIA v7.3."""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import psutil

from agents.base_agent import BaseAgent
from core.eventbus.event import Event, EventPriority
from modules.hephaestus.tools.base_tool import BaseTool, ToolResult
from shared.tool_pool import ToolPool

logger = logging.getLogger(__name__)

_COLLECT_INTERVAL_SEC = 30   # сбор метрик каждые 30 сек
_CPU_WARN_THRESHOLD = 80.0   # % CPU
_RAM_WARN_THRESHOLD = 85.0   # % RAM


class _SystemMetricsTool(BaseTool):
    """Инструмент сбора системных метрик."""

    def __init__(self) -> None:
        super().__init__(
            name="system_metrics",
            description="Collect CPU, RAM, disk metrics",
        )

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        import json
        try:
            metrics = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "ram_percent": psutil.virtual_memory().percent,
                "ram_used_mb": psutil.virtual_memory().used // (1024 * 1024),
                "ram_total_mb": psutil.virtual_memory().total // (1024 * 1024),
                "disk_percent": psutil.disk_usage("/").percent,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return ToolResult(
                success=True,
                output=json.dumps(metrics),
                metadata=metrics,
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    def validate_params(self, params: Dict[str, Any]):
        return True, None


class _AgentHealthTool(BaseTool):
    """Инструмент проверки здоровья агентов."""

    def __init__(self, agent_registry: Dict[str, BaseAgent]) -> None:
        super().__init__(
            name="agent_health",
            description="Check health status of all registered agents",
        )
        self._registry = agent_registry

    def execute(self, params: Dict[str, Any]) -> ToolResult:
        import json
        health = {}
        for name, agent in self._registry.items():
            health[name] = {
                "status": agent.status.value,
                "rogue_score": agent.rogue_score,
                "tasks_processed": agent.tasks_processed,
                "isolation_count": agent._isolation_count,
                "last_heartbeat": agent.last_heartbeat.isoformat(),
            }
        return ToolResult(
            success=True,
            output=json.dumps(health),
            metadata={"agent_count": len(health)},
        )

    def validate_params(self, params: Dict[str, Any]):
        return True, None


class PrometheusAgent(BaseAgent):
    """L1 агент мониторинга — собирает метрики и публикует алерты."""

    def __init__(self) -> None:
        super().__init__(agent_type="prometheus_l1", level=1)
        self._agent_registry: Dict[str, BaseAgent] = {}
        self._metrics_thread: Optional[threading.Thread] = None
        self._collecting = False

        # Инструменты
        self._metrics_tool = _SystemMetricsTool()
        self._health_tool = _AgentHealthTool(self._agent_registry)

        # Регистрируем в ToolPool
        pool = ToolPool()
        pool.register(self._metrics_tool)
        pool.register(self._health_tool)

        logger.info("PrometheusAgent %s initialized.", self.agent_id)

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Регистрирует агента для мониторинга."""
        self._agent_registry[name] = agent
        logger.info("Prometheus: monitoring agent '%s'.", name)

    def _subscribe_to_events(self) -> None:
        self.eventbus.subscribe(
            name=f"{self.agent_id}_subscriber",
            event_types=["metrics.request", "agent.isolated", "agent.restored"],
            callback=self._handle_event,
        )

    def _handle_event(self, event: Event) -> None:
        try:
            if event.event_type == "metrics.request":
                self._publish_metrics()
            elif event.event_type == "agent.isolated":
                logger.warning(
                    "Prometheus: agent isolated — %s rogue_score=%.1f",
                    event.payload.get("agent_id"),
                    event.payload.get("rogue_score", 0),
                )
            elif event.event_type == "agent.restored":
                logger.info(
                    "Prometheus: agent restored — %s",
                    event.payload.get("agent_id"),
                )
        except Exception as e:
            logger.error("PrometheusAgent event error: %s", e)

    def _publish_metrics(self) -> None:
        """Собирает и публикует системные метрики."""
        result = self._metrics_tool.execute({})
        if not result.success:
            return

        metrics = result.metadata
        cpu = metrics.get("cpu_percent", 0)
        ram = metrics.get("ram_percent", 0)

        priority = EventPriority.NORMAL
        if cpu > _CPU_WARN_THRESHOLD or ram > _RAM_WARN_THRESHOLD:
            priority = EventPriority.HIGH
            logger.warning(
                "Prometheus ALERT: cpu=%.1f%% ram=%.1f%%", cpu, ram
            )

        self.publish_event(
            event_type="metrics.system",
            priority=priority,
            payload={
                **metrics,
                "alert": cpu > _CPU_WARN_THRESHOLD or ram > _RAM_WARN_THRESHOLD,
            },
        )

    def start_collection(self) -> None:
        """Запускает фоновый сбор метрик."""
        if self._collecting:
            return
        self._collecting = True
        self._metrics_thread = threading.Thread(
            target=self._collection_loop,
            name="gaia-prometheus",
            daemon=True,
        )
        self._metrics_thread.start()
        logger.info("Prometheus: metrics collection started.")

    def stop_collection(self) -> None:
        """Останавливает сбор метрик."""
        self._collecting = False
        logger.info("Prometheus: metrics collection stopped.")

    def _collection_loop(self) -> None:
        while self._collecting:
            try:
                self._publish_metrics()
            except Exception as e:
                logger.error("Prometheus collection error: %s", e)
            time.sleep(_COLLECT_INTERVAL_SEC)

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.mark_busy()
        try:
            action = task.get("action", "metrics")
            if action == "metrics":
                result = self._metrics_tool.execute({})
            elif action == "health":
                result = self._health_tool.execute({})
            else:
                result = self._metrics_tool.execute({})

            self.tasks_processed += 1
            return {
                "success": result.success,
                "output": result.output,
                "action": action,
                "agent_id": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error("PrometheusAgent task error: %s", e)
            raise
        finally:
            self.mark_idle()
