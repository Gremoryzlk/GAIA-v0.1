"""TitanAgent (L2) — тяжёлые вычисления и ML обучение для GAIA v7.3.

Архитектура:
- По умолчанию enabled=false — система работает без Titan
- При compute.request → TitanCluster.dispatch() → Titan сервер
- Локальные tools (scraping, compute, ml_train) как fallback
- Результаты → MemoryBroker / файл / chunked (по размеру)
- DataLifecycleManager управляет хранением данных
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from agents.base_agent import BaseAgent
from core.eventbus.event import Event, EventPriority
from modules.hephaestus.tools.base_tool import ToolResult
from modules.titan.cluster import TitanCluster
from modules.titan.lifecycle import DataLifecycleManager
from modules.titan.worker import TitanWorker
from modules.titan.tools.scraping import ScrapingTool
from modules.titan.tools.compute import ComputeTool
from modules.titan.tools.ml import MLTrainTool
from shared.tool_pool import ToolPool

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.json"


class TitanAgent(BaseAgent):
    """L2 агент тяжёлых вычислений.

    Принимает compute.request события, диспатчит через TitanCluster.
    При disabled или недоступности серверов — выполняет локально через tools.
    """

    def __init__(self, meta_controller=None, memory_broker=None) -> None:
        super().__init__(agent_type="titan_l2", level=2)

        self._cfg = self._load_config()
        self._enabled: bool = self._cfg.get("enabled", False)

        # Кластер серверов
        self._cluster = TitanCluster()

        # Воркер (маршрутизация результатов)
        self._worker = TitanWorker(memory_broker=memory_broker)

        # DataLifecycleManager
        self._lifecycle = DataLifecycleManager(self._cfg)

        # Локальные tools (fallback)
        self._scraping_tool = ScrapingTool()
        self._compute_tool = ComputeTool()
        self._ml_tool = MLTrainTool(
            meta_controller=meta_controller,
            lifecycle=self._lifecycle,
        )

        # Регистрация в ToolPool
        pool = ToolPool()
        pool.register(self._scraping_tool)
        pool.register(self._compute_tool)
        pool.register(self._ml_tool)

        logger.info(
            "TitanAgent %s initialized enabled=%s cluster_servers=%d",
            self.agent_id,
            self._enabled,
            len(self._cluster._servers),
        )

    def _load_config(self) -> Dict[str, Any]:
        try:
            return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("TitanAgent: config load failed: %s", e)
            return {"enabled": False}

    def _subscribe_to_events(self) -> None:
        self.eventbus.subscribe(
            name=f"{self.agent_id}_subscriber",
            event_types=["compute.request", "agent.restored"],
            callback=self._handle_event,
        )

    def _handle_event(self, event: Event) -> None:
        try:
            if event.event_type == "compute.request":
                self._handle_compute_request(event)
            elif event.event_type == "agent.restored":
                # После восстановления — запускаем обслуживание данных
                if event.payload.get("agent_id") == self.agent_id:
                    self._lifecycle.run_maintenance()
        except Exception as e:
            logger.error("TitanAgent event error: %s", e)
            self.increment_rogue_score(0.1)

    def _handle_compute_request(self, event: Event) -> None:
        """Обрабатывает compute.request из EventBus."""
        payload = event.payload
        task_id = payload.get("task_id", uuid4().hex)
        task_type = payload.get("task_type", "compute")
        task_payload = payload.get("payload", {})
        source_agent = event.source

        logger.info(
            "TitanAgent: compute.request task_id=%s task_type=%s from=%s",
            task_id, task_type, source_agent,
        )

        if not self.accept_task():
            return

        result = self._dispatch(task_id, task_type, task_payload)

        # Сохраняем результат
        storage_info = self._worker.store_result(task_id, task_type, result)

        self.publish_event(
            event_type="compute.result",
            priority=EventPriority.NORMAL,
            payload={
                "task_id": task_id,
                "task_type": task_type,
                "result": result,
                "storage": storage_info,
                "source_agent": source_agent,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.tasks_processed += 1

    def _dispatch(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Диспатчит задачу: Titan сервер → локальный fallback."""

        # Чекпоинт callback
        def on_checkpoint(cp: Dict[str, Any]) -> None:
            step = cp.get("step", 0)
            state = cp.get("state", {})
            path = self._worker.save_checkpoint(task_id, step, state)
            self.publish_event(
                event_type="titan.checkpoint",
                priority=EventPriority.LOW,
                payload={"task_id": task_id, "step": step, "checkpoint_path": path},
            )

        # Пробуем Titan кластер
        if self._enabled and self._cluster.enabled:
            cluster_result = self._cluster.dispatch(task_type, payload, on_checkpoint)
            if cluster_result.get("success", False) or not cluster_result.get("fallback"):
                # Чистим чекпоинты после успешного завершения
                self._worker.cleanup_checkpoints(task_id)
                return cluster_result

            logger.warning(
                "TitanAgent: cluster dispatch failed/disabled reason=%s — using local fallback",
                cluster_result.get("reason"),
            )

        # Локальный fallback
        return self._local_dispatch(task_type, payload)

    def _local_dispatch(
        self, task_type: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Выполняет задачу локально через соответствующий tool."""
        logger.info("TitanAgent: local dispatch task_type=%s", task_type)

        tool_map = {
            "scraping": self._scraping_tool,
            "compute": self._compute_tool,
            "ml_train": self._ml_tool,
            "ml_batch": self._ml_tool,
        }

        tool = tool_map.get(task_type)
        if tool is None:
            logger.warning("TitanAgent: unknown task_type=%s", task_type)
            return {
                "success": False,
                "error": f"Unknown task_type: {task_type}",
                "fallback": True,
            }

        result: ToolResult = tool.execute(payload)
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
            "fallback": True,
            "local": True,
        }

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Прямой вызов задачи (минуя EventBus)."""
        if not self.accept_task():
            return {"success": False, "error": "agent isolated"}

        self.mark_busy()
        try:
            task_id = task.get("task_id", uuid4().hex)
            task_type = task.get("task_type", "compute")
            payload = task.get("payload", task)

            result = self._dispatch(task_id, task_type, payload)
            storage_info = self._worker.store_result(task_id, task_type, result)
            self.tasks_processed += 1

            return {
                "task_id": task_id,
                "task_type": task_type,
                "result": result,
                "storage": storage_info,
                "agent_id": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            self.increment_rogue_score(0.1)
            logger.error("TitanAgent process_task error: %s", e)
            raise
        finally:
            self.mark_idle()

    def run_maintenance(self) -> Dict[str, Any]:
        """Запускает обслуживание данных вручную."""
        return self._lifecycle.run_maintenance()

    def get_cluster_status(self) -> Dict[str, Any]:
        """Возвращает статус Titan кластера."""
        return self._cluster.get_status()
