"""TitanCluster — менеджер серверов Titan.

Выбор сервера по task_affinity + Circuit Breaker.
До 10 серверов. По умолчанию отключён (enabled: false в config.json).
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from modules.titan.client import TitanClient

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.json"

# Типы задач, которые требуют WebSocket (долгие)
_WS_TASK_TYPES = {"scraping", "ml_train", "ml_batch"}


class _CircuitBreaker:
    """Per-сервер Circuit Breaker.

    States: CLOSED → OPEN (при failure_rate > threshold) → HALF_OPEN → CLOSED.
    """

    def __init__(self, failure_threshold: float, recovery_timeout_sec: float) -> None:
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_sec
        self._failures = 0
        self._successes = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if time.time() - self._opened_at >= self._recovery_timeout:
                # Переходим в HALF_OPEN
                logger.info("CircuitBreaker: HALF_OPEN after recovery timeout")
                self._opened_at = None
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._successes += 1
            self._failures = max(0, self._failures - 1)
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            total = self._failures + self._successes
            if total > 0 and self._failures / total >= self._threshold:
                if self._opened_at is None:
                    self._opened_at = time.time()
                    logger.warning("CircuitBreaker: OPEN — failure_rate=%.2f", self._failures / total)

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._successes = 0
            self._opened_at = None


class _ServerEntry:
    """Один Titan сервер с Circuit Breaker и метаданными."""

    def __init__(
        self,
        client: TitanClient,
        cb_failure_threshold: float,
        cb_recovery_timeout: float,
    ) -> None:
        self.client = client
        self.cb = _CircuitBreaker(cb_failure_threshold, cb_recovery_timeout)
        self.dispatched: int = 0
        self.errors: int = 0

    @property
    def available(self) -> bool:
        return not self.cb.is_open


class TitanCluster:
    """Менеджер кластера Titan серверов.

    Singleton — создавать через TitanCluster.get_instance().
    При enabled=false все dispatch() немедленно возвращают disabled-результат.
    """

    _instance: Optional["TitanCluster"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TitanCluster":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._enabled: bool = False
        self._servers: List[_ServerEntry] = []
        self._agent_id: str = "titan_cluster"
        self._cfg: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        try:
            cfg = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            self._cfg = cfg
            self._enabled = cfg.get("enabled", False)

            if not self._enabled:
                logger.info("TitanCluster: disabled (config.json enabled=false)")
                return

            cb_cfg = cfg.get("circuit_breaker", {})
            cb_ft = cb_cfg.get("failure_threshold", 0.2)
            cb_rt = cb_cfg.get("recovery_timeout_sec", 60)

            for srv in cfg.get("servers", []):
                url = srv.get("url", "")
                secret = srv.get("secret", "")
                affinity = srv.get("task_affinity", [])
                if not url or not secret:
                    logger.warning("TitanCluster: skipping server — missing url/secret")
                    continue
                client = TitanClient(
                    server_url=url,
                    server_secret=secret,
                    agent_id=self._agent_id,
                    task_affinity=affinity,
                )
                self._servers.append(_ServerEntry(client, cb_ft, cb_rt))
                logger.info("TitanCluster: registered server %s affinity=%s", url, affinity)

            logger.info(
                "TitanCluster: initialized with %d servers", len(self._servers)
            )
        except Exception as e:
            logger.error("TitanCluster: config load failed: %s", e)
            self._enabled = False

    # ─── Server selection ─────────────────────────────────────────────────────

    def _select_server(self, task_type: str) -> Optional[_ServerEntry]:
        """Выбирает сервер по task_affinity, потом по наименьшей нагрузке."""
        available = [s for s in self._servers if s.available]
        if not available:
            return None

        # Предпочитаем серверы с matching affinity
        affinity_match = [s for s in available if task_type in s.client.task_affinity]
        pool = affinity_match if affinity_match else available

        # Round-robin по dispatched count
        return min(pool, key=lambda s: s.dispatched)

    # ─── Public API ───────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def dispatch(
        self,
        task_type: str,
        payload: Dict[str, Any],
        on_checkpoint: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Dispatches задачу на доступный Titan сервер.

        При enabled=false или нет доступных серверов — возвращает disabled/unavailable.
        Автоматически выбирает HTTP vs WebSocket по task_type.
        """
        if not self._enabled:
            return {"success": False, "reason": "titan_disabled", "fallback": True}

        entry = self._select_server(task_type)
        if entry is None:
            logger.warning("TitanCluster: no available servers for task_type=%s", task_type)
            return {"success": False, "reason": "no_available_servers", "fallback": True}

        use_ws = task_type in _WS_TASK_TYPES
        entry.dispatched += 1

        try:
            if use_ws:
                result = entry.client.ws_dispatch(task_type, payload, on_checkpoint)
            else:
                result = entry.client.http_dispatch(task_type, payload)

            entry.cb.record_success()
            result["titan_server"] = entry.client.server_url
            return result

        except Exception as e:
            entry.cb.record_failure()
            entry.errors += 1
            logger.error(
                "TitanCluster: dispatch failed task_type=%s server=%s error=%s",
                task_type, entry.client.server_url, e,
            )
            return {
                "success": False,
                "reason": "dispatch_error",
                "error": str(e),
                "fallback": True,
            }

    def health_check_all(self) -> Dict[str, bool]:
        """Проверяет доступность всех серверов."""
        return {
            s.client.server_url: s.client.health_check()
            for s in self._servers
        }

    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус кластера."""
        return {
            "enabled": self._enabled,
            "server_count": len(self._servers),
            "available_servers": sum(1 for s in self._servers if s.available),
            "servers": [
                {
                    "url": s.client.server_url,
                    "available": s.available,
                    "dispatched": s.dispatched,
                    "errors": s.errors,
                    "task_affinity": s.client.task_affinity,
                }
                for s in self._servers
            ],
        }

    @classmethod
    def reset(cls) -> None:
        """Сброс singleton — только для тестов."""
        with cls._lock:
            cls._instance = None
