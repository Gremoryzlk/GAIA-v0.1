"""EventBus module for GAIA v7.3 event-driven architecture."""

import hashlib
import heapq
import hmac
import logging
import threading
import traceback
import uuid
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .event import Event, EventPriority
from .subscriber import Subscriber

try:
    from shared.logging import sign_event
except ImportError:
    warnings.warn(
        "shared.logging.sign_event not found — using insecure fallback. "
        "Do not use in production.",
        RuntimeWarning,
        stacklevel=2,
    )

    def sign_event(event: Event) -> str:
        import os
        import hashlib as _hl
        # Пытаемся взять ключ из окружения даже в fallback
        secret = os.environ.get("GAIA_SECRET_KEY", "")
        key = _hl.sha256(secret.encode()).digest() if secret else bytes(32)
        event_data = (
            f"{event.event_type}:{event.source}:{event.timestamp.isoformat()}"
        )
        return hmac.new(key, event_data.encode(), digestmod=hashlib.sha256).hexdigest()


logger = logging.getLogger(__name__)

_DEDUP_MAX = 10_000
_DEDUP_TRIM = 5_000


class EventBus:
    """Thread-safe singleton EventBus for event-driven communication."""

    _instance: Optional["EventBus"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EventBus":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._queue_lock = threading.Lock()
        self._queue: List[tuple] = []
        self._subscribers: List[Subscriber] = []
        self._subscribers_lock = threading.Lock()
        # OrderedDict — детерминированный FIFO purge вместо set
        self._processed_ids: OrderedDict = OrderedDict()
        self._processed_ids_lock = threading.Lock()
        self._rate_limits: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"tokens": 10, "last_update": datetime.now(timezone.utc)}
        )
        self._rate_limits_lock = threading.Lock()
        self._running = True
        self._max_queue_size = 1000
        self._counter: int = 0  # ← сюда
        self._initialized = True

    def publish(self, event: Event) -> None:
        try:
            if event.message_id is None:
                event.message_id = str(uuid.uuid4())

            if event.signature is None:
                event.signature = sign_event(event)

            # Дедупликация с FIFO-purge
            with self._processed_ids_lock:
                if event.message_id in self._processed_ids:
                    return
                if len(self._processed_ids) >= _DEDUP_MAX:
                    # Удаляем самые старые (_DEDUP_TRIM штук)
                    for _ in range(_DEDUP_TRIM):
                        self._processed_ids.popitem(last=False)
                self._processed_ids[event.message_id] = None

            if event.priority != EventPriority.CRITICAL:
                if not self._check_rate_limit(event.source):
                    return

            with self._queue_lock:
                pressure = len(self._queue) / self._max_queue_size
                if pressure >= 0.9 and event.priority == EventPriority.LOW:
                    return
                if pressure >= 1.0 and event.priority == EventPriority.NORMAL:
                    return
                self._counter += 1  # ← сюда
                heapq.heappush(self._queue, (-event.priority.value, self._counter, event))  # ← и сюда

        except Exception as e:
            self._log_exception("Error publishing event", e)

    def subscribe(
        self,
        name: str,
        event_types: List[str],
        callback: Callable[[Any], None],
        priority_filter: Optional[EventPriority] = None,
    ) -> Subscriber:
        subscriber = Subscriber(
            name=name,
            event_types=event_types,
            callback=callback,
            priority_filter=priority_filter,
        )
        with self._subscribers_lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._subscribers_lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def dispatch_next(self) -> None:
        with self._queue_lock:
            if not self._queue:
                return
            _, _cnt, event = heapq.heappop(self._queue)

        with self._subscribers_lock:
            subscribers = list(self._subscribers)

        for subscriber in subscribers:
            try:
                if subscriber.matches(event.event_type) and subscriber.should_receive(
                    event.priority
                ):
                    subscriber.callback(event)
            except Exception as e:
                self._log_exception(f"Error in subscriber {subscriber.name}", e)

    def get_pressure(self) -> float:
        with self._queue_lock:
            return len(self._queue) / self._max_queue_size

    def shutdown(self) -> None:
        """Shutdown EventBus и сбрасывает singleton для возможного перезапуска."""
        with self.__class__._lock:
            self._running = False
        with self._queue_lock:
            self._queue.clear()
        with self._subscribers_lock:
            self._subscribers.clear()
        with self._processed_ids_lock:
            self._processed_ids.clear()
        # Сброс singleton — следующий EventBus() создаст новый экземпляр
        with self.__class__._lock:
            self.__class__._instance = None
            self._initialized = False

    def _check_rate_limit(self, agent_id: str) -> bool:
        now = datetime.now(timezone.utc)
        with self._rate_limits_lock:
            rate_info = self._rate_limits[agent_id]
            elapsed = (now - rate_info["last_update"]).total_seconds()
            rate_info["tokens"] = min(10, rate_info["tokens"] + elapsed * 10)
            rate_info["last_update"] = now
            if rate_info["tokens"] >= 1:
                rate_info["tokens"] -= 1
                return True
            return False

    def _log_exception(self, message: str, exception: Exception) -> None:
        tb_str = traceback.format_exc()
        logger.error("%s: %s\n%s", message, exception, tb_str)