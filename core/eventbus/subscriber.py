"""Subscriber module for EventBus event handling."""

from typing import Any, Callable, List, Optional

from shared.types import EventPriority


class Subscriber:
    """Subscriber for EventBus events."""

    def __init__(
        self,
        name: str,
        event_types: List[str],
        callback: Callable[[Any], None],
        priority_filter: Optional[EventPriority] = None,
        subscriber_id: str = "",
    ) -> None:
        self.name = name
        self.event_types = event_types
        self.callback = callback
        self.priority_filter = priority_filter
        self.subscriber_id = subscriber_id if subscriber_id else name

    def matches(self, event_type: str) -> bool:
        return event_type in self.event_types

    def should_receive(self, priority: EventPriority) -> bool:
        if self.priority_filter is None:
            return True
        return priority.value >= self.priority_filter.value