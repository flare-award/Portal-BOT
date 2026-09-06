"""Event bus for asynchronous decoupled communication between modules."""

import asyncio
from collections import defaultdict
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger("portal_bot.events")


class EventBus:
    """Thread-safe and async-safe event dispatcher."""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable[[Any], Any]]] = defaultdict(list)
        self._async_listeners: Dict[str, List[Callable[[Any], Any]]] = defaultdict(list)

    def subscribe(self, event_name: str, callback: Callable[[Any], Any]) -> None:
        if asyncio.iscoroutinefunction(callback):
            self._async_listeners[event_name].append(callback)
        else:
            self._listeners[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable[[Any], Any]) -> None:
        if callback in self._listeners[event_name]:
            self._listeners[event_name].remove(callback)
        if callback in self._async_listeners[event_name]:
            self._async_listeners[event_name].remove(callback)

    def publish(self, event_name: str, data: Any = None) -> None:
        for cb in list(self._listeners.get(event_name, [])):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Error in sync listener for {event_name}: {e}")

        # If there is a running asyncio loop, schedule async callbacks
        if self._async_listeners.get(event_name):
            try:
                loop = asyncio.get_running_loop()
                for acb in list(self._async_listeners[event_name]):
                    loop.create_task(self._safe_async_call(acb, data))
            except RuntimeError:
                pass

    async def _safe_async_call(self, callback: Callable, data: Any) -> None:
        try:
            await callback(data)
        except Exception as e:
            logger.error(f"Error in async listener: {e}")


# Global singleton instance
event_bus = EventBus()
