from __future__ import annotations
from typing import Callable, Dict, Any, List


class EventBus:
    """
    Simple pub-sub event bus.
    - Always prints events to console.
    - Allows subscribers to receive (name, payload) for persistence, metrics, etc.
    """

    def __init__(self) -> None:
        self._subs: List[Callable[[str, Dict[str, Any]], None]] = []

    def subscribe(self, fn: Callable[[str, Dict[str, Any]], None]) -> None:
        self._subs.append(fn)

    def emit(self, name: str, **payload: Any) -> None:
        # Console output
        print(f"[EVENT] {name} | {payload}")

        # Subscribers
        for fn in list(self._subs):
            fn(name, dict(payload))
