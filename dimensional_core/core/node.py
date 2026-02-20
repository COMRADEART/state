from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import time
import traceback
import uuid


class NodeStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class NodeResult:
    ok: bool
    value: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_s: Optional[float] = None


@dataclass
class Node:
    """
    A unit of work executed by the Engine's worker pool.

    Engine expects:
        - node.run() exists (callable with no args)
    """

    # Required
    fn: Callable[..., Any]

    # Optional execution inputs
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    name: str = "node"
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # Dependency / graph support
    depends_on: Set[str] = field(default_factory=set)   # node_ids this node depends on
    produces: Optional[str] = None                      # optional key to store output in engine state
    consumes: List[str] = field(default_factory=list)   # keys to read from engine state into kwargs

    # Reliability
    max_retries: int = 0
    retry_delay_s: float = 0.0

    # Runtime state
    status: NodeStatus = NodeStatus.PENDING
    attempts: int = 0
    last_error: Optional[str] = None
    result: Optional[NodeResult] = None

    # Hooks (optional)
    on_success: Optional[Callable[[Any], None]] = None
    on_failure: Optional[Callable[[str], None]] = None

    def run(self) -> Any:
        """
        Called by Engine inside a worker thread/process.
        Must not require parameters (Engine submits node.run directly).
        """
        self.status = NodeStatus.RUNNING
        self.attempts += 1

        started = time.time()
        try:
            value = self.fn(*self.args, **self.kwargs)

            finished = time.time()
            self.status = NodeStatus.DONE
            self.result = NodeResult(
                ok=True,
                value=value,
                error=None,
                started_at=started,
                finished_at=finished,
                duration_s=finished - started,
            )

            if self.on_success:
                try:
                    self.on_success(value)
                except Exception:
                    # Hooks should never crash execution
                    pass

            return value

        except Exception as e:
            finished = time.time()
            tb = traceback.format_exc()
            self.last_error = f"{type(e).__name__}: {e}\n{tb}"

            self.status = NodeStatus.FAILED
            self.result = NodeResult(
                ok=False,
                value=None,
                error=self.last_error,
                started_at=started,
                finished_at=finished,
                duration_s=finished - started,
            )

            if self.on_failure:
                try:
                    self.on_failure(self.last_error)
                except Exception:
                    pass

            # Re-raise so Engine can detect failure
            raise

    # Optional helpers -------------------------------------------------

    def can_retry(self) -> bool:
        return self.status == NodeStatus.FAILED and self.attempts <= self.max_retries

    def reset_for_retry(self) -> None:
        self.status = NodeStatus.PENDING
        # keep attempts count
        # keep last_error for debugging

    @staticmethod
    def simple(fn: Callable[..., Any], *args: Any, name: str = "node", **kwargs: Any) -> "Node":
        """
        Convenience constructor:
            Node.simple(func, 1, 2, name="add", a=3)
        """
        return Node(fn=fn, args=args, kwargs=kwargs, name=name)
