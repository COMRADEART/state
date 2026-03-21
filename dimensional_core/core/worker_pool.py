# EXPERIMENTAL — not used by the production engine
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any


class WorkerPool:
    def __init__(self, max_workers: int = 4) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        return self.executor.submit(fn, *args, **kwargs)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)