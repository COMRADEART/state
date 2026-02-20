from __future__ import annotations
from typing import Dict, Any, Callable

OpFn = Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Any]]


class OpRuntime:
    """
    Per-graph local state (C+5). For C+8 we expose local dict for worker use.
    """
    def __init__(self, ops: Dict[str, OpFn]) -> None:
        self.ops = ops
        self.local: Dict[str, Dict[str, Any]] = {}

    def get_local(self, gid: str) -> Dict[str, Any]:
        if gid not in self.local:
            self.local[gid] = {}
        return self.local[gid]

    def get_op(self, op_name: str) -> OpFn:
        fn = self.ops.get(op_name)
        if fn is None:
            raise RuntimeError(f"missing op: {op_name}")
        return fn

    def reset_graph(self, gid: str) -> None:
        self.local.pop(gid, None)
