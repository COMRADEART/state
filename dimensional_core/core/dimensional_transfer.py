# EXPERIMENTAL — not used by the production engine
from __future__ import annotations
from typing import Dict, Any
import hashlib
import json
import time


class DimensionalTransfer:
    """
    C23 — Dimensional Transfer Layer

    Moves a warp state from the Execution Dimension
    into another logical Dimension (storage / replay / resume).
    """

    def __init__(self, store, writer) -> None:
        self.store = store
        self.writer = writer

    def capture(self, engine, warp_id: str) -> Dict[str, Any]:
        """
        Capture minimal transferable state of a warp.
        """
        graph = engine.warps.get(warp_id)
        if graph is None:
            raise ValueError(f"Warp not found: {warp_id}")

        local = getattr(graph, "_local", {})
        nodes = {nid: n.status for nid, n in graph.nodes.items()}

        payload = {
            "warp": warp_id,
            "cycle": engine.cycle,
            "global_step": engine.global_step,
            "local": dict(local),
            "nodes": nodes,
            "_ts": time.time(),
        }
        return payload

    def encode(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode payload with hash + metadata.
        """
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()

        return {
            "digest": digest,
            "payload": payload,
        }

    def transfer(self, encoded: Dict[str, Any], target: str) -> None:
        """
        Persist encoded payload into a target dimension.
        """
        rec = {
            "event": "DIMENSION_TRANSFER",
            "target_dimension": target,
            "digest": encoded["digest"],
            "payload": encoded["payload"],
            "_ts": time.time(),
        }
        self.writer.event("DIMENSION_TRANSFER", rec)

        # Persist as instance point for resume dimension
        if target == "resume":
            self.store.save_instance_point(encoded["payload"])

    def execute(self, engine, warp_id: str, target: str = "storage") -> None:
        """
        Full C23 pipeline: capture → encode → transfer
        """
        payload = self.capture(engine, warp_id)
        encoded = self.encode(payload)
        self.transfer(encoded, target)