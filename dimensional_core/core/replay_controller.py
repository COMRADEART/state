# dimensional_core/core/replay_controller.py
"""
Deterministic event replay controller.

Reads events from the append-only JSONL log and re-applies them to the
engine's in-memory state, restoring the exact execution path up to the
last checkpoint.

Event format (as written by StateStore / SimpleWriter)
------------------------------------------------------
    {
        "type": "NODE_RESULT",          ← event name key
        "payload": { … },              ← structured payload
        "eid":  42,
        "_ts":  1700000000.0
    }

Bug fixed vs prior version: the old code used ``event_rec.get("event")``
which was always None because the stored key is ``"type"``.  Fixed here.

Handled event types
-------------------
NODE_RESULT
    - Marks the named node DONE with its result
    - Applies instance_updates (deterministic)
    - Updates the scheduler score if a score is present
    - Applies local_delta to graph._local
    - Rearms the X→Y→Z cycle if rearm=True

WARP_ROLLED_BACK
    - Triggers graph.rollback() to restore _local and re-queue X

WARP_LOCAL_SNAPSHOT
    - Full replacement of graph._local (used by warp_store snapshots)

WARP_SPAWNED
    - No-op during replay (warps are already built at init time)

ENGINE_START / STOP
    - Update monotonic counters only

All other event types
    - Silently update global_step / cycle counters only (forward-compatible)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ReplayController:
    """
    Applies a single persisted event to the live engine state.

    Called once per event during resume/replay.  All operations must be
    idempotent so that replaying the same log twice produces the same result.
    """

    def apply(self, engine: Any, event_rec: Dict[str, Any]) -> None:
        # ── Unpack stored format ────────────────────────────────────────────────
        # Events are stored as {"type": "…", "payload": {…}, "eid": …, "_ts": …}
        event_type: str = str(event_rec.get("type", "") or "")
        payload: Dict[str, Any] = event_rec.get("payload") or {}

        if not event_type:
            return  # malformed record — skip

        # ── Advance engine counters monotonically ──────────────────────────────
        gs  = payload.get("global_step")
        cyc = payload.get("cycle")
        if isinstance(gs, int) and gs > engine.global_step:
            engine.global_step = gs
        if isinstance(cyc, int) and cyc > engine.cycle:
            engine.cycle = cyc

        # ── Dispatch by event type ─────────────────────────────────────────────
        if event_type == "NODE_RESULT":
            self._apply_node_result(engine, payload)

        elif event_type == "WARP_ROLLED_BACK":
            self._apply_rollback(engine, payload)

        elif event_type == "WARP_LOCAL_SNAPSHOT":
            self._apply_local_snapshot(engine, payload)

        # All other types (WARP_SPAWNED, ENGINE_START, STOP, WARP_SCORE, …)
        # update counters only (already done above).

    # ── NODE_RESULT ────────────────────────────────────────────────────────────

    def _apply_node_result(self, engine: Any, payload: Dict[str, Any]) -> None:
        wid = payload.get("warp")
        nid = payload.get("node_id")
        res = payload.get("res") or {}

        if not (isinstance(wid, str) and isinstance(nid, str)):
            return

        graph = engine.warps.get(wid)

        # Mark node DONE
        if graph and hasattr(graph, "nodes") and nid in graph.nodes:
            n = graph.nodes[nid]
            n.status = "DONE"
            n.result = dict(res)

        # Apply instance updates (deterministic side-effect)
        inst_upd = res.get("instance_updates")
        if isinstance(inst_upd, dict) and isinstance(getattr(engine.store, "instance", None), dict):
            engine.store.instance.update(inst_upd)

        # Update scheduler score
        score = res.get("score") or res.get("min_score")
        if score is not None and wid:
            try:
                engine.scheduler.update_score(wid, float(score))
            except Exception:
                pass

        # Apply local delta
        local_delta = payload.get("local_delta")
        if isinstance(local_delta, dict) and graph is not None:
            if not isinstance(getattr(graph, "_local", None), dict):
                graph._local = {}
            graph._local.update(local_delta)

        # Rearm cycle on Z success
        if payload.get("rearm") and graph is not None and hasattr(graph, "rearm_cycle"):
            graph.rearm_cycle()

    # ── WARP_ROLLED_BACK ───────────────────────────────────────────────────────

    def _apply_rollback(self, engine: Any, payload: Dict[str, Any]) -> None:
        wid = payload.get("warp")
        if not isinstance(wid, str):
            return
        graph = engine.warps.get(wid)
        if graph is not None and hasattr(graph, "rollback"):
            graph.rollback()

    # ── WARP_LOCAL_SNAPSHOT ────────────────────────────────────────────────────

    def _apply_local_snapshot(self, engine: Any, payload: Dict[str, Any]) -> None:
        wid   = payload.get("warp")
        local = payload.get("local")
        if not isinstance(wid, str) or not isinstance(local, dict):
            return
        graph = engine.warps.get(wid)
        if graph is not None:
            import copy
            graph._local = copy.deepcopy(local)
