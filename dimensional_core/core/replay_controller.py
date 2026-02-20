# dimensional_core/core/replay_controller.py
from __future__ import annotations
from typing import Dict, Any


class ReplayController:
    """
    Deterministic event replay.

    Rebuilds:
      - store.instance
      - warp graph node status/result
      - warp local state (graph._local)
      - scheduler score (optional)
    """

    def apply(self, engine, event_rec: Dict[str, Any]) -> None:
        et = event_rec.get("event")
        if not et:
            return

        # Keep engine counters monotonic during replay
        gs = event_rec.get("global_step")
        cyc = event_rec.get("cycle")
        if isinstance(gs, int) and gs > engine.global_step:
            engine.global_step = gs
        if isinstance(cyc, int) and cyc > engine.cycle:
            engine.cycle = cyc

        if et == "NODE_RESULT":
            wid = event_rec.get("warp")
            nid = event_rec.get("node_id")
            res = event_rec.get("res") or {}

            graph = engine.warps.get(wid)
            if graph and hasattr(graph, "nodes") and isinstance(graph.nodes, dict) and nid in graph.nodes:
                n = graph.nodes[nid]
                n.status = "DONE"
                n.result = dict(res)

            # Apply instance updates (deterministic)
            inst_upd = res.get("instance_updates")
            if isinstance(inst_upd, dict) and isinstance(engine.store.instance, dict):
                engine.store.instance.update(inst_upd)

            # Update scheduler (deterministic)
            score = res.get("score")
            if score is not None:
                try:
                    engine.scheduler.update_score(wid, float(score))
                except Exception:
                    pass

            # Apply local delta/snapshot
            local_delta = event_rec.get("local_delta")
            if isinstance(local_delta, dict) and graph is not None:
                if not hasattr(graph, "_local") or not isinstance(graph._local, dict):
                    graph._local = {}
                graph._local.update(local_delta)

            # Re-arm loop (so step/score continue)
            if event_rec.get("rearm_step_score") is True and graph is not None and hasattr(graph, "nodes"):
                step_id = f"{wid}:step"
                score_id = f"{wid}:score"
                if step_id in graph.nodes:
                    graph.nodes[step_id].status = "PENDING"
                if score_id in graph.nodes:
                    graph.nodes[score_id].status = "PENDING"

            return

        if et == "WARP_LOCAL_SNAPSHOT":
            wid = event_rec.get("warp")
            local = event_rec.get("local")
            graph = engine.warps.get(wid)
            if graph is not None and isinstance(local, dict):
                graph._local = dict(local)

            return
