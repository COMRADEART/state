# dimensional_core/core/engine.py
from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor

from .isa_vector import LANES_DEFAULT, VecVM, VInstr
from .warp_store import WarpStore
from .batch_scheduler import BatchScheduler
from .warp_factory import build_warp_graph

from .resume_controller import ResumeController
from .replay_controller import ReplayController
from .triggers import TriggerConfig
from .priority_scheduler import HeapWarpScheduler


class Engine:
    """
    C21 = C20 + Data Structures
      - Graph ReadySet (O(1) ready node scheduling)
      - Heap warp scheduler (O(log n) warp selection)
      - Replay index (seek-based replay)
    """

    def __init__(
        self,
        store,
        writer,
        resume: bool = False,
        triggers: TriggerConfig | None = None,
        batch_scheduler=None,
        base_graphs=None,
        max_workers: int = 4,
        max_in_flight: int = 8,
        lanes: int = LANES_DEFAULT,
        **_ignore_extra,
    ):
        self.store = store
        self.writer = writer
        self.resume = bool(resume)

        self.tr_cfg = triggers or TriggerConfig()
        self.resumer = ResumeController()
        self.replayer = ReplayController()

        self.max_in_flight = int(max_in_flight)
        self.lanes = int(lanes)

        # BatchScheduler remains (optional), but graph has ReadySet now.
        self.batch = batch_scheduler or BatchScheduler()

        # C21 scheduler DS
        self.scheduler = HeapWarpScheduler()

        self.warp_store = WarpStore(root_dir="dimensional_core/state")

        self.pool = ThreadPoolExecutor(max_workers=int(max_workers))
        self.inflight = {}  # future -> (warp_id, graph_node)
        self.lock = threading.Lock()

        self.vvm = VecVM()

        self.global_step = 0
        self.cycle = 0
        self.last_eid = 0
        self._events_since_commit = 0
        self._commit_count = 0

        self.active = True
        self.warps = {}

        if not hasattr(self.store, "instance") or not isinstance(getattr(self.store, "instance"), dict):
            self.store.instance = {}

        # snapshot load
        self._snapshot = None
        if self.resume and hasattr(self.store, "load_instance_point"):
            ip = self.store.load_instance_point()
            snap = self.resumer.resolve(ip)
            self._snapshot = snap
            self.global_step = snap["global_step"]
            self.cycle = snap["cycle"]
            self.last_eid = snap["last_eid"]
            self.store.instance = dict(snap["instance"])

        # build warps
        self._spawn_initial_warps(base_graphs)

        # apply snapshot warps state
        if self._snapshot:
            self._apply_warps_state(self._snapshot.get("warps_state", {}))

        # replay (fast if available)
        if self.resume:
            if hasattr(self.store, "iter_events_after_fast"):
                it = self.store.iter_events_after_fast(self.last_eid)
            elif hasattr(self.store, "iter_events_after"):
                it = self.store.iter_events_after(self.last_eid)
            else:
                it = []

            for e in it:
                eid = int(e.get("eid", 0) or 0)
                if eid > self.last_eid:
                    self.last_eid = eid
                self.replayer.apply(self, e)

    # ------------------------------------------------------------------

    def _normalize_base_graphs(self, base_graphs):
        bg = base_graphs
        if bg is None:
            gids = [f"G{i}" for i in range(8)]
            return {"W0": gids[:4], "W1": gids[4:]}
        if isinstance(bg, int):
            n = max(1, int(bg))
            gids = [f"G{i}" for i in range(n)]
            mid = max(1, len(gids) // 2)
            return {"W0": gids[:mid], "W1": gids[mid:]}
        if isinstance(bg, (list, tuple)):
            gids = list(bg) or ["G0"]
            mid = max(1, len(gids) // 2)
            return {"W0": gids[:mid], "W1": gids[mid:]}
        if isinstance(bg, dict):
            out = {}
            for wid, lane_gids in bg.items():
                out[str(wid)] = list(lane_gids) if isinstance(lane_gids, (list, tuple)) else [str(lane_gids)]
            return out or {"W0": ["G0"], "W1": []}
        gids = [f"G{i}" for i in range(8)]
        return {"W0": gids[:4], "W1": gids[4:]}

    def _spawn_initial_warps(self, base_graphs=None):
        bg = self._normalize_base_graphs(base_graphs)
        for wid, lane_gids in bg.items():
            g = build_warp_graph(wid, lane_gids, self.store.instance, lanes=self.lanes)
            if not hasattr(g, "_local"):
                g._local = {}
            self.warps[wid] = g
            self.scheduler.register(wid)

    def _apply_warps_state(self, warps_state: dict) -> None:
        if not isinstance(warps_state, dict):
            return
        for wid, st in warps_state.items():
            g = self.warps.get(wid)
            if g is None:
                continue

            local = st.get("local")
            if isinstance(local, dict):
                g._local = dict(local)

            nodes = st.get("nodes")
            if isinstance(nodes, dict) and hasattr(g, "nodes") and isinstance(g.nodes, dict):
                for nid, nd in nodes.items():
                    if nid in g.nodes:
                        n = g.nodes[nid]
                        n.status = nd.get("status", getattr(n, "status", "PENDING"))
                        n.result = nd.get("result", getattr(n, "result", None))
                # rebuild ReadySet from node statuses (important!)
                for nid, n in g.nodes.items():
                    if getattr(n, "status", "PENDING") == "PENDING" and not g.deps.get(nid):
                        g.ready.add(nid)

    # ------------------------------------------------------------------
    # emit + commit
    # ------------------------------------------------------------------

    def _emit(self, name: str, payload: dict | None = None) -> None:
        payload = dict(payload or {})
        payload.setdefault("global_step", int(self.global_step))
        payload.setdefault("cycle", int(self.cycle))

        rec = None
        if hasattr(self.writer, "event"):
            rec = self.writer.event(name, payload)

        if isinstance(rec, dict):
            eid = int(rec.get("eid", 0) or 0)
            if eid > self.last_eid:
                self.last_eid = eid

        self._events_since_commit += 1
        if self._events_since_commit >= int(self.tr_cfg.commit_every_n_events):
            self._commit_instance_point(reason=f"every_{self.tr_cfg.commit_every_n_events}_events")

    def _commit_instance_point(self, reason: str) -> None:
        warps_state = {}
        for wid, g in self.warps.items():
            nodes_state = {}
            if hasattr(g, "nodes") and isinstance(g.nodes, dict):
                for nid, n in g.nodes.items():
                    nodes_state[nid] = {
                        "status": getattr(n, "status", "PENDING"),
                        "result": getattr(n, "result", None),
                    }
            warps_state[wid] = {
                "local": dict(getattr(g, "_local", {}) or {}),
                "nodes": nodes_state,
            }

        ip = {
            "global_step": int(self.global_step),
            "cycle": int(self.cycle),
            "last_eid": int(self.last_eid),
            "instance": dict(getattr(self.store, "instance", {}) or {}),
            "warps_state": warps_state,
            "reason": str(reason),
            "_ts": time.time(),
        }

        if hasattr(self.store, "save_instance_point"):
            self.store.save_instance_point(ip)
        elif hasattr(self.writer, "submit_instance_point"):
            self.writer.submit_instance_point(ip)

        self._events_since_commit = 0
        self._commit_count += 1

        if self.tr_cfg.rotate_every_n_commits > 0 and (
            self._commit_count % int(self.tr_cfg.rotate_every_n_commits) == 0
        ):
            if hasattr(self.store, "rotate_events"):
                self.store.rotate_events(keep_archives=int(self.tr_cfg.keep_archives))

    # ------------------------------------------------------------------
    # graph node -> task
    # ------------------------------------------------------------------

    def _compile_task(self, wid: str, graph, gnode):
        params = getattr(gnode, "params", {}) or {}
        op = getattr(gnode, "op", None)
        stage = str(params.get("stage", "")).lower()
        lane_gids = list(params.get("lane_gids", []))
        lr = float(params.get("lr", 0.1))

        local = getattr(graph, "_local", None)
        if local is None:
            local = {}
            graph._local = local

        def task():
            try:
                gnode.status = "RUNNING"
            except Exception:
                pass

            if op != "VISA":
                raise RuntimeError(f"Unsupported node op={op} id={getattr(gnode,'id','?')}")

            if stage == "init":
                out = self.vvm.run(VInstr(op="VINIT", args={}), self.store.instance, local, lane_gids)
                return {"value": out}

            if stage == "step":
                out = self.vvm.run(VInstr(op="VSTEP", args={"lr": lr}), self.store.instance, local, lane_gids)
                return {"value": out}

            if stage == "score":
                out = self.vvm.run(VInstr(op="VSCORE", args={}), self.store.instance, local, lane_gids)
                min_score = out.get("min_score")
                if min_score is None:
                    sv = out.get("score_vec") or []
                    min_score = float(min(sv)) if sv else None
                res = {"value": out}
                if min_score is not None:
                    res["score"] = float(min_score)
                return res

            raise RuntimeError(f"Unknown VISA stage={stage} id={getattr(gnode,'id','?')}")

        return task

    # ------------------------------------------------------------------

    def run_forever(self):
        self._emit("ENGINE_START", {"resume": self.resume})

        while self.active:
            self.global_step += 1

            self._dispatch()

            finished = self._collect_finished()
            if finished:
                self._apply_results(finished)

            if self.global_step % 5 == 0:
                for wid in list(self.warps.keys()):
                    self._save_warp_point(wid, tag="PERIODIC")

            max_steps = getattr(self.store, "max_steps", None)
            if isinstance(max_steps, int) and max_steps > 0 and self.global_step >= max_steps:
                self._stop_all()
                break

            time.sleep(0.01)

    def _dispatch(self):
        with self.lock:
            if len(self.inflight) >= self.max_in_flight:
                return

            wid = self.scheduler.choose()
            if wid is None:
                return

            graph = self.warps.get(wid)
            if graph is None:
                return

            # C21: O(1) ready-node scheduling
            nodes = graph.take_ready(2)  # deterministic
            if not nodes:
                return

            for gnode in nodes:
                task = self._compile_task(wid, graph, gnode)
                fut = self.pool.submit(task)
                self.inflight[fut] = (wid, gnode)

            self._emit(
                "WARP_BATCH_SUBMITTED",
                {
                    "warp": wid,
                    "batch": len(nodes),
                    "inflight": len(self.inflight),
                    "priority_value": self.scheduler.priority(wid),
                },
            )

    def _collect_finished(self):
        done = []
        with self.lock:
            for fut in list(self.inflight.keys()):
                if fut.done():
                    wid, gnode = self.inflight.pop(fut)
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {"error": str(e)}
                    done.append((wid, gnode, res))
        return done

    def _apply_results(self, finished):
        for wid, gnode, res in finished:
            if not isinstance(res, dict):
                res = {"value": res}

            graph = self.warps.get(wid)

            # advance deps and ready nodes
            if graph is not None and hasattr(graph, "mark_done"):
                graph.mark_done(getattr(gnode, "id"), dict(res))

            # score -> scheduler heap update
            score = res.get("score")
            if score is not None:
                self.scheduler.update_score(wid, float(score))
                self._emit(
                    "WARP_SCORE",
                    {"warp": wid, "min_score": float(score), "priority_value": self.scheduler.priority(wid)},
                )

            nid = str(getattr(gnode, "id", ""))
            rearm = nid.endswith(":score")

            # authoritative replay event
            self._emit(
                "NODE_RESULT",
                {
                    "warp": wid,
                    "node_id": nid,
                    "res": dict(res),
                    "local_delta": {},
                    "rearm_step_score": bool(rearm),
                },
            )

            # keep warp loop alive: after score, rearm step + score
            if rearm and graph is not None:
                graph.rearm(f"{wid}:step")
                graph.rearm(f"{wid}:score")

    def _save_warp_point(self, wid, tag):
        graph = self.warps.get(wid)
        local = getattr(graph, "_local", {}) if graph is not None else {}
        self.warp_store.save(
            wid,
            {"warp": wid, "tag": tag, "global_step": self.global_step, "local": local, "_ts": time.time()},
        )
        self._emit("WARP_POINT_SAVED", {"warp": wid, "tag": tag})

    def _stop_all(self):
        for wid in self.warps:
            self._save_warp_point(wid, tag="STOP")
        self.pool.shutdown(wait=True)
        self.active = False
        self._emit("STOP", {"global_step": self.global_step, "cycle": self.cycle})
        self._commit_instance_point(reason="stop")
