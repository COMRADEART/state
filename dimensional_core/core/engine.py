from __future__ import annotations

import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

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
    C21/C22/C24 engine:
    - deterministic replay + resume
    - graceful SIGINT/SIGTERM
    - buffered event flush on shutdown
    - watchdog for stuck warps
    - stable requested warp count
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
        target_warps: int | None = None,
        min_warps: int = 1,
        watchdog_timeout_s: float = 30.0,
        watchdog_poll_s: float = 1.0,
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
        self.batch = batch_scheduler or BatchScheduler()
        self.scheduler = HeapWarpScheduler()
        self.warp_store = WarpStore(root_dir="dimensional_core/state")
        self.pool = ThreadPoolExecutor(max_workers=int(max_workers))

        self.lock = threading.RLock()
        self.vvm = VecVM()

        self.global_step = 0
        self.cycle = 0
        self.last_eid = 0
        self._events_since_commit = 0
        self._commit_count = 0

        self.active = True
        self._stop_requested = threading.Event()
        self._shutdown_once = threading.Event()

        self.warps: Dict[str, object] = {}
        self._target_warps = max(1, int(target_warps)) if target_warps else None
        self._min_warps = max(1, int(min_warps))

        self.inflight: Dict[object, Dict[str, object]] = {}

        self._watchdog_timeout_s = max(1.0, float(watchdog_timeout_s))
        self._watchdog_poll_s = max(0.1, float(watchdog_poll_s))
        self._watchdog_thread: threading.Thread | None = None

        self._last_score_ts: Dict[str, float] = {}
        self._last_progress_ts: Dict[str, float] = {}
        self._last_stuck_alert_ts: Dict[str, float] = {}

        if not hasattr(self.store, "instance") or not isinstance(getattr(self.store, "instance"), dict):
            self.store.instance = {}

        self._install_signal_handlers()

        self._snapshot = None
        if self.resume and hasattr(self.store, "load_instance_point"):
            ip = self.store.load_instance_point()
            snap = self.resumer.resolve(ip)
            self._snapshot = snap
            self.global_step = snap["global_step"]
            self.cycle = snap["cycle"]
            self.last_eid = snap["last_eid"]
            self.store.instance = dict(snap["instance"])

        self._spawn_initial_warps(base_graphs)

        if self._snapshot:
            self._apply_warps_state(self._snapshot.get("warps_state", {}))

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
    # signals / watchdog
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self._handle_stop_signal)
                if hasattr(signal, "SIGTERM"):
                    signal.signal(signal.SIGTERM, self._handle_stop_signal)
        except Exception:
            pass

    def _handle_stop_signal(self, signum, frame) -> None:
        self._emit("ENGINE_SIGNAL", {"signal": int(signum), "kind": "stop"})
        self._stop_requested.set()
        self.active = False

    def _start_watchdog(self) -> None:
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="dimensional-core-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while self.active and not self._stop_requested.is_set():
            now = time.time()
            alerts: List[Tuple[str, float, str]] = []

            with self.lock:
                for wid in list(self.warps.keys()):
                    last_score = self._last_score_ts.get(wid)
                    last_progress = self._last_progress_ts.get(wid, last_score or now)

                    score_age = (now - last_score) if last_score is not None else None
                    progress_age = now - last_progress

                    should_alert = False
                    reason = ""
                    age = 0.0

                    if score_age is not None and score_age >= self._watchdog_timeout_s:
                        should_alert = True
                        reason = "no_score"
                        age = score_age
                    elif progress_age >= self._watchdog_timeout_s:
                        should_alert = True
                        reason = "no_progress"
                        age = progress_age

                    if not should_alert:
                        continue

                    last_alert = self._last_stuck_alert_ts.get(wid, 0.0)
                    if (now - last_alert) >= self._watchdog_timeout_s:
                        alerts.append((wid, age, reason))
                        self._last_stuck_alert_ts[wid] = now

            for wid, age, reason in alerts:
                self._emit(
                    "WARP_STUCK_ALERT",
                    {
                        "warp": wid,
                        "age_s": round(age, 3),
                        "reason": reason,
                    },
                )
                try:
                    self._save_warp_point(wid, tag="WATCHDOG")
                except Exception:
                    pass

            time.sleep(self._watchdog_poll_s)

    # ------------------------------------------------------------------
    # graph / warp init
    # ------------------------------------------------------------------

    def _normalize_base_graphs(self, base_graphs):
        bg = base_graphs
        target = self._target_warps

        if bg is None:
            gids = [f"G{i}" for i in range(max(8, target or 2))]
            if target:
                return self._partition_gids(gids, target)
            return {"W0": gids[:4], "W1": gids[4:8]}

        if isinstance(bg, int):
            n = max(1, int(bg))
            gids = [f"G{i}" for i in range(n)]
            if target:
                return self._partition_gids(gids, target)
            mid = max(1, len(gids) // 2)
            return {"W0": gids[:mid], "W1": gids[mid:]}

        if isinstance(bg, (list, tuple)):
            gids = list(bg) or ["G0"]
            if target:
                return self._partition_gids(gids, target)
            mid = max(1, len(gids) // 2)
            return {"W0": gids[:mid], "W1": gids[mid:]}

        if isinstance(bg, dict):
            out = {}
            for wid, lane_gids in bg.items():
                out[str(wid)] = list(lane_gids) if isinstance(lane_gids, (list, tuple)) else [str(lane_gids)]
            if target and len(out) < target:
                for i in range(len(out), target):
                    out[f"W{i}"] = []
            return out or {"W0": ["G0"], "W1": []}

        gids = [f"G{i}" for i in range(max(8, target or 2))]
        if target:
            return self._partition_gids(gids, target)
        return {"W0": gids[:4], "W1": gids[4:8]}

    def _partition_gids(self, gids: List[str], warp_count: int) -> Dict[str, List[str]]:
        warp_count = max(1, int(warp_count))
        out: Dict[str, List[str]] = {f"W{i}": [] for i in range(warp_count)}
        for i, gid in enumerate(gids):
            out[f"W{i % warp_count}"].append(gid)
        return out

    def _spawn_initial_warps(self, base_graphs=None):
        bg = self._normalize_base_graphs(base_graphs)
        for wid, lane_gids in bg.items():
            g = build_warp_graph(wid, lane_gids, self.store.instance, lanes=self.lanes)
            if not hasattr(g, "_local"):
                g._local = {}
            self.warps[wid] = g
            self.scheduler.register(wid)
            now = time.time()
            self._last_score_ts[wid] = now
            self._last_progress_ts[wid] = now

    def _spawn_warp(self, wid: str, lane_gids: List[str] | None = None) -> None:
        lane_gids = list(lane_gids or [])
        g = build_warp_graph(wid, lane_gids, self.store.instance, lanes=self.lanes)
        if not hasattr(g, "_local"):
            g._local = {}
        with self.lock:
            self.warps[wid] = g
            self.scheduler.register(wid)
            now = time.time()
            self._last_score_ts[wid] = now
            self._last_progress_ts[wid] = now
        self._emit("WARP_SPAWNED", {"warp": wid, "lane_gids": lane_gids})

    def _remove_warp_if_idle(self, wid: str) -> bool:
        with self.lock:
            if wid not in self.warps:
                return False

            for meta in self.inflight.values():
                if meta.get("warp_id") == wid:
                    return False

            g = self.warps[wid]
            ready_count = len(getattr(g, "ready", []) or [])
            if ready_count > 0:
                return False

            if len(self.warps) <= self._min_warps:
                return False

            self.warps.pop(wid, None)
            self.scheduler.unregister(wid)
            self._last_score_ts.pop(wid, None)
            self._last_progress_ts.pop(wid, None)
            self._last_stuck_alert_ts.pop(wid, None)

        self._emit("WARP_REMOVED", {"warp": wid})
        return True

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

            last_score_ts = st.get("last_score_ts")
            if isinstance(last_score_ts, (int, float)):
                self._last_score_ts[wid] = float(last_score_ts)

            nodes = st.get("nodes")
            if isinstance(nodes, dict) and hasattr(g, "nodes") and isinstance(g.nodes, dict):
                for nid, nd in nodes.items():
                    if nid in g.nodes:
                        n = g.nodes[nid]
                        n.status = nd.get("status", getattr(n, "status", "PENDING"))
                        n.result = nd.get("result", getattr(n, "result", None))

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
        with self.lock:
            warps_snapshot = {}
            instance_snapshot = dict(getattr(self.store, "instance", {}) or {})

            for wid, g in self.warps.items():
                nodes_state = {}
                if hasattr(g, "nodes") and isinstance(g.nodes, dict):
                    for nid, n in g.nodes.items():
                        nodes_state[nid] = {
                            "status": getattr(n, "status", "PENDING"),
                            "result": getattr(n, "result", None),
                        }

                warps_snapshot[wid] = {
                    "local": dict(getattr(g, "_local", {}) or {}),
                    "nodes": nodes_state,
                    "last_score_ts": self._last_score_ts.get(wid),
                }

            ip = {
                "global_step": int(self.global_step),
                "cycle": int(self.cycle),
                "last_eid": int(self.last_eid),
                "instance": instance_snapshot,
                "warps_state": warps_snapshot,
                "graphs_state": warps_snapshot,
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
                raise RuntimeError(f"Unsupported node op={op} id={getattr(gnode, 'id', '?')}")

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

            raise RuntimeError(f"Unknown VISA stage={stage} id={getattr(gnode, 'id', '?')}")

        return task

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def run_forever(self):
        self._start_watchdog()
        self._emit("ENGINE_START", {"resume": self.resume})

        try:
            while self.active and not self._stop_requested.is_set():
                self.global_step += 1

                self._adjust_warp_count()
                self._dispatch()

                finished = self._collect_finished()
                if finished:
                    self._apply_results(finished)

                if self.global_step % 5 == 0:
                    for wid in list(self.warps.keys()):
                        self._save_warp_point(wid, tag="PERIODIC")

                max_steps = getattr(self.store, "max_steps", None)
                if isinstance(max_steps, int) and max_steps > 0 and self.global_step >= max_steps:
                    self._stop_requested.set()
                    break

                time.sleep(0.01)
        finally:
            self._stop_all()

    def _adjust_warp_count(self) -> None:
        if self._target_warps is None:
            return

        with self.lock:
            current_warps = list(self.warps.keys())

        desired = max(self._min_warps, self._target_warps)

        if len(current_warps) < desired:
            for i in range(len(current_warps), desired):
                wid = f"W{i}"
                if wid not in self.warps:
                    self._spawn_warp(wid, [])
        elif len(current_warps) > desired:
            return

    # ------------------------------------------------------------------
    # dispatch / collect / apply
    # ------------------------------------------------------------------

    def _dispatch(self):
        submissions: List[Tuple[str, object, object]] = []

        with self.lock:
            if len(self.inflight) >= self.max_in_flight:
                return

            capacity = self.max_in_flight - len(self.inflight)

            while capacity > 0:
                wid = self.scheduler.choose()
                if wid is None:
                    break

                graph = self.warps.get(wid)
                if graph is None:
                    break

                nodes = graph.take_ready(min(2, capacity))
                if not nodes:
                    break

                for gnode in nodes:
                    task = self._compile_task(wid, graph, gnode)
                    submissions.append((wid, gnode, task))
                    capacity -= 1
                    if capacity <= 0:
                        break

        for wid, gnode, task in submissions:
            fut = self.pool.submit(task)
            with self.lock:
                self.inflight[fut] = {
                    "warp_id": wid,
                    "node_id": str(getattr(gnode, "id", "?")),
                    "gnode": gnode,
                    "started_at": time.time(),
                }

        if submissions:
            by_warp: Dict[str, int] = {}
            for wid, _, _ in submissions:
                by_warp[wid] = by_warp.get(wid, 0) + 1

            for wid, batch_n in by_warp.items():
                self._emit(
                    "WARP_BATCH_SUBMITTED",
                    {
                        "warp": wid,
                        "batch": batch_n,
                        "inflight": len(self.inflight),
                        "priority_value": self.scheduler.priority(wid),
                    },
                )

    def _collect_finished(self):
        ready: List[Tuple[object, Dict[str, object]]] = []

        with self.lock:
            for fut, meta in list(self.inflight.items()):
                if fut.done():
                    ready.append((fut, meta))
                    self.inflight.pop(fut, None)

        finished = []
        for fut, meta in ready:
            try:
                res = fut.result()
            except Exception as e:
                res = {"error": str(e)}

            finished.append((
                str(meta["warp_id"]),
                meta["gnode"],
                res,
            ))

        return finished

    def _apply_results(self, finished):
        for wid, gnode, res in finished:
            if not isinstance(res, dict):
                res = {"value": res}

            graph = self.warps.get(wid)

            if graph is not None and hasattr(graph, "mark_done"):
                graph.mark_done(getattr(gnode, "id"), dict(res))

            self._last_progress_ts[wid] = time.time()

            score = res.get("score")
            if score is not None:
                self.scheduler.update_score(wid, float(score))
                self._last_score_ts[wid] = time.time()
                self._emit(
                    "WARP_SCORE",
                    {
                        "warp": wid,
                        "min_score": float(score),
                        "priority_value": self.scheduler.priority(wid),
                    },
                )

            nid = str(getattr(gnode, "id", ""))
            rearm = nid.endswith(":score")

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

            if rearm and graph is not None:
                graph.rearm(f"{wid}:step")
                graph.rearm(f"{wid}:score")

    # ------------------------------------------------------------------
    # warp points / shutdown
    # ------------------------------------------------------------------

    def _save_warp_point(self, wid, tag):
        graph = self.warps.get(wid)
        local = getattr(graph, "_local", {}) if graph is not None else {}

        self.warp_store.save(
            wid,
            {
                "warp": wid,
                "tag": tag,
                "global_step": self.global_step,
                "local": local,
                "_ts": time.time(),
            },
        )
        self._emit("WARP_POINT_SAVED", {"warp": wid, "tag": tag})

    def _stop_all(self):
        if self._shutdown_once.is_set():
            return
        self._shutdown_once.set()

        self.active = False

        warp_ids = list(self.warps.keys())
        for wid in warp_ids:
            try:
                self._save_warp_point(wid, tag="STOP")
            except Exception:
                pass

        try:
            self.pool.shutdown(wait=True)
        except Exception:
            pass

        try:
            self._emit("STOP", {"global_step": self.global_step, "cycle": self.cycle})
        except Exception:
            pass

        try:
            self._commit_instance_point(reason="stop")
        except Exception:
            pass

        try:
            if hasattr(self.store, "flush"):
                self.store.flush()
            elif hasattr(self.store, "flush_events"):
                self.store.flush_events()
        except Exception:
            pass