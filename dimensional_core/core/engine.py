# dimensional_core/core/engine.py
"""
Dimensional Core execution engine — C20/C21/C22/C23/C24.

Architecture
------------
Each *warp* owns a 3D cyclic task graph (X→Y→Z→X→…).  The engine manages
N warps in a thread pool, dispatches ready nodes, collects results, and
applies them — triggering rollback when Dimension Z verification fails.

Lock hierarchy (must always be acquired in this order, never reversed)
----------------------------------------------------------------------
  engine._lock  (RLock)
    └─ scheduler._lock  (threading.Lock, internal)
         └─ store._events_lock  (threading.Lock, internal)
              └─ store._eid_lock  (threading.Lock, internal)

The engine never holds its lock while waiting for futures (thread pool
results), avoiding deadlocks between the pool and the event machinery.

Rollback model
--------------
1. After Dimension X completes, the engine saves a rollback snapshot of the
   warp's _local state (pre-Y state) via ``graph.save_rollback_snapshot()``.
2. If Dimension Z returns ``verified=False`` the engine calls
   ``graph.rollback()``, which restores _local from the snapshot and re-queues
   X for re-execution.
3. All rollback events are logged so replay can reconstruct the same path.
"""
from __future__ import annotations

import copy
import logging
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

from .task_graph import TaskGraph3D, build_3d_graph, Dimension
from .dimensions import DimXOperator, DimYOperator, DimZOperator
from .priority_scheduler import HeapWarpScheduler
from .state_store import StateStore
from .warp_store import WarpStore
from .replay_controller import ReplayController
from .resume_controller import ResumeController
from .triggers import TriggerConfig

logger = logging.getLogger(__name__)

# ── Tuning constants ────────────────────────────────────────────────────────────
_CHECKPOINT_EVERY_N_STEPS = 5   # periodic warp-point save interval
_MAIN_LOOP_SLEEP_S        = 0.01
_DEFAULT_LR               = 0.10
_DEFAULT_WARP_POOL_SIZE   = 8   # default logical lane count
_Y_OPCODE_GRADIENT        = "VSTEP_Y"
_Y_OPCODE_MUTATE          = "VMUTATE_Y"


class Engine:
    """
    Dimensional Core engine: C20 (event log) + C21 (scheduler) +
    C22 (resume/replay) + C23 (3D dimensions) + C24 (watchdog/monitor).

    Parameters
    ----------
    store           : StateStore instance (event log + instance point)
    writer          : event writer (SimpleWriter or equivalent)
    resume          : load latest checkpoint before running
    triggers        : flush / rotation policy
    max_workers     : thread pool size
    max_in_flight   : maximum concurrent task futures
    target_warps    : desired warp count
    min_warps       : minimum warp count (never drop below this)
    watchdog_timeout_s : seconds before a warp is flagged as STUCK
    watchdog_poll_s    : watchdog check interval
    lr              : gradient-descent learning rate (Dimension Y)
    y_opcode        : VISA opcode for Dimension Y
    """

    def __init__(
        self,
        store: StateStore,
        writer: Any,
        resume: bool = False,
        triggers: Optional[TriggerConfig] = None,
        max_workers: int = 4,
        max_in_flight: int = 8,
        target_warps: Optional[int] = None,
        min_warps: int = 1,
        watchdog_timeout_s: float = 30.0,
        watchdog_poll_s: float = 1.0,
        lr: float = _DEFAULT_LR,
        y_opcode: str = _Y_OPCODE_GRADIENT,
        **_ignored,
    ) -> None:
        self.store   = store
        self.writer  = writer
        self.resume  = bool(resume)
        self.tr_cfg  = triggers or TriggerConfig()
        self.lr      = float(lr)
        self.y_opcode = str(y_opcode)

        # ── Sub-systems ────────────────────────────────────────────────────────
        self.scheduler   = HeapWarpScheduler()
        self.warp_store  = WarpStore(root_dir=getattr(store, "_state_dir", "dimensional_core/state"))
        self.pool        = ThreadPoolExecutor(max_workers=int(max_workers))
        self.resumer     = ResumeController()
        self.replayer    = ReplayController()

        # ── Dimension operators ────────────────────────────────────────────────
        self._dim_ops: Dict[str, Any] = {
            "X": DimXOperator(),
            "Y": DimYOperator(),
            "Z": DimZOperator(),
        }

        # ── Config ─────────────────────────────────────────────────────────────
        self.max_in_flight = max(1, int(max_in_flight))
        self._target_warps = max(1, int(target_warps)) if target_warps else None
        self._min_warps    = max(1, int(min_warps))
        self._watchdog_timeout_s = max(1.0, float(watchdog_timeout_s))
        self._watchdog_poll_s    = max(0.1, float(watchdog_poll_s))

        # ── Runtime state ──────────────────────────────────────────────────────
        self._lock = threading.RLock()

        self.global_step: int = 0
        self.cycle: int       = 0
        self.last_eid: int    = 0
        self._events_since_commit: int = 0
        self._commit_count: int        = 0

        self.active = True
        self._stop_requested = threading.Event()
        self._shutdown_once  = threading.Event()

        self.warps: Dict[str, TaskGraph3D] = {}
        self.inflight: Dict[Any, Dict[str, Any]] = {}

        # Watchdog bookkeeping
        self._last_z_ts: Dict[str, float]     = {}  # last successful Z time
        self._last_prog_ts: Dict[str, float]  = {}  # last any-node completion
        self._last_alert_ts: Dict[str, float] = {}

        # ── Shared instance state ──────────────────────────────────────────────
        if not isinstance(getattr(store, "instance", None), dict):
            store.instance = {}

        # ── Resume ────────────────────────────────────────────────────────────
        self._snapshot: Optional[Dict[str, Any]] = None
        if self.resume and hasattr(store, "load_instance_point"):
            ip   = store.load_instance_point()
            snap = self.resumer.resolve(ip)
            self._snapshot   = snap
            self.global_step = snap["global_step"]
            self.cycle       = snap["cycle"]
            self.last_eid    = snap["last_eid"]
            store.instance   = dict(snap["instance"])

        # ── Build initial warps ────────────────────────────────────────────────
        self._spawn_initial_warps()

        # ── Restore warp state from snapshot ──────────────────────────────────
        if self._snapshot:
            self._apply_warps_state(self._snapshot.get("warps_state", {}))

        # ── Replay events after last_eid ───────────────────────────────────────
        if self.resume:
            self._replay_tail()

        # ── Signal handlers ────────────────────────────────────────────────────
        self._install_signal_handlers()

        # ── Watchdog thread (started in run_forever) ───────────────────────────
        self._watchdog_thread: Optional[threading.Thread] = None

    # ═══════════════════════════════════════════════════════════════════════════
    # Signal / watchdog
    # ═══════════════════════════════════════════════════════════════════════════

    def _install_signal_handlers(self) -> None:
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, self._handle_stop)
                if hasattr(signal, "SIGTERM"):
                    signal.signal(signal.SIGTERM, self._handle_stop)
        except Exception:
            pass

    def _handle_stop(self, signum: int, frame: Any) -> None:
        self._emit("ENGINE_SIGNAL", {"signal": signum, "kind": "stop"})
        self._stop_requested.set()
        self.active = False

    def _start_watchdog(self) -> None:
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        t = threading.Thread(
            target=self._watchdog_loop,
            name="dc-watchdog",
            daemon=True,
        )
        t.start()
        self._watchdog_thread = t

    def _watchdog_loop(self) -> None:
        while self.active and not self._stop_requested.is_set():
            now = time.time()
            alerts: List[Tuple[str, float, str]] = []

            with self._lock:
                for wid in list(self.warps.keys()):
                    last_z    = self._last_z_ts.get(wid)
                    last_prog = self._last_prog_ts.get(wid, last_z or now)

                    z_age   = (now - last_z) if last_z is not None else None
                    prog_age = now - last_prog

                    reason = age = None
                    if z_age is not None and z_age >= self._watchdog_timeout_s:
                        reason, age = "no_z_pass", z_age
                    elif prog_age >= self._watchdog_timeout_s:
                        reason, age = "no_progress", prog_age

                    if reason is None:
                        continue

                    last_alert = self._last_alert_ts.get(wid, 0.0)
                    if (now - last_alert) >= self._watchdog_timeout_s:
                        alerts.append((wid, age, reason))
                        self._last_alert_ts[wid] = now

            for wid, age, reason in alerts:
                self._emit("WARP_STUCK_ALERT", {
                    "warp": wid,
                    "age_s": round(age, 2),
                    "reason": reason,
                })
                try:
                    self._save_warp_point(wid, "WATCHDOG")
                except Exception:
                    pass

            time.sleep(self._watchdog_poll_s)

    # ═══════════════════════════════════════════════════════════════════════════
    # Warp lifecycle
    # ═══════════════════════════════════════════════════════════════════════════

    def _make_lane_gids(self, wid: str, n: int = _DEFAULT_WARP_POOL_SIZE) -> List[str]:
        """Generate deterministic lane IDs for a warp."""
        return [f"{wid}_G{i}" for i in range(max(1, n))]

    def _spawn_initial_warps(self) -> None:
        target = self._target_warps or 2
        for i in range(target):
            wid = f"W{i}"
            lane_gids = self._make_lane_gids(wid, n=4)
            self._spawn_warp(wid, lane_gids)

    def _spawn_warp(self, wid: str, lane_gids: Optional[List[str]] = None) -> None:
        lane_gids = list(lane_gids or self._make_lane_gids(wid))
        g = build_3d_graph(
            wid=wid,
            lane_gids=lane_gids,
            shared_instance=self.store.instance,
            lr=self.lr,
            y_opcode=self.y_opcode,
        )
        now = time.time()
        with self._lock:
            self.warps[wid] = g
            self.scheduler.register(wid)
            self._last_z_ts[wid]    = now
            self._last_prog_ts[wid] = now

        self._emit("WARP_SPAWNED", {"warp": wid, "lane_gids": lane_gids})
        logger.debug("spawned warp %s lanes=%s", wid, lane_gids)

    def _remove_warp(self, wid: str) -> None:
        with self._lock:
            self.warps.pop(wid, None)
            self.scheduler.unregister(wid)
            self._last_z_ts.pop(wid, None)
            self._last_prog_ts.pop(wid, None)
            self._last_alert_ts.pop(wid, None)
        self._emit("WARP_REMOVED", {"warp": wid})

    def _adjust_warp_count(self) -> None:
        """Add or remove warps to match the target count."""
        if self._target_warps is None:
            return

        desired = max(self._min_warps, self._target_warps)
        with self._lock:
            current = list(self.warps.keys())

        if len(current) < desired:
            for i in range(len(current), desired):
                wid = f"W{i}"
                if wid not in self.warps:
                    self._spawn_warp(wid)

        elif len(current) > desired:
            # Only remove idle warps
            with self._lock:
                candidates = [
                    w for w in current
                    if w not in {m["warp_id"] for m in self.inflight.values()}
                ]
            for wid in candidates[desired:]:
                if len(self.warps) <= self._min_warps:
                    break
                self._remove_warp(wid)

    def _apply_warps_state(self, warps_state: Dict[str, Any]) -> None:
        """Restore per-warp state from a checkpoint snapshot."""
        if not isinstance(warps_state, dict):
            return
        for wid, st in warps_state.items():
            g = self.warps.get(wid)
            if g is None:
                continue
            local = st.get("local")
            if isinstance(local, dict):
                g._local = copy.deepcopy(local)

            lts = st.get("last_z_ts")
            if isinstance(lts, (int, float)):
                self._last_z_ts[wid] = float(lts)

            nodes = st.get("nodes")
            if isinstance(nodes, dict) and isinstance(g.nodes, dict):
                for nid, nd in nodes.items():
                    if nid in g.nodes:
                        n = g.nodes[nid]
                        n.status = nd.get("status", n.status)
                        n.result = nd.get("result")
                # Re-seed ready set from restored node states
                g.finalize()

    # ═══════════════════════════════════════════════════════════════════════════
    # Replay
    # ═══════════════════════════════════════════════════════════════════════════

    def _replay_tail(self) -> None:
        """Replay events written after the last checkpoint."""
        if hasattr(self.store, "iter_events_after_fast"):
            events = self.store.iter_events_after_fast(self.last_eid)
        elif hasattr(self.store, "iter_events_after"):
            events = self.store.iter_events_after(self.last_eid)
        else:
            events = []

        for e in events:
            eid = int(e.get("eid", 0) or 0)
            if eid > self.last_eid:
                self.last_eid = eid
            self.replayer.apply(self, e)

    # ═══════════════════════════════════════════════════════════════════════════
    # Event emission + checkpoints
    # ═══════════════════════════════════════════════════════════════════════════

    def _emit(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
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
            self._save_instance_point(reason=f"every_{self.tr_cfg.commit_every_n_events}_events")

    def _save_instance_point(self, reason: str) -> None:
        """Atomically persist the full engine state."""
        with self._lock:
            warps_snap: Dict[str, Any] = {}
            for wid, g in self.warps.items():
                nodes_snap = {}
                for nid, n in g.nodes.items():
                    nodes_snap[nid] = {
                        "status": str(n.status),
                        "result": n.result,
                    }
                warps_snap[wid] = {
                    "local": copy.deepcopy(g._local),
                    "nodes": nodes_snap,
                    "last_z_ts": self._last_z_ts.get(wid),
                    "verify_pass": g.verify_pass_count,
                    "verify_fail": g.verify_fail_count,
                }
            instance_snap = dict(getattr(self.store, "instance", {}) or {})

        ip = {
            "global_step": int(self.global_step),
            "cycle": int(self.cycle),
            "last_eid": int(self.last_eid),
            "instance": instance_snap,
            "warps_state": warps_snap,
            "graphs_state": warps_snap,   # legacy alias
            "reason": str(reason),
            "_ts": time.time(),
        }

        if hasattr(self.store, "save_instance_point"):
            self.store.save_instance_point(ip)
        elif hasattr(self.writer, "submit_instance_point"):
            self.writer.submit_instance_point(ip)

        self._events_since_commit = 0
        self._commit_count += 1

        if (
            self.tr_cfg.rotate_every_n_commits > 0
            and self._commit_count % int(self.tr_cfg.rotate_every_n_commits) == 0
            and hasattr(self.store, "rotate_events")
        ):
            self.store.rotate_events(keep_archives=int(self.tr_cfg.keep_archives))

    def _save_warp_point(self, wid: str, tag: str) -> None:
        g = self.warps.get(wid)
        local = copy.deepcopy(g._local) if g else {}
        self.warp_store.save(wid, {
            "warp": wid,
            "tag": tag,
            "global_step": self.global_step,
            "local": local,
            "_ts": time.time(),
        })
        self._emit("WARP_POINT_SAVED", {"warp": wid, "tag": tag})

    # ═══════════════════════════════════════════════════════════════════════════
    # Task compilation
    # ═══════════════════════════════════════════════════════════════════════════

    def _compile_task(self, wid: str, graph: TaskGraph3D, node: Any):
        """
        Return a zero-argument callable that executes *node* on the thread pool.

        The dimension operator is chosen by parsing the node's dimension field.
        All state referenced inside the closure is snapshotted at compile time
        so the closure is self-contained and thread-safe.
        """
        params    = dict(getattr(node, "params", {}) or {})
        dim_key   = str(node.dimension.value if hasattr(node.dimension, "value") else node.dimension)
        op        = self._dim_ops.get(dim_key)
        lane_gids = list(params.get("lane_gids", []))
        args      = {k: v for k, v in params.items() if k not in ("stage", "lane_gids")}
        step      = self.global_step

        # Snapshot local at compile time — the task runs in a worker thread and
        # must not touch the live local dict concurrently with other tasks.
        local_snapshot = copy.deepcopy(graph._local)

        # Instance is shared + mutable; operators read from it, rarely write.
        instance_ref = self.store.instance

        node_id = str(node.id)

        def task() -> Dict[str, Any]:
            if op is None:
                raise RuntimeError(f"No operator for dimension {dim_key!r}")
            try:
                node.status = "RUNNING"
            except Exception:
                pass
            return op.execute(
                node_id=node_id,
                warp_id=wid,
                instance=instance_ref,
                local=local_snapshot,
                lane_gids=lane_gids,
                args=args,
                step=step,
            )

        # Attach snapshot so engine can merge it back after the task completes
        task._local_snapshot = local_snapshot  # type: ignore[attr-defined]
        task._dim_key = dim_key                # type: ignore[attr-defined]
        task._node_id = node_id               # type: ignore[attr-defined]
        return task

    # ═══════════════════════════════════════════════════════════════════════════
    # Main loop
    # ═══════════════════════════════════════════════════════════════════════════

    def run_forever(self) -> None:
        """Run the engine until stop is requested or max_steps reached."""
        self._start_watchdog()
        self._emit("ENGINE_START", {"resume": self.resume, "warps": list(self.warps.keys())})
        logger.info("engine started warps=%s", list(self.warps.keys()))

        try:
            while self.active and not self._stop_requested.is_set():
                self.global_step += 1
                self.cycle       += 1

                self._adjust_warp_count()
                self._dispatch()

                finished = self._collect_finished()
                if finished:
                    self._apply_results(finished)

                # Periodic warp-point saves
                if self.global_step % _CHECKPOINT_EVERY_N_STEPS == 0:
                    for wid in list(self.warps.keys()):
                        try:
                            self._save_warp_point(wid, "PERIODIC")
                        except Exception as exc:
                            logger.warning("warp_point save failed wid=%s: %s", wid, exc)

                max_steps = getattr(self.store, "max_steps", None)
                if isinstance(max_steps, int) and max_steps > 0 and self.global_step >= max_steps:
                    logger.info("max_steps=%d reached, stopping", max_steps)
                    self._stop_requested.set()
                    break

                time.sleep(_MAIN_LOOP_SLEEP_S)

        finally:
            self._stop_all()

    # ═══════════════════════════════════════════════════════════════════════════
    # Dispatch / collect / apply
    # ═══════════════════════════════════════════════════════════════════════════

    def _dispatch(self) -> None:
        submissions: List[Tuple[str, Any, Any]] = []

        with self._lock:
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
                for node in nodes:
                    task = self._compile_task(wid, graph, node)
                    submissions.append((wid, node, task))
                    capacity -= 1

        for wid, node, task in submissions:
            fut = self.pool.submit(task)
            with self._lock:
                self.inflight[fut] = {
                    "warp_id": wid,
                    "node_id": str(node.id),
                    "node": node,
                    "task": task,
                    "started_at": time.time(),
                }

        if submissions:
            by_warp: Dict[str, int] = {}
            for wid, _, _ in submissions:
                by_warp[wid] = by_warp.get(wid, 0) + 1
            for wid, n in by_warp.items():
                self._emit("WARP_BATCH_SUBMITTED", {
                    "warp": wid,
                    "batch": n,
                    "inflight": len(self.inflight),
                    "priority_value": self.scheduler.priority(wid),
                })

    def _collect_finished(self) -> List[Tuple[str, Any, Dict[str, Any], Any]]:
        ready = []
        with self._lock:
            for fut, meta in list(self.inflight.items()):
                if fut.done():
                    ready.append((fut, meta))
                    self.inflight.pop(fut, None)

        finished = []
        for fut, meta in ready:
            try:
                res = fut.result()
            except Exception as exc:
                logger.warning("node %s raised: %s", meta["node_id"], exc)
                res = {"error": str(exc), "verified": False, "rollback": False}
            if not isinstance(res, dict):
                res = {"value": res}
            finished.append((meta["warp_id"], meta["node"], res, meta["task"]))

        return finished

    def _apply_results(self, finished: List[Tuple[str, Any, Dict[str, Any], Any]]) -> None:
        for wid, node, res, task in finished:
            graph = self.warps.get(wid)
            if graph is None:
                continue

            now    = time.time()
            nid    = str(node.id)
            dim    = str(node.dimension.value if hasattr(node.dimension, "value") else node.dimension)

            # ── Merge the task's local snapshot back into the live graph._local
            task_local = getattr(task, "_local_snapshot", None)
            if isinstance(task_local, dict):
                graph._local.update(task_local)

            self._last_prog_ts[wid] = now

            # ── Dimension-specific post-processing ─────────────────────────────
            if dim == "X":
                # Save rollback snapshot now — before Y can modify _local
                graph.save_rollback_snapshot()
                graph.mark_done(nid, res)
                self._emit("NODE_RESULT", {
                    "warp": wid, "node_id": nid, "dimension": "X",
                    "res": res, "rearm": False,
                })

            elif dim == "Y":
                graph.mark_done(nid, res)
                self._emit("NODE_RESULT", {
                    "warp": wid, "node_id": nid, "dimension": "Y",
                    "res": res, "rearm": False,
                })

            elif dim == "Z":
                verified = bool(res.get("verified", False))
                rollback = bool(res.get("rollback", not verified))

                if verified:
                    graph.mark_done(nid, res)
                    score = res.get("score", res.get("min_score"))
                    if score is not None:
                        self.scheduler.update_score(wid, float(score))
                        self._last_z_ts[wid] = now
                        self._emit("WARP_SCORE", {
                            "warp": wid,
                            "min_score": float(score),
                            "priority_value": self.scheduler.priority(wid),
                            "verify_pass": graph.verify_pass_count + 1,
                        })

                    self._emit("NODE_RESULT", {
                        "warp": wid, "node_id": nid, "dimension": "Z",
                        "res": res, "rearm": True,
                    })

                    # Rearm for next X→Y→Z cycle
                    graph.rearm_cycle()

                else:
                    # Verification failure — rollback to X
                    graph.mark_failed(nid, res.get("reason", "unknown"))
                    self._emit("VERIFICATION_FAILED", {
                        "warp": wid,
                        "node_id": nid,
                        "reason": res.get("reason", "unknown"),
                        "min_score": res.get("min_score"),
                        "verify_fail": graph.verify_fail_count + 1,
                    })

                    if rollback:
                        rollback_target = graph.rollback()
                        self._emit("WARP_ROLLED_BACK", {
                            "warp": wid,
                            "rolled_back_to": rollback_target,
                            "verify_fail": graph.verify_fail_count,
                        })
                        logger.debug("warp %s rolled back to %s", wid, rollback_target)
                    else:
                        # No rollback — just rearm so execution continues
                        graph.rearm_cycle()

    # ═══════════════════════════════════════════════════════════════════════════
    # Shutdown
    # ═══════════════════════════════════════════════════════════════════════════

    def _stop_all(self) -> None:
        if self._shutdown_once.is_set():
            return
        self._shutdown_once.set()
        self.active = False

        for wid in list(self.warps.keys()):
            try:
                self._save_warp_point(wid, "STOP")
            except Exception:
                pass

        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            self.pool.shutdown(wait=True)
        except Exception:
            pass

        try:
            self._emit("STOP", {"global_step": self.global_step, "cycle": self.cycle})
        except Exception:
            pass

        try:
            self._save_instance_point(reason="stop")
        except Exception:
            pass

        try:
            if hasattr(self.store, "flush"):
                self.store.flush()
        except Exception:
            pass

        logger.info("engine stopped global_step=%d", self.global_step)
