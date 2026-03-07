from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from .transfer import TransferEnvelope
from .snapshot_store import SnapshotStore
from .coordinator_3d import Coordinator3D
from .verifier_z import VerifyRefinedZ
from .operators_demo import OpInitX, OpRefineY


@dataclass
class WorkerConfig:
    run_id: str
    # how many Y candidates to produce per X output
    y_branch_k: int = 2


class WarpXWorker(threading.Thread):
    def __init__(self, cfg: WorkerConfig, coord: Coordinator3D, store: SnapshotStore, q_xy: "queue.Queue[TransferEnvelope]") -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.coord = coord
        self.store = store
        self.q_xy = q_xy
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        root_id = self.coord.graph.root_mp_id
        if root_id is None:
            return

        mp = self.coord.graph.memory_points[root_id]
        self.store.verify_hash(mp.snapshot_ref, mp.state_hash)
        state = self.store.load_json(mp.snapshot_ref)

        opx = OpInitX()
        out = opx.apply(state)

        commit = self.coord.commit_next(
            axis="X",
            parent_mp_id=root_id,
            state=out.state,
            seq_end=mp.seq_end + 1,
            op_name="op_init_x",
        )

        env = TransferEnvelope(
            run_id=self.cfg.run_id,
            mp_id=commit.mp_id,
            axis_from="X",
            axis_to="Y",
            seq_end=commit.seq_end,
            snapshot_ref=commit.snapshot_ref,
            state_hash=commit.state_hash,
            op_trace=out.meta,
        )
        self.q_xy.put(env)

        while not self._stop.is_set():
            time.sleep(0.05)


class WarpYWorker(threading.Thread):
    """
    Y stage now BRANCHES: for each incoming X envelope, it produces multiple candidates.
    We tag candidates with candidate_id so Z can compare them.
    """
    def __init__(
        self,
        cfg: WorkerConfig,
        coord: Coordinator3D,
        store: SnapshotStore,
        q_xy: "queue.Queue[TransferEnvelope]",
        q_yz: "queue.Queue[TransferEnvelope]",
    ) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.coord = coord
        self.store = store
        self.q_xy = q_xy
        self.q_yz = q_yz
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                env = self.q_xy.get(timeout=0.1)
            except queue.Empty:
                continue

            self.store.verify_hash(env.snapshot_ref, env.state_hash)
            state = self.store.load_json(env.snapshot_ref)

            # Branch candidates: FAST, SAFE (extendable)
            modes = ["FAST", "SAFE"][: max(1, self.cfg.y_branch_k)]

            for idx, mode in enumerate(modes):
                opy = OpRefineY(mode=mode)
                out = opy.apply(state)

                commit = self.coord.commit_next(
                    axis="Y",
                    parent_mp_id=env.mp_id,
                    state=out.state,
                    seq_end=env.seq_end + 1,
                    op_name="op_refine_y",
                    op_params_hash=mode,
                )

                out_env = TransferEnvelope(
                    run_id=self.cfg.run_id,
                    mp_id=commit.mp_id,
                    axis_from="Y",
                    axis_to="Z",
                    seq_end=commit.seq_end,
                    snapshot_ref=commit.snapshot_ref,
                    state_hash=commit.state_hash,
                    op_trace={
                        **out.meta,
                        "candidate_id": f"{env.mp_id}:{idx}",
                        "parent_x_mp": env.mp_id,
                    },
                )
                self.q_yz.put(out_env)

            self.q_xy.task_done()


class WarpZWorker(threading.Thread):
    """
    Z stage now groups Y candidates by parent_x_mp and selects the winner.
    - Verifies each candidate
    - Among PASS candidates, pick the lowest loss
    - If all fail, emit rewind
    """
    def __init__(
        self,
        cfg: WorkerConfig,
        coord: Coordinator3D,
        store: SnapshotStore,
        q_yz: "queue.Queue[TransferEnvelope]",
        verifier: VerifyRefinedZ,
        control: "queue.Queue[dict]",
    ) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.coord = coord
        self.store = store
        self.q_yz = q_yz
        self.verifier = verifier
        self.control = control
        self._stop = threading.Event()

        # grouping buffer: parent_x_mp -> list of envs
        self._buffer: Dict[str, List[TransferEnvelope]] = {}

    def stop(self) -> None:
        self._stop.set()

    def _try_select(self, parent_x: str) -> None:
        envs = self._buffer.get(parent_x, [])
        if len(envs) < self.cfg.y_branch_k:
            return  # wait for more candidates

        # verify all candidates
        passed: List[tuple[float, TransferEnvelope]] = []
        failed_reports = []

        for env in envs:
            self.store.verify_hash(env.snapshot_ref, env.state_hash)
            state = self.store.load_json(env.snapshot_ref)

            rewind_default = self.coord.latest_verified_ancestor(env.mp_id)
            ctx = {
                "prev_loss": env.op_trace.get("prev_loss"),
                "rewind_default_mp_id": rewind_default,
            }
            report = self.verifier.verify(state, ctx)

            if report.status == "PASS":
                metrics = state.get("metrics", {})
                loss = float(metrics.get("loss", 1e9))
                passed.append((loss, env))
            else:
                failed_reports.append((env, report))

        # Choose winner among PASS candidates
        if passed:
            passed.sort(key=lambda x: x[0])
            best_loss, best_env = passed[0]
            self.coord.mark_verified(best_env.mp_id)
            self.control.put({
                "type": "VERIFIED",
                "mp_id": best_env.mp_id,
                "parent_x_mp": parent_x,
                "best_loss": best_loss,
                "best_mode": best_env.op_trace.get("mode"),
            })
        else:
            # all failed -> rewind to latest verified ancestor of parent_x
            rewind_to = self.coord.latest_verified_ancestor(parent_x)
            self.coord.invalidate_descendants(rewind_to)
            # report first failure as representative
            env0, rep = failed_reports[0] if failed_reports else (None, None)
            self.control.put({
                "type": "REWIND",
                "rewind_to": rewind_to,
                "failed_parent_x": parent_x,
                "report": rep,
            })

        # clear buffer for this group
        self._buffer.pop(parent_x, None)

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                env = self.q_yz.get(timeout=0.1)
            except queue.Empty:
                continue

            parent_x = str(env.op_trace.get("parent_x_mp", ""))
            if not parent_x:
                self.q_yz.task_done()
                continue

            self._buffer.setdefault(parent_x, []).append(env)
            self._try_select(parent_x)

            self.q_yz.task_done()