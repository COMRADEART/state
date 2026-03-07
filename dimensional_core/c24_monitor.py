from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
WHITE = "\033[37m"

CLEAR = "\033[2J"
HOME = "\033[H"


def color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


class MonitorState:
    def __init__(self) -> None:
        self.last_eid = 0
        self.engine_started = False
        self.stop_seen = False

        self.warp_scores: Dict[str, float] = {}
        self.last_score_ts: Dict[str, float] = {}
        self.last_event_ts: Dict[str, float] = {}
        self.last_alert_ts: Dict[str, float] = {}
        self.warp_status: Dict[str, str] = defaultdict(lambda: "IDLE")

        self.node_events_window: deque[float] = deque(maxlen=5000)
        self.per_warp_node_events: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=2000))

        self.priority_values: Dict[str, float] = {}
        self.global_step = 0
        self.cycle = 0

    def apply_event(self, e: Dict[str, Any]) -> None:
        eid = int(e.get("eid", 0) or 0)
        ts = float(e.get("_ts", time.time()) or time.time())
        name = str(e.get("type", ""))
        payload = e.get("payload", {}) or {}

        self.last_eid = max(self.last_eid, eid)
        self.global_step = int(payload.get("global_step", self.global_step) or self.global_step)
        self.cycle = int(payload.get("cycle", self.cycle) or self.cycle)

        if name == "ENGINE_START":
            self.engine_started = True
            self.stop_seen = False
        elif name == "STOP":
            self.stop_seen = True

        wid = payload.get("warp")
        if isinstance(wid, str):
            self.last_event_ts[wid] = ts

        if name == "WARP_BATCH_SUBMITTED":
            if isinstance(wid, str):
                self.warp_status[wid] = "RUNNING"
                pv = payload.get("priority_value")
                if pv is not None:
                    try:
                        self.priority_values[wid] = float(pv)
                    except Exception:
                        pass

        elif name == "WARP_SCORE":
            if isinstance(wid, str):
                self.warp_status[wid] = "SCORING"
                try:
                    self.warp_scores[wid] = float(payload.get("min_score"))
                except Exception:
                    pass
                self.last_score_ts[wid] = ts
                pv = payload.get("priority_value")
                if pv is not None:
                    try:
                        self.priority_values[wid] = float(pv)
                    except Exception:
                        pass

        elif name == "NODE_RESULT":
            if isinstance(wid, str):
                self.warp_status[wid] = "ACTIVE"
                self.node_events_window.append(ts)
                self.per_warp_node_events[wid].append(ts)

        elif name == "WARP_STUCK_ALERT":
            if isinstance(wid, str):
                self.warp_status[wid] = "STUCK"
                self.last_alert_ts[wid] = ts

        elif name == "WARP_REMOVED":
            if isinstance(wid, str):
                self.warp_status[wid] = "REMOVED"

        elif name == "WARP_SPAWNED":
            if isinstance(wid, str):
                self.warp_status[wid] = "SPAWNED"

        elif name == "WARP_POINT_SAVED":
            if isinstance(wid, str) and self.warp_status[wid] not in ("STUCK", "REMOVED"):
                self.warp_status[wid] = "CHECKPOINT"

    def throughput_eps(self, horizon_s: float = 10.0) -> float:
        now = time.time()
        while self.node_events_window and (now - self.node_events_window[0]) > horizon_s:
            self.node_events_window.popleft()
        return len(self.node_events_window) / max(horizon_s, 1e-9)

    def throughput_eps_warp(self, wid: str, horizon_s: float = 10.0) -> float:
        dq = self.per_warp_node_events[wid]
        now = time.time()
        while dq and (now - dq[0]) > horizon_s:
            dq.popleft()
        return len(dq) / max(horizon_s, 1e-9)


def load_instance_point(instance_path: Path) -> Dict[str, Any] | None:
    for _ in range(5):
        try:
            if not instance_path.exists():
                return None
            with open(instance_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except PermissionError:
            time.sleep(0.03)
        except json.JSONDecodeError:
            time.sleep(0.03)
        except Exception:
            return None
    return None


def get_file_signature(path: Path) -> Tuple[int, int]:
    """
    Returns (size, mtime_ns). Good enough to detect replace/truncate on Windows.
    """
    try:
        st = path.stat()
        return int(st.st_size), int(st.st_mtime_ns)
    except Exception:
        return 0, 0


def read_new_events(
    events_path: Path,
    last_pos: int,
    last_sig: Tuple[int, int],
) -> Tuple[List[Dict[str, Any]], int, Tuple[int, int]]:
    """
    Rotation-aware reader.
    If the file is replaced or truncated, restart from position 0.
    """
    if not events_path.exists():
        return [], 0, (0, 0)

    cur_sig = get_file_signature(events_path)
    cur_size = cur_sig[0]

    # file replaced/truncated
    if cur_sig != last_sig and cur_size < last_pos:
        last_pos = 0
    elif cur_size < last_pos:
        last_pos = 0

    events: List[Dict[str, Any]] = []

    for _ in range(3):
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                f.seek(last_pos)
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        continue
                last_pos = f.tell()
            return events, last_pos, get_file_signature(events_path)
        except PermissionError:
            time.sleep(0.03)
        except Exception:
            break

    return events, last_pos, get_file_signature(events_path)


def render_dashboard(state: MonitorState, instance_path: Path) -> str:
    ip = load_instance_point(instance_path) or {}
    warps_state = ip.get("warps_state") or ip.get("graphs_state") or {}

    warp_ids = sorted(set(list(warps_state.keys()) + list(state.warp_scores.keys()) + list(state.warp_status.keys())))

    lines: List[str] = []
    lines.append(CLEAR + HOME)
    lines.append(color("Dimensional Core C24 Monitor", BOLD + CYAN))
    lines.append(
        f"eid={state.last_eid}  step={state.global_step}  cycle={state.cycle}  "
        f"throughput={state.throughput_eps():.2f} ev/s  engine_started={state.engine_started}  stop_seen={state.stop_seen}"
    )
    lines.append("")
    lines.append(color(f"{'Warp':<8} {'Status':<12} {'Score':>10} {'Priority':>10} {'Thrpt':>8} {'LastScore':>10}", BOLD + WHITE))
    lines.append("-" * 72)

    now = time.time()
    for wid in warp_ids:
        status = state.warp_status.get(wid, "IDLE")
        score = state.warp_scores.get(wid)
        priority = state.priority_values.get(wid)
        thrpt = state.throughput_eps_warp(wid)

        last_score_ts = state.last_score_ts.get(wid)
        age = now - last_score_ts if last_score_ts else None

        if status == "STUCK":
            status_s = color(f"{status:<12}", RED + BOLD)
        elif status in ("ACTIVE", "RUNNING", "SCORING"):
            status_s = color(f"{status:<12}", GREEN)
        elif status in ("REMOVED",):
            status_s = color(f"{status:<12}", DIM + WHITE)
        else:
            status_s = color(f"{status:<12}", YELLOW)

        score_s = f"{score:10.4f}" if score is not None else f"{'-':>10}"
        prio_s = f"{priority:10.4f}" if priority is not None else f"{'-':>10}"
        thrpt_s = f"{thrpt:8.2f}"
        age_s = f"{age:10.1f}s" if age is not None else f"{'-':>10}"

        lines.append(f"{wid:<8} {status_s} {score_s} {prio_s} {thrpt_s} {age_s}")

    lines.append("")
    lines.append(color("Alerts", BOLD + YELLOW))
    any_alert = False
    for wid in warp_ids:
        age = now - state.last_score_ts[wid] if wid in state.last_score_ts else None
        if age is not None and age >= 30.0:
            any_alert = True
            lines.append(color(f"  WARP {wid} has not scored for {age:.1f}s", RED + BOLD))
    if not any_alert:
        lines.append(color("  no stuck warp alerts", GREEN))

    lines.append("")
    lines.append(color("Controls: Ctrl+C to exit monitor", DIM + WHITE))
    return "\n".join(lines)


def run_monitor(
    state_dir: str = "dimensional_core/state",
    refresh_s: float = 0.5,
    once: bool = False,
) -> None:
    state_path = Path(state_dir)
    events_path = state_path / "events.jsonl"
    instance_path = state_path / "instance_point.json"

    m = MonitorState()

    last_pos = 0
    last_sig = (0, 0)

    if events_path.exists():
        try:
            last_pos = events_path.stat().st_size
            last_sig = get_file_signature(events_path)
        except Exception:
            last_pos = 0
            last_sig = (0, 0)

    try:
        while True:
            events, last_pos, last_sig = read_new_events(events_path, last_pos, last_sig)
            for e in events:
                m.apply_event(e)

            # If event tailing stalls, still refresh step/status from instance point
            ip = load_instance_point(instance_path) or {}
            if ip:
                m.global_step = max(m.global_step, int(ip.get("global_step", 0) or 0))
                m.cycle = max(m.cycle, int(ip.get("cycle", 0) or 0))
                if str(ip.get("reason", "")) == "stop":
                    m.stop_seen = True

            sys.stdout.write(render_dashboard(m, instance_path))
            sys.stdout.flush()

            if once:
                break

            time.sleep(refresh_s)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default="dimensional_core/state")
    ap.add_argument("--refresh", type=float, default=0.5)
    args = ap.parse_args()
    run_monitor(state_dir=args.state_dir, refresh_s=args.refresh, once=False)


if __name__ == "__main__":
    main()