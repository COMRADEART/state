# dimensional_core/c22_stability_suite.py
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List


STATE_DIR = os.path.join("dimensional_core", "state")
EVENTS_PATH = os.path.join(STATE_DIR, "events.jsonl")
INSTANCE_POINT_PATH = os.path.join(STATE_DIR, "instance_point.json")


@dataclass
class SuiteConfig:
    python_exe: str = sys.executable
    module_to_run: str = "dimensional_core.run_demo"
    warmup_seconds: float = 1.0
    run_seconds: float = 2.5
    restart_seconds: float = 1.5
    crash_run_seconds: float = 2.5


def _read_instance_point() -> Optional[Dict[str, Any]]:
    if not os.path.exists(INSTANCE_POINT_PATH):
        return None
    with open(INSTANCE_POINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_last_n_events(n: int = 2000) -> List[Dict[str, Any]]:
    if not os.path.exists(EVENTS_PATH):
        return []
    # Efficient tail read for jsonl (simple approach)
    with open(EVENTS_PATH, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        block = 65536
        data = b""
        while size > 0 and data.count(b"\n") < n + 5:
            step = min(block, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
        lines = data.splitlines()[-n:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line.decode("utf-8")))
        except Exception:
            continue
    return out


def _last_eid_and_global_step_from_events() -> Tuple[int, int]:
    events = _read_last_n_events(3000)
    last_eid = 0
    last_gs = 0
    for e in events:
        eid = int(e.get("eid", 0) or 0)
        gs = e.get("global_step")
        if eid > last_eid:
            last_eid = eid
        if isinstance(gs, int) and gs > last_gs:
            last_gs = gs
    return last_eid, last_gs


def _count_recent(events: List[Dict[str, Any]], name: str) -> int:
    return sum(1 for e in events if e.get("event") == name)


def _compute_metrics(window_seconds: float = 3.0) -> Dict[str, Any]:
    """
    Uses timestamps in events.jsonl (_ts) to compute rates.
    """
    events = _read_last_n_events(6000)
    now = time.time()
    win = [e for e in events if isinstance(e.get("_ts"), (int, float)) and (now - float(e["_ts"])) <= window_seconds]
    total = len(win)
    node_results = _count_recent(win, "NODE_RESULT")
    warp_batches = _count_recent(win, "WARP_BATCH_SUBMITTED")
    warp_scores = _count_recent(win, "WARP_SCORE")

    # latency estimate: if node_result has no explicit duration, approximate with inter-arrival
    times = sorted(float(e["_ts"]) for e in win if isinstance(e.get("_ts"), (int, float)))
    avg_inter = None
    if len(times) >= 2:
        diffs = [times[i] - times[i - 1] for i in range(1, len(times))]
        avg_inter = sum(diffs) / len(diffs)

    return {
        "window_seconds": window_seconds,
        "events_per_sec": (total / window_seconds) if window_seconds > 0 else None,
        "node_results_per_sec": (node_results / window_seconds) if window_seconds > 0 else None,
        "warp_batches_per_sec": (warp_batches / window_seconds) if window_seconds > 0 else None,
        "warp_scores_per_sec": (warp_scores / window_seconds) if window_seconds > 0 else None,
        "avg_event_interarrival_sec": avg_inter,
        "sample_size": total,
    }


def _start_engine(cfg: SuiteConfig) -> subprocess.Popen:
    # IMPORTANT: run as module so it uses your package imports
    cmd = [cfg.python_exe, "-m", cfg.module_to_run]
    # keep output visible for debugging, but don’t block on it
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p


def _stop_engine_graceful(p: subprocess.Popen, timeout: float = 5.0) -> None:
    if p.poll() is not None:
        return
    # Send CTRL+C equivalent where possible
    try:
        if os.name == "nt":
            # Windows: best effort terminate (CTRL+C requires special console handling)
            p.terminate()
        else:
            p.send_signal(signal.SIGINT)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass

    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            return
        time.sleep(0.05)

    try:
        p.kill()
    except Exception:
        pass


def _kill_engine_hard(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    try:
        p.kill()
    except Exception:
        pass


def _drain_output(p: subprocess.Popen, max_lines: int = 60) -> str:
    """
    Get some output for debugging without hanging.
    """
    if p.stdout is None:
        return ""
    lines = []
    try:
        for _ in range(max_lines):
            line = p.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    except Exception:
        pass
    return "\n".join(lines)


def _assert(cond: bool, msg: str) -> Tuple[bool, str]:
    return (True, "PASS: " + msg) if cond else (False, "FAIL: " + msg)


def test_replay_does_not_reset(cfg: SuiteConfig) -> Tuple[bool, List[str]]:
    """
    Run -> stop -> run, ensure global_step and eid keep increasing.
    """
    logs: List[str] = []
    eid0, gs0 = _last_eid_and_global_step_from_events()
    logs.append(f"baseline: last_eid={eid0}, last_global_step={gs0}")

    p = _start_engine(cfg)
    time.sleep(cfg.warmup_seconds)
    time.sleep(cfg.run_seconds)
    _stop_engine_graceful(p)
    out1 = _drain_output(p)
    if out1:
        logs.append("[engine output 1]\n" + out1)

    eid1, gs1 = _last_eid_and_global_step_from_events()
    logs.append(f"after run1: last_eid={eid1}, last_global_step={gs1}")

    ok1, m1 = _assert(eid1 > eid0, "events appended during run1")
    ok2, m2 = _assert(gs1 >= gs0, "global_step did not go backwards on run1")
    logs += [m1, m2]
    if not (ok1 and ok2):
        return False, logs

    # restart
    p2 = _start_engine(cfg)
    time.sleep(cfg.restart_seconds)
    _stop_engine_graceful(p2)
    out2 = _drain_output(p2)
    if out2:
        logs.append("[engine output 2]\n" + out2)

    eid2, gs2 = _last_eid_and_global_step_from_events()
    logs.append(f"after run2: last_eid={eid2}, last_global_step={gs2}")

    ok3, m3 = _assert(eid2 > eid1, "events appended during run2 (replay continues)")
    ok4, m4 = _assert(gs2 >= gs1, "global_step did not go backwards on run2")
    logs += [m3, m4]
    return (ok3 and ok4), logs


def test_crash_recovery(cfg: SuiteConfig) -> Tuple[bool, List[str]]:
    """
    Hard kill the process, then restart and ensure it continues.
    """
    logs: List[str] = []
    eid0, gs0 = _last_eid_and_global_step_from_events()
    logs.append(f"baseline: last_eid={eid0}, last_global_step={gs0}")

    p = _start_engine(cfg)
    time.sleep(cfg.warmup_seconds)
    time.sleep(cfg.crash_run_seconds)

    _kill_engine_hard(p)
    time.sleep(0.3)  # allow OS to finalize

    eid1, gs1 = _last_eid_and_global_step_from_events()
    logs.append(f"after crash: last_eid={eid1}, last_global_step={gs1}")

    ok1, m1 = _assert(eid1 >= eid0, "events exist after crash run")
    logs.append(m1)

    # restart after crash
    p2 = _start_engine(cfg)
    time.sleep(cfg.restart_seconds)
    _stop_engine_graceful(p2)

    eid2, gs2 = _last_eid_and_global_step_from_events()
    logs.append(f"after restart: last_eid={eid2}, last_global_step={gs2}")

    ok2, m2 = _assert(eid2 > eid1, "events appended after restart following crash")
    ok3, m3 = _assert(gs2 >= gs1, "global_step did not go backwards after crash recovery")
    logs += [m2, m3]
    return (ok1 and ok2 and ok3), logs


def test_metrics(cfg: SuiteConfig) -> Tuple[bool, List[str]]:
    """
    Robust metrics test (Windows-friendly):
    - Start engine
    - Wait up to N seconds until we observe at least one NODE_RESULT in events.jsonl
    - Then compute rates over a larger window
    """
    logs: List[str] = []

    p = _start_engine(cfg)
    time.sleep(cfg.warmup_seconds)

    # Wait until NODE_RESULT appears (or timeout)
    timeout = 10.0
    t0 = time.time()
    saw_node_result = False

    while time.time() - t0 < timeout:
        evs = _read_last_n_events(1500)
        if _count_recent(evs, "NODE_RESULT") > 0:
            saw_node_result = True
            break
        time.sleep(0.25)

    metrics = _compute_metrics(window_seconds=8.0)
    logs.append("metrics: " + json.dumps(metrics, indent=2))

    _stop_engine_graceful(p)

    ok0, m0 = _assert(saw_node_result, f"observed at least one NODE_RESULT within {timeout}s")
    ok1, m1 = _assert((metrics.get("node_results_per_sec") or 0) > 0.0, "node_results_per_sec > 0")
    ok2, m2 = _assert((metrics.get("events_per_sec") or 0) > 0.0, "events_per_sec > 0")
    logs += [m0, m1, m2]

    return (ok0 and ok1 and ok2), logs



def main() -> None:
    cfg = SuiteConfig()

    print("\n=== C22 Stability Suite ===\n")
    print(f"Python: {cfg.python_exe}")
    print(f"Module: {cfg.module_to_run}\n")

    # IMPORTANT: Your demo must not auto-stop immediately (max_steps)
    ip = _read_instance_point()
    if ip:
        gs = ip.get("global_step")
        mx = ip.get("max_steps")
        print(f"[info] instance_point global_step={gs} (max_steps in file? {mx})")
    else:
        print("[info] no instance_point.json found yet (ok).")

    results: List[Tuple[str, bool, List[str]]] = []

    ok, logs = test_replay_does_not_reset(cfg)
    results.append(("Replay does not reset", ok, logs))

    ok, logs = test_crash_recovery(cfg)
    results.append(("Crash recovery", ok, logs))

    ok, logs = test_metrics(cfg)
    results.append(("Metrics activity", ok, logs))

    print("\n--- RESULTS ---")
    all_ok = True
    for name, ok, logs in results:
        print(f"{'PASS' if ok else 'FAIL'} - {name}")
        if not ok:
            all_ok = False
        # print compact logs
        print("\n".join("  " + line for line in logs[-12:]))
        print()

    if all_ok:
        print("✅ C22: STABLE BASELINE CONFIRMED")
        sys.exit(0)
    else:
        print("❌ C22: FAILURES DETECTED (see logs above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
