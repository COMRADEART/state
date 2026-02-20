# dimensional_core/run_demo.py
from __future__ import annotations

import sys
import traceback
import argparse
import os
import json
import time
import threading
import hashlib
from typing import Any, Dict


class SimpleWriter:
    def __init__(self, store):
        self.store = store

    def event(self, name: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
        payload = payload or {}
        print(f"[EVENT] {name:<22} | {payload}")

        rec = {"event": name, **payload}
        if hasattr(self.store, "append_event"):
            return self.store.append_event(rec)
        return None

    def submit_instance_point(self, data: Dict[str, Any]) -> None:
        if hasattr(self.store, "save_instance_point"):
            self.store.save_instance_point(data)


def _capture_warp_state(engine, wid: str) -> Dict[str, Any]:
    """
    C23 Capture: minimal transferable warp state.
    """
    g = engine.warps.get(wid)
    if g is None:
        return {"warp": wid, "missing": True}

    local = dict(getattr(g, "_local", {}) or {})
    nodes_state: Dict[str, Any] = {}
    if hasattr(g, "nodes") and isinstance(g.nodes, dict):
        for nid, n in g.nodes.items():
            nodes_state[str(nid)] = {"status": getattr(n, "status", "PENDING")}

    return {
        "warp": wid,
        "global_step": int(engine.global_step),
        "cycle": int(engine.cycle),
        "local": local,
        "nodes": nodes_state,
        "_ts": time.time(),
    }


def _encode_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    C23 Encode: canonical json -> sha256 digest.
    """
    raw = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return {"digest": digest, "snapshot": snapshot}


def _persist_dimension_snapshot(encoded: Dict[str, Any], target_dimension: str) -> str:
    """
    C23 Transfer: persist into dimensional_core/state/dimensions/<target>/
    """
    root = os.path.join("dimensional_core", "state", "dimensions", str(target_dimension))
    os.makedirs(root, exist_ok=True)

    snap = encoded["snapshot"]
    wid = snap.get("warp", "W?")
    ts = int(float(snap.get("_ts", time.time())))
    path = os.path.join(root, f"{ts}.{wid}.{encoded['digest'][:12]}.json")

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(encoded, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def _c23_transfer_loop(engine, writer: SimpleWriter, every_sec: float, target_dimension: str, stop_evt: threading.Event) -> None:
    """
    Background demo loop:
      capture -> encode -> persist -> emit DIMENSION_TRANSFER
    """
    while not stop_evt.is_set():
        time.sleep(max(0.05, float(every_sec)))

        # Read engine state safely
        try:
            with engine.lock:
                warp_ids = list(engine.warps.keys())

            for wid in warp_ids:
                with engine.lock:
                    snap = _capture_warp_state(engine, wid)
                encoded = _encode_snapshot(snap)
                path = _persist_dimension_snapshot(encoded, target_dimension)

                writer.event(
                    "DIMENSION_TRANSFER",
                    {
                        "warp": wid,
                        "target_dimension": target_dimension,
                        "digest": encoded["digest"],
                        "snapshot_path": path,
                        "global_step": int(engine.global_step),
                        "cycle": int(engine.cycle),
                    },
                )
        except Exception as e:
            writer.event("DIMENSION_TRANSFER_ERROR", {"error": str(e)})


def main() -> None:
    from dimensional_core.core.engine import Engine
    from dimensional_core.core.state_store import StateStore
    from dimensional_core.core.triggers import TriggerConfig

    ap = argparse.ArgumentParser(description="Dimensional Core demo runner")
    ap.add_argument("--resume", action="store_true", help="Enable C20/C22 resume+replay behavior")
    ap.add_argument("--c23-demo", action="store_true", help="Enable C23 dimensional transfer demo loop")
    ap.add_argument("--c23-every", type=float, default=2.0, help="C23 transfer interval (seconds)")
    ap.add_argument("--c23-target", type=str, default="storage", help="Target dimension name (folder)")
    ap.add_argument("--max-steps", type=int, default=0, help="Optional auto-stop after N steps (0=off)")
    args = ap.parse_args()

    print("\n=== Dimensional Core :: Demo Runner (C20/C22 + C23 optional) ===\n")
    print(f"resume={args.resume} | c23_demo={args.c23_demo} | c23_every={args.c23_every}s | c23_target={args.c23_target}\n")

    store = StateStore()
    writer = SimpleWriter(store)

    # optional auto-stop
    store.max_steps = (args.max_steps if args.max_steps and args.max_steps > 0 else None)

    engine = Engine(
        store=store,
        writer=writer,
        resume=bool(args.resume),
        base_graphs=8,
        lanes=4,
        max_workers=4,
        max_in_flight=8,
        triggers=TriggerConfig(
            commit_every_n_events=25,
            commit_on_cycle_done=True,
            rotate_every_n_commits=5,
            keep_archives=10,
        ),
    )

    stop_evt = threading.Event()
    t = None

    if args.c23_demo:
        t = threading.Thread(
            target=_c23_transfer_loop,
            args=(engine, writer, float(args.c23_every), str(args.c23_target), stop_evt),
            daemon=True,
        )
        t.start()
        print("[C23] Dimensional transfer loop started.\n")
        print("Snapshots will be written under:")
        print(f"  dimensional_core/state/dimensions/{args.c23_target}/\n")

    print("Engine started. Press CTRL+C to stop.\n")
    try:
        engine.run_forever()
    finally:
        stop_evt.set()
        if t is not None:
            t.join(timeout=1.0)
        print("\nEngine finished cleanly.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CTRL+C] Interrupted by user. Shutting down.\n")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
