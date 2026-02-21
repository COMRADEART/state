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
import glob
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


# ---------------------------------------------------------------------
# C23 Snapshot Capture / Encode / Persist (Export)
# ---------------------------------------------------------------------

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
    wid = str(snap.get("warp", "W?"))
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
                        "target_dimension": str(target_dimension),
                        "digest": encoded["digest"],
                        "snapshot_path": path,
                        "global_step": int(engine.global_step),
                        "cycle": int(engine.cycle),
                    },
                )
        except Exception as e:
            writer.event("DIMENSION_TRANSFER_ERROR", {"error": str(e)})


# ---------------------------------------------------------------------
# C23 Restore + Branch (Dimensional Jump)
# ---------------------------------------------------------------------

def _latest_snapshot_path(target_dimension: str, warp_id: str) -> str | None:
    root = os.path.join("dimensional_core", "state", "dimensions", str(target_dimension))
    if not os.path.isdir(root):
        return None
    pat = os.path.join(root, f"*.{warp_id}.*.json")  # <ts>.<warp>.<digestprefix>.json
    files = sorted(glob.glob(pat), key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0] if files else None


def _load_encoded_snapshot(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _inject_branch_from_snapshot(engine, writer: SimpleWriter, encoded: Dict[str, Any], branch_suffix: str = "B") -> str:
    """
    Create a new warp (branch) from a snapshot:
      - new warp id: <warp><suffix>
      - apply local + node statuses
      - register to scheduler
    """
    from dimensional_core.core.warp_factory import build_warp_graph

    # Our demo snapshots store {digest, snapshot}; older variants might use payload
    snap = encoded.get("snapshot") or encoded.get("payload") or {}
    src_warp = str(snap.get("warp", "W0"))
    new_warp = f"{src_warp}{branch_suffix}"

    # Build a fresh warp graph. We don't require lane_gids for the prototype demo.
    lane_gids: list[str] = []
    g = build_warp_graph(new_warp, lane_gids, engine.store.instance, lanes=engine.lanes)

    if not hasattr(g, "_local") or not isinstance(getattr(g, "_local"), dict):
        g._local = {}

    # Apply local state
    local = snap.get("local")
    if isinstance(local, dict):
        g._local = dict(local)

    # Apply node statuses
    nodes = snap.get("nodes")
    if isinstance(nodes, dict) and hasattr(g, "nodes") and isinstance(g.nodes, dict):
        for nid, st in nodes.items():
            if nid in g.nodes:
                n = g.nodes[nid]
                if isinstance(st, str):
                    n.status = st
                elif isinstance(st, dict):
                    n.status = st.get("status", getattr(n, "status", "PENDING"))

        # Rebuild ReadySet from node statuses (dependency-free pending nodes)
        if hasattr(g, "ready") and hasattr(g, "deps"):
            for nid, n in g.nodes.items():
                if getattr(n, "status", "PENDING") == "PENDING" and not g.deps.get(nid):
                    g.ready.add(nid)

    # Inject into engine
    with engine.lock:
        engine.warps[new_warp] = g
        engine.scheduler.register(new_warp)

    writer.event(
        "DIMENSION_RESTORE",
        {
            "source_warp": src_warp,
            "new_warp": new_warp,
            "digest": encoded.get("digest"),
        },
    )
    return new_warp


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    from dimensional_core.core.engine import Engine
    from dimensional_core.core.state_store import StateStore
    from dimensional_core.core.triggers import TriggerConfig

    ap = argparse.ArgumentParser(description="Dimensional Core demo runner")
    ap.add_argument("--resume", action="store_true", help="Enable C20/C22 resume+replay behavior")
    ap.add_argument("--c23-demo", action="store_true", help="Enable C23 dimensional transfer demo loop")
    ap.add_argument("--c23-every", type=float, default=2.0, help="C23 transfer interval (seconds)")
    ap.add_argument("--c23-target", type=str, default="storage", help="Target dimension name (folder)")
    ap.add_argument("--c23-restore-latest", type=str, default="", help="Restore latest snapshot for warp (ex: W0)")
    ap.add_argument("--c23-branch-suffix", type=str, default="B", help="Suffix for branch warp id (default B)")
    ap.add_argument("--max-steps", type=int, default=0, help="Optional auto-stop after N steps (0=off)")
    args = ap.parse_args()

    print("\n=== Dimensional Core :: Demo Runner (C20/C22 + C23 optional) ===\n")
    print(
        f"resume={bool(args.resume)} | c23_demo={bool(args.c23_demo)} | "
        f"c23_every={float(args.c23_every)}s | c23_target={str(args.c23_target)} | "
        f"restore_latest={str(args.c23_restore_latest) or 'None'}\n"
    )

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

    # Optional: restore + branch from latest snapshot before starting execution
    if args.c23_restore_latest:
        wid = str(args.c23_restore_latest)
        p = _latest_snapshot_path(str(args.c23_target), wid)
        if not p:
            writer.event(
                "DIMENSION_RESTORE_ERROR",
                {"error": f"No snapshot found for {wid} in dimension={args.c23_target}"},
            )
        else:
            try:
                enc = _load_encoded_snapshot(p)
                new_warp = _inject_branch_from_snapshot(
                    engine,
                    writer,
                    enc,
                    branch_suffix=str(args.c23_branch_suffix),
                )
                writer.event("DIMENSION_RESTORE_OK", {"snapshot_path": p, "new_warp": new_warp})
            except Exception as e:
                writer.event("DIMENSION_RESTORE_ERROR", {"error": str(e), "snapshot_path": p})

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
