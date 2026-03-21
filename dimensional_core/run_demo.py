# dimensional_core/run_demo.py
"""
CLI entry point for the Dimensional Core 3D execution engine.

Usage
-----
    python -m dimensional_core.run_demo [options]

Options
-------
  --resume               Load the latest checkpoint before running
  --warps N              Number of parallel warps (default: 2)
  --monitor              Launch the C24 real-time terminal dashboard
  --max-steps N          Stop after N global steps (default: 1000)
  --quiet                Suppress per-event log lines
  --lr FLOAT             Dimension Y gradient-descent learning rate (default: 0.10)
  --y-opcode STR         Dimension Y VISA opcode: VSTEP_Y | VMUTATE_Y (default: VSTEP_Y)
  --flush-every N        Flush event buffer every N events (default: 16)
  --flush-interval-ms T  Flush event buffer every T ms (default: 250)
  --rotate-every-commits K  Rotate events log after K commits (0=disable)
  --keep-archives K      Number of rotated archives to keep (default: 10)
  --state-dir DIR        State directory for checkpoints and event log
"""
from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from dimensional_core.core.engine import Engine
from dimensional_core.core.state_store import StateStore
from dimensional_core.core.triggers import TriggerConfig

logger = logging.getLogger(__name__)


# ── SimpleWriter ────────────────────────────────────────────────────────────────

class SimpleWriter:
    """
    Thin adapter between the engine's _emit() calls and StateStore.

    The ``event()`` method stores each event in the append-only JSONL log.
    With ``verbose=True`` each event is also emitted to the logger at DEBUG
    level so operators can follow execution in real time without the C24
    monitor.
    """

    def __init__(self, store: StateStore, verbose: bool = True) -> None:
        self.store   = store
        self.verbose = verbose

    def event(self, name: str, payload: dict | None = None) -> dict:
        payload = dict(payload or {})
        rec = {"type": name, "payload": payload}
        saved = self.store.append_event(rec)

        if self.verbose:
            eid = saved.get("eid", "?")
            logger.debug("[EVENT %s] %s %s", eid, name, payload)

        return saved

    def submit_instance_point(self, data: dict) -> None:
        self.store.save_instance_point(data)
        if self.verbose:
            logger.debug("[INSTANCE_POINT] saved step=%s", data.get("global_step"))


# ── Argument parser ─────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m dimensional_core.run_demo",
        description="Dimensional Core — 3D execution engine (X/Y/Z dimensions)",
    )

    # Execution model
    ap.add_argument("--resume", action="store_true",
                    help="Load latest checkpoint before running")
    ap.add_argument("--warps", type=int, default=2,
                    help="Number of parallel warps (default: 2)")
    ap.add_argument("--max-steps", type=int, default=1000,
                    help="Stop after N global steps (default: 1000)")

    # Dimension Y configuration
    ap.add_argument("--lr", type=float, default=0.10,
                    help="Dimension Y learning rate (default: 0.10)")
    ap.add_argument("--y-opcode", type=str, default="VSTEP_Y",
                    choices=["VSTEP_Y", "VMUTATE_Y"],
                    help="Dimension Y VISA opcode (default: VSTEP_Y)")

    # UI
    ap.add_argument("--monitor", action="store_true",
                    help="Launch C24 real-time terminal dashboard")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-event debug output")

    # Flush policy
    ap.add_argument("--flush-every", type=int, default=16,
                    help="Flush event buffer every N events")
    ap.add_argument("--flush-interval-ms", type=int, default=250,
                    help="Flush event buffer every T milliseconds")
    ap.add_argument("--rotate-every-commits", type=int, default=50,
                    help="Rotate events log after K commits (0=disable)")
    ap.add_argument("--keep-archives", type=int, default=10,
                    help="Number of rotated event archives to keep")

    # Paths
    ap.add_argument("--state-dir", type=str, default="dimensional_core/state",
                    help="Directory for instance point, event log, and EID state")

    return ap


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap   = build_arg_parser()
    args = ap.parse_args()

    # Configure logging before anything else
    log_level = logging.WARNING if args.quiet else logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── State directory ────────────────────────────────────────────────────────
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    # ── StateStore ────────────────────────────────────────────────────────────
    store = StateStore(
        instance_path=str(state_dir / "instance_point.json"),
        events_path=str(state_dir / "events.jsonl"),
        eid_path=str(state_dir / "eid.txt"),
        event_flush_every=max(1, args.flush_every),
        event_flush_interval_s=max(0.0, args.flush_interval_ms / 1000.0),
    )

    # ── Resume: extend max_steps from resumed checkpoint ──────────────────────
    loaded_ip = None
    if args.resume:
        loaded_ip = store.load_instance_point()

    if args.resume and loaded_ip:
        resumed_step = int(loaded_ip.get("global_step", 0) or 0)
        store.max_steps = resumed_step + max(1, args.max_steps)
    else:
        store.max_steps = max(1, args.max_steps)

    # ── TriggerConfig ─────────────────────────────────────────────────────────
    triggers = TriggerConfig.normalized(
        commit_every_n_events=max(1, args.flush_every),
        rotate_every_n_commits=max(0, args.rotate_every_commits),
        keep_archives=max(1, args.keep_archives),
    )

    # ── Writer ────────────────────────────────────────────────────────────────
    verbose = not args.quiet and not args.monitor
    writer  = SimpleWriter(store, verbose=verbose)

    # ── Startup log ───────────────────────────────────────────────────────────
    logger.info(
        "dimensional_core starting  resume=%s warps=%s max_steps=%s "
        "lr=%s y_opcode=%s state_dir=%s",
        args.resume, args.warps, store.max_steps,
        args.lr, args.y_opcode, state_dir,
    )

    if args.resume:
        if loaded_ip:
            logger.info(
                "loaded checkpoint  global_step=%s cycle=%s last_eid=%s reason=%s",
                loaded_ip.get("global_step"), loaded_ip.get("cycle"),
                loaded_ip.get("last_eid"),    loaded_ip.get("reason"),
            )
        else:
            logger.info("no checkpoint found — starting fresh")

    # ── Optional C24 monitor thread ────────────────────────────────────────────
    if args.monitor:
        from dimensional_core.c24_monitor import run_monitor
        monitor_thread = threading.Thread(
            target=run_monitor,
            kwargs={"state_dir": str(state_dir), "refresh_s": 0.5, "once": False},
            daemon=True,
            name="c24-monitor",
        )
        monitor_thread.start()

    # ── Engine ────────────────────────────────────────────────────────────────
    eng = Engine(
        store=store,
        writer=writer,
        resume=args.resume,
        triggers=triggers,
        max_workers=max(4, args.warps),
        max_in_flight=max(8, args.warps * 2),
        target_warps=max(1, args.warps),
        min_warps=max(1, min(args.warps, 2)),
        watchdog_timeout_s=30.0,
        watchdog_poll_s=1.0,
        lr=args.lr,
        y_opcode=args.y_opcode,
    )

    try:
        eng.run_forever()
    finally:
        logger.info("engine stopped")
        ip = store.load_instance_point()
        if ip:
            logger.info(
                "final checkpoint  global_step=%s cycle=%s last_eid=%s reason=%s",
                ip.get("global_step"), ip.get("cycle"),
                ip.get("last_eid"),    ip.get("reason"),
            )
        try:
            store.flush()
        except Exception:
            pass
        store.close()


if __name__ == "__main__":
    main()
