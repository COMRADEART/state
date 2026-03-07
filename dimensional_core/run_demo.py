from __future__ import annotations

import argparse
import threading
from pathlib import Path

from dimensional_core.core.engine import Engine
from dimensional_core.core.state_store import StateStore


class SimpleWriter:
    def __init__(self, store: StateStore, verbose: bool = True) -> None:
        self.store = store
        self.verbose = verbose

    def event(self, name: str, payload: dict | None = None) -> dict:
        payload = dict(payload or {})
        rec = {
            "type": name,
            "payload": payload,
        }
        saved = self.store.append_event(rec)

        if self.verbose:
            eid = saved.get("eid", "?")
            print(f"[EVENT {eid}] {name} {payload}")

        return saved

    def submit_instance_point(self, data: dict) -> None:
        self.store.save_instance_point(data)
        if self.verbose:
            print("[INSTANCE_POINT] saved")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--warps", type=int, default=2)
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    state_dir = Path("dimensional_core/state")
    state_dir.mkdir(parents=True, exist_ok=True)

    store = StateStore(
        instance_path=str(state_dir / "instance_point.json"),
        events_path=str(state_dir / "events.jsonl"),
        eid_path=str(state_dir / "eid.txt"),
        event_flush_every=16,
        event_flush_interval_s=0.25,
    )

    loaded_ip = None
    if args.resume:
        loaded_ip = store.load_instance_point()

    if args.resume and loaded_ip:
        loaded_step = int(loaded_ip.get("global_step", 0) or 0)
        store.max_steps = loaded_step + max(1, args.max_steps)
    else:
        store.max_steps = max(1, args.max_steps)

    # If monitor is on, suppress event spam so the dashboard is visible.
    writer = SimpleWriter(store, verbose=(not args.quiet and not args.monitor))

    print(f"[RUN_DEMO] resume={args.resume}")
    print(f"[RUN_DEMO] warps={args.warps}")
    print(f"[RUN_DEMO] monitor={args.monitor}")
    print(f"[RUN_DEMO] max_steps={store.max_steps}")

    if args.resume:
        if loaded_ip:
            print(
                "[RUN_DEMO] loaded checkpoint:",
                {
                    "global_step": loaded_ip.get("global_step"),
                    "cycle": loaded_ip.get("cycle"),
                    "last_eid": loaded_ip.get("last_eid"),
                    "reason": loaded_ip.get("reason"),
                },
            )
        else:
            print("[RUN_DEMO] no checkpoint found, starting fresh")
    else:
        print("[RUN_DEMO] fresh start")

    monitor_thread = None
    if args.monitor:
        from dimensional_core.c24_monitor import run_monitor

        monitor_thread = threading.Thread(
            target=run_monitor,
            kwargs={"state_dir": str(state_dir), "refresh_s": 0.5, "once": False},
            daemon=True,
            name="c24-monitor",
        )
        monitor_thread.start()

    eng = Engine(
        store=store,
        writer=writer,
        resume=args.resume,
        max_workers=max(4, args.warps),
        max_in_flight=max(8, args.warps * 2),
        target_warps=max(1, args.warps),
        min_warps=max(1, min(args.warps, 2)),
        watchdog_timeout_s=30.0,
        watchdog_poll_s=1.0,
    )

    try:
        eng.run_forever()
    finally:
        print("[RUN_DEMO] engine stopped")
        ip = store.load_instance_point()
        if ip:
            print(
                "[RUN_DEMO] final checkpoint:",
                {
                    "global_step": ip.get("global_step"),
                    "cycle": ip.get("cycle"),
                    "last_eid": ip.get("last_eid"),
                    "reason": ip.get("reason"),
                },
            )
        try:
            store.flush()
        except Exception:
            pass
        store.close()


if __name__ == "__main__":
    main()