# dimensional_core/core/state_store.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional


class StateStore:
    def __init__(
        self,
        instance_path: str = "dimensional_core/state/instance_point.json",
        events_path: str = "dimensional_core/state/events.jsonl",
        eid_path: str = "dimensional_core/state/eid.txt",
    ):
        self.instance_path = instance_path
        self.events_path = events_path
        self.eid_path = eid_path

        os.makedirs(os.path.dirname(self.instance_path), exist_ok=True)

        if not os.path.exists(self.eid_path):
            with open(self.eid_path, "w", encoding="utf-8") as f:
                f.write("0")

        # Engine may store mutable instance here
        self.instance: Dict[str, Any] = {}

        # Optional stop condition used by demo
        self.max_steps: int | None = None

    # ---------- instance point ----------
    def load_instance_point(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.instance_path):
            return None
        with open(self.instance_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_instance_point(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.instance_path), exist_ok=True)
        tmp = self.instance_path + ".tmp"
        payload = dict(data)
        payload.setdefault("_ts", time.time())

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.instance_path)

    # ---------- eid ----------
    def _next_eid(self) -> int:
        with open(self.eid_path, "r+", encoding="utf-8") as f:
            last = int((f.read().strip() or "0"))
            eid = last + 1
            f.seek(0)
            f.truncate()
            f.write(str(eid))
            f.flush()
            os.fsync(f.fileno())
        return eid

    # ---------- events ----------
    def append_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        eid = self._next_eid()
        e = dict(event)
        e["eid"] = eid
        e["_ts"] = time.time()

        os.makedirs(os.path.dirname(self.events_path), exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(e) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return e

    def iter_events_after(self, last_eid: int) -> Iterable[Dict[str, Any]]:
        if not os.path.exists(self.events_path):
            return []
        def _gen():
            with open(self.events_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    if int(e.get("eid", 0)) > last_eid:
                        yield e
        return _gen()

    # ---------- rotation ----------
    def rotate_events(self, keep_archives: int = 10) -> Optional[str]:
        if not os.path.exists(self.events_path):
            return None
        if os.path.getsize(self.events_path) == 0:
            return None

        ts = int(time.time())
        archive = os.path.join(os.path.dirname(self.events_path), f"events.archive.{ts}.jsonl")
        os.replace(self.events_path, archive)

        with open(self.events_path, "w", encoding="utf-8") as f:
            f.write("")
            f.flush()
            os.fsync(f.fileno())

        self._cleanup_archives(keep_archives)
        return archive

    def _cleanup_archives(self, keep_archives: int) -> None:
        folder = os.path.dirname(self.events_path)
        files = os.listdir(folder)
        archives: List[str] = sorted(
            [f for f in files if f.startswith("events.archive.") and f.endswith(".jsonl")]
        )
        if len(archives) <= keep_archives:
            return
        for name in archives[: len(archives) - keep_archives]:
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass

    # ---------- C21 fast replay (uses replay_index.json) ----------
    def iter_events_after_fast(self, last_eid: int):
        """
        Faster replay by seeking into events.jsonl from replay_index.json if possible.
        Falls back gracefully.
        """
        from .replay_index import ReplayIndex

        idx = ReplayIndex()
        info = idx.load() or {}

        start_offset = 0
        if int(info.get("last_eid", -1)) == int(last_eid):
            start_offset = int(info.get("byte_offset", 0))

        if not os.path.exists(self.events_path):
            return []

        def _gen():
            last_seen_eid = int(last_eid)
            last_good_offset = int(start_offset)

            with open(self.events_path, "r", encoding="utf-8") as f:
                try:
                    f.seek(start_offset)
                except Exception:
                    f.seek(0)
                    last_good_offset = 0

                while True:
                    offset_here = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        e = json.loads(line)
                    except Exception:
                        continue

                    eid = int(e.get("eid", 0) or 0)
                    if eid > last_eid:
                        last_seen_eid = eid
                        last_good_offset = offset_here
                        yield e

            # Save replay cursor for next startup
            try:
                idx.save(last_seen_eid, last_good_offset)
            except Exception:
                pass

        return _gen()
