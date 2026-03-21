from __future__ import annotations

import atexit
import json
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

_READ_RETRY_COUNT = 8       # max retries for reading a file under contention
_READ_RETRY_SLEEP_S = 0.03  # base sleep for read retries (doubles each attempt)
_WRITE_RETRY_COUNT = 12     # max retries for atomic file writes
_WRITE_RETRY_SLEEP_S = 0.05 # base sleep for write retries (doubles each attempt)


class StateStore:
    def __init__(
        self,
        instance_path: str = "dimensional_core/state/instance_point.json",
        events_path: str = "dimensional_core/state/events.jsonl",
        eid_path: str = "dimensional_core/state/eid.txt",
        event_flush_every: int = 64,
        event_flush_interval_s: float = 0.50,
    ) -> None:
        self.instance_path = instance_path
        self.events_path = events_path
        self.eid_path = eid_path
        self._state_dir = os.path.dirname(instance_path)  # exposed for engine/warp_store

        self.event_flush_every = max(1, int(event_flush_every))
        self.event_flush_interval_s = max(0.0, float(event_flush_interval_s))

        os.makedirs(os.path.dirname(self.instance_path), exist_ok=True)

        self.instance: Dict[str, Any] = {}
        self.max_steps: int | None = None

        self._eid_lock = threading.Lock()
        self._events_lock = threading.Lock()
        self._events_fp = None
        self._events_since_flush = 0
        self._last_flush_ts = time.time()

        self._eid_value = self._load_eid_initial()

        atexit.register(self.close)

    # ------------------------------------------------------------------
    # instance point
    # ------------------------------------------------------------------

    def load_instance_point(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.instance_path):
            return None

        for attempt in range(_READ_RETRY_COUNT):
            try:
                with open(self.instance_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except PermissionError:
                time.sleep(_READ_RETRY_SLEEP_S * (2 ** attempt))
            except json.JSONDecodeError:
                time.sleep(_READ_RETRY_SLEEP_S * (2 ** attempt))
            except Exception:
                return None

        return None

    def save_instance_point(self, data: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.instance_path), exist_ok=True)

        payload = dict(data)
        payload.setdefault("_ts", time.time())

        last_err = None

        for attempt in range(_WRITE_RETRY_COUNT):
            tmp = f"{self.instance_path}.tmp.{os.getpid()}.{threading.get_ident()}"

            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(tmp, self.instance_path)
                return

            except PermissionError as e:
                last_err = e
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                time.sleep(_WRITE_RETRY_SLEEP_S * (2 ** attempt))

            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                raise

        raise last_err if last_err is not None else PermissionError(
            f"Could not replace instance point: {self.instance_path}"
        )

    # ------------------------------------------------------------------
    # eid
    # ------------------------------------------------------------------

    def _load_eid_initial(self) -> int:
        os.makedirs(os.path.dirname(self.eid_path), exist_ok=True)

        if not os.path.exists(self.eid_path):
            try:
                with open(self.eid_path, "w", encoding="utf-8") as f:
                    f.write("0")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                pass
            return 0

        for attempt in range(_READ_RETRY_COUNT):
            try:
                with open(self.eid_path, "r", encoding="utf-8") as f:
                    raw = f.read().strip() or "0"
                    return int(raw)
            except PermissionError:
                time.sleep(_READ_RETRY_SLEEP_S * (2 ** attempt))
            except Exception:
                return 0

        return 0

    def _persist_eid(self) -> None:
        last_err = None

        for attempt in range(_WRITE_RETRY_COUNT):
            tmp = f"{self.eid_path}.tmp.{os.getpid()}.{threading.get_ident()}"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(str(self._eid_value))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.eid_path)
                return
            except PermissionError as e:
                last_err = e
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                time.sleep(_WRITE_RETRY_SLEEP_S * (2 ** attempt))
            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                raise

        if last_err is not None:
            raise last_err

    def _next_eid(self) -> int:
        with self._eid_lock:
            self._eid_value += 1
            return self._eid_value

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------

    def _ensure_events_fp(self) -> None:
        if self._events_fp is not None and not self._events_fp.closed:
            return
        os.makedirs(os.path.dirname(self.events_path), exist_ok=True)
        self._events_fp = open(self.events_path, "a", encoding="utf-8")

    def append_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        eid = self._next_eid()
        e = dict(event)
        e["eid"] = eid
        e["_ts"] = time.time()

        line = json.dumps(e, separators=(",", ":")) + "\n"

        with self._events_lock:
            self._ensure_events_fp()
            self._events_fp.write(line)
            self._events_since_flush += 1

            now = time.time()
            if (
                self._events_since_flush >= self.event_flush_every
                or (now - self._last_flush_ts) >= self.event_flush_interval_s
            ):
                self._flush_events_locked()

        return e

    def flush_events(self) -> None:
        with self._events_lock:
            self._flush_events_locked()

    def flush(self) -> None:
        self.flush_events()

    def _flush_events_locked(self) -> None:
        if self._events_fp is not None and not self._events_fp.closed:
            self._events_fp.flush()
            os.fsync(self._events_fp.fileno())

        with self._eid_lock:
            self._persist_eid()

        self._events_since_flush = 0
        self._last_flush_ts = time.time()

    def close(self) -> None:
        with self._events_lock:
            try:
                self._flush_events_locked()
            except Exception:
                pass
            try:
                if self._events_fp is not None and not self._events_fp.closed:
                    self._events_fp.close()
            except Exception:
                pass
            self._events_fp = None

    def iter_events_after(self, last_eid: int) -> Iterable[Dict[str, Any]]:
        if not os.path.exists(self.events_path):
            return []

        def _gen():
            with open(self.events_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    if int(e.get("eid", 0) or 0) > last_eid:
                        yield e

        return _gen()

    # ------------------------------------------------------------------
    # rotation
    # ------------------------------------------------------------------

    def rotate_events(self, keep_archives: int = 10) -> Optional[str]:
        with self._events_lock:
            self._flush_events_locked()

            if not os.path.exists(self.events_path):
                return None
            if os.path.getsize(self.events_path) == 0:
                return None

            archive = os.path.join(
                os.path.dirname(self.events_path),
                f"events.archive.{int(time.time())}.jsonl",
            )

            last_err = None

            for attempt in range(_WRITE_RETRY_COUNT):
                try:
                    if self._events_fp is not None and not self._events_fp.closed:
                        self._events_fp.close()
                        self._events_fp = None

                    os.replace(self.events_path, archive)

                    with open(self.events_path, "w", encoding="utf-8") as f:
                        f.write("")
                        f.flush()
                        os.fsync(f.fileno())

                    self._cleanup_archives(keep_archives)
                    self._ensure_events_fp()
                    self._events_since_flush = 0
                    self._last_flush_ts = time.time()
                    return archive

                except PermissionError as e:
                    last_err = e
                    time.sleep(_WRITE_RETRY_SLEEP_S * (2 ** attempt))
                except Exception:
                    try:
                        self._ensure_events_fp()
                    except Exception:
                        pass
                    raise

            try:
                self._ensure_events_fp()
            except Exception:
                pass

            raise last_err if last_err is not None else PermissionError(
                f"Could not rotate events file: {self.events_path}"
            )

    def _cleanup_archives(self, keep_archives: int) -> None:
        folder = os.path.dirname(self.events_path)
        files = os.listdir(folder)
        archives: List[str] = sorted(
            f for f in files if f.startswith("events.archive.") and f.endswith(".jsonl")
        )
        if len(archives) <= keep_archives:
            return
        for name in archives[: len(archives) - keep_archives]:
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass

    # ------------------------------------------------------------------
    # fast replay
    # ------------------------------------------------------------------

    def iter_events_after_fast(self, last_eid: int):
        from .replay_index import ReplayIndex

        idx = ReplayIndex()
        info = idx.load() or {}

        start_offset = 0
        if int(info.get("last_eid", -1)) == int(last_eid):
            start_offset = int(info.get("byte_offset", 0) or 0)

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

            try:
                idx.save(last_seen_eid, last_good_offset)
            except Exception:
                pass

        return _gen()