"""Unit tests for StateStore."""
from __future__ import annotations

import os
import json
import tempfile
import threading
import unittest
from pathlib import Path

from dimensional_core.core.state_store import StateStore


class TestStateStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="dc_store_test_")
        self.store = StateStore(
            instance_path=os.path.join(self.tmpdir, "ip.json"),
            events_path=os.path.join(self.tmpdir, "events.jsonl"),
            eid_path=os.path.join(self.tmpdir, "eid.txt"),
            event_flush_every=1,
            event_flush_interval_s=0.0,
        )

    def tearDown(self):
        self.store.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── EID ───────────────────────────────────────────────────────────────────

    def test_eid_starts_at_one(self):
        self.assertEqual(self.store._next_eid(), 1)

    def test_eid_monotonically_increases(self):
        eids = [self.store._next_eid() for _ in range(10)]
        self.assertEqual(eids, list(range(1, 11)))

    def test_eid_unique_across_threads(self):
        results = []
        lock = threading.Lock()

        def worker():
            for _ in range(50):
                eid = self.store._next_eid()
                with lock:
                    results.append(eid)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), len(set(results)), "Duplicate EIDs found")

    # ── append_event ──────────────────────────────────────────────────────────

    def test_append_event_assigns_eid(self):
        e1 = self.store.append_event({"type": "A"})
        e2 = self.store.append_event({"type": "B"})
        self.assertLess(e1["eid"], e2["eid"])

    def test_append_event_adds_timestamp(self):
        e = self.store.append_event({"type": "T"})
        self.assertIn("_ts", e)
        self.assertIsInstance(e["_ts"], float)

    def test_events_written_to_file(self):
        self.store.append_event({"type": "X"})
        self.store.flush()
        self.assertGreater(os.path.getsize(self.store.events_path), 0)

    # ── iter_events_after ─────────────────────────────────────────────────────

    def test_iter_events_after_filters_correctly(self):
        for _ in range(5):
            self.store.append_event({"type": "E"})
        self.store.flush()
        events = list(self.store.iter_events_after(3))
        self.assertTrue(all(e["eid"] > 3 for e in events))
        self.assertEqual(len(events), 2)

    def test_iter_events_after_missing_file_returns_empty(self):
        if os.path.exists(self.store.events_path):
            os.remove(self.store.events_path)
        self.assertEqual(list(self.store.iter_events_after(0)), [])

    # ── instance point ────────────────────────────────────────────────────────

    def test_load_instance_point_missing_returns_none(self):
        self.assertIsNone(self.store.load_instance_point())

    def test_save_and_load_instance_point(self):
        data = {"global_step": 42, "cycle": 3, "last_eid": 10, "instance": {"x": 1}}
        self.store.save_instance_point(data)
        loaded = self.store.load_instance_point()
        self.assertEqual(loaded["global_step"], 42)
        self.assertEqual(loaded["instance"], {"x": 1})

    def test_repeated_saves_leave_valid_json(self):
        for i in range(20):
            self.store.save_instance_point({"global_step": i})
        loaded = self.store.load_instance_point()
        self.assertIsNotNone(loaded)
        self.assertIn("global_step", loaded)

    # ── rotation ──────────────────────────────────────────────────────────────

    def test_rotate_events_creates_archive(self):
        for i in range(5):
            self.store.append_event({"type": f"E{i}"})
        self.store.flush()
        archive = self.store.rotate_events(keep_archives=5)
        self.assertIsNotNone(archive)
        self.assertTrue(os.path.exists(archive))

    def test_rotate_events_empties_main_log(self):
        for i in range(5):
            self.store.append_event({"type": f"E{i}"})
        self.store.flush()
        self.store.rotate_events(keep_archives=5)
        self.assertEqual(os.path.getsize(self.store.events_path), 0)

    def test_rotate_empty_file_returns_none(self):
        self.store.flush()
        self.assertIsNone(self.store.rotate_events(keep_archives=5))

    def test_cleanup_archives_keeps_limit(self):
        folder = os.path.dirname(self.store.events_path)
        for i in range(8):
            p = os.path.join(folder, f"events.archive.100000{i}.jsonl")
            with open(p, "w") as f:
                f.write("{}")
        self.store._cleanup_archives(keep_archives=3)
        remaining = [f for f in os.listdir(folder) if f.startswith("events.archive.")]
        self.assertEqual(len(remaining), 3)


if __name__ == "__main__":
    unittest.main()
