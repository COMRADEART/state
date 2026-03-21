"""
Integration test: crash → resume → verify correctness.

Simulates the full crash-recovery cycle:
  1. Start engine, let it run 2 s
  2. Kill the process (simulated crash)
  3. Assert checkpoint was written and contains progress
  4. Resume from checkpoint and run to completion
  5. Assert monotonic step + EID increase
  6. Assert checkpoint structure (warps_state, graphs_state alias, instance)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

PYTHON = sys.executable


class CrashReplayTest(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir   = Path(tempfile.mkdtemp(prefix="dc_crash_replay_"))
        self.state_dir = self.tmpdir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_json(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _instance_path(self) -> Path:
        return self.state_dir / "instance_point.json"

    def _events_path(self) -> Path:
        return self.state_dir / "events.jsonl"

    def _start_engine(self, extra: list | None = None) -> subprocess.Popen:
        cmd = [
            PYTHON, "-m", "dimensional_core.run_demo",
            "--state-dir",  str(self.state_dir),
            "--warps",      "2",
            "--max-steps",  "2000",
            "--flush-every","4",
            "--flush-interval-ms", "50",
            "--rotate-every-commits", "0",
            "--quiet",
        ] + (extra or [])
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _run_engine(self, extra: list | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
        cmd = [
            PYTHON, "-m", "dimensional_core.run_demo",
            "--state-dir",  str(self.state_dir),
            "--warps",      "2",
            "--max-steps",  "200",
            "--flush-every","4",
            "--flush-interval-ms", "50",
            "--rotate-every-commits", "0",
            "--quiet",
            "--resume",
        ] + (extra or [])
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_crash_and_resume(self) -> None:
        # ── Phase 1: run then crash ────────────────────────────────────────────
        proc = self._start_engine()
        time.sleep(2.0)
        proc.kill()
        proc.wait(timeout=10)

        ip_path = self._instance_path()
        ev_path = self._events_path()

        self.assertTrue(ip_path.exists(), "instance_point.json must exist after crash")
        self.assertTrue(ev_path.exists(), "events.jsonl must exist after crash")

        ip1 = self._load_json(ip_path)
        step1 = int(ip1.get("global_step", 0) or 0)
        eid1  = int(ip1.get("last_eid",    0) or 0)

        self.assertGreater(step1, 0, "engine must have made progress before crash")
        self.assertGreater(eid1,  0, "at least one event must be logged before crash")

        # ── Phase 2: resume ────────────────────────────────────────────────────
        result = self._run_engine(timeout=60)
        self.assertEqual(
            result.returncode, 0,
            msg=f"resume failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

        ip2 = self._load_json(ip_path)
        step2 = int(ip2.get("global_step", 0) or 0)
        eid2  = int(ip2.get("last_eid",    0) or 0)

        self.assertGreater(step2, step1, "step must increase after resume")
        self.assertGreater(eid2,  eid1,  "EID must increase after resume")

        # ── Phase 3: structure checks ──────────────────────────────────────────
        self.assertIn("warps_state",  ip2)
        self.assertIn("graphs_state", ip2, "legacy alias must be present")
        self.assertIsInstance(ip2["warps_state"],  dict)
        self.assertIsInstance(ip2["graphs_state"], dict)
        self.assertIn("instance",     ip2)
        self.assertIsInstance(ip2["instance"], dict)

        # Warp state entries must have local + nodes
        for wid, warp_st in ip2["warps_state"].items():
            self.assertIn("local", warp_st, f"warp {wid} missing local")
            self.assertIn("nodes", warp_st, f"warp {wid} missing nodes")

    def test_events_are_valid_jsonl(self) -> None:
        """After a clean run, every line in events.jsonl must be valid JSON."""
        proc = self._start_engine(extra=["--max-steps", "50"])
        proc.wait(timeout=30)

        ev_path = self._events_path()
        if not ev_path.exists():
            self.skipTest("events.jsonl not written (engine may have been too fast)")

        with open(ev_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.fail(f"Invalid JSON on line {lineno}: {exc}\n  {line[:120]}")
                self.assertIn("type",    obj, f"line {lineno}: missing 'type'")
                self.assertIn("payload", obj, f"line {lineno}: missing 'payload'")
                self.assertIn("eid",     obj, f"line {lineno}: missing 'eid'")

    def test_eids_are_monotonic(self) -> None:
        """EID values in events.jsonl must be strictly increasing."""
        proc = self._start_engine(extra=["--max-steps", "80"])
        proc.wait(timeout=30)

        ev_path = self._events_path()
        if not ev_path.exists():
            self.skipTest("no events.jsonl")

        prev_eid = -1
        with open(ev_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                eid = int(obj.get("eid", 0) or 0)
                self.assertGreater(eid, prev_eid, f"EID not monotonic: {eid} after {prev_eid}")
                prev_eid = eid

    def test_resume_without_checkpoint(self) -> None:
        """--resume with no checkpoint must still start cleanly from zero."""
        result = self._run_engine(timeout=30)
        self.assertEqual(result.returncode, 0,
                         msg=f"Fresh resume failed:\n{result.stderr}")

        ip_path = self._instance_path()
        self.assertTrue(ip_path.exists(), "checkpoint must be written even on fresh start")
        ip = self._load_json(ip_path)
        self.assertGreaterEqual(ip.get("global_step", 0), 0)


if __name__ == "__main__":
    unittest.main()
