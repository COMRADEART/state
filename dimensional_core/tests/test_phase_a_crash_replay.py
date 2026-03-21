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


class PhaseACrashReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="dimcore_phase_a_"))
        self.state_dir = self.tmpdir / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _instance_path(self) -> Path:
        return self.state_dir / "instance_point.json"

    def _events_path(self) -> Path:
        return self.state_dir / "events.jsonl"

    def _load_json(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_crash_and_resume(self) -> None:
        # Start engine
        cmd = [
            PYTHON,
            "-m",
            "dimensional_core.run_demo",
            "--state-dir",
            str(self.state_dir),
            "--warps",
            "4",
            "--max-steps",
            "400",
            "--flush-every",
            "8",
            "--flush-interval-ms",
            "100",
            "--rotate-every-commits",
            "0",
            "--quiet",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Let it run long enough to create events/checkpoint
        time.sleep(2.0)

        # Simulate crash
        proc.kill()
        proc.wait(timeout=10)

        self.assertTrue(self._instance_path().exists(), "instance_point.json should exist after crash")
        self.assertTrue(self._events_path().exists(), "events.jsonl should exist after crash")

        ip1 = self._load_json(self._instance_path())

        step1 = int(ip1.get("global_step", 0) or 0)
        eid1 = int(ip1.get("last_eid", 0) or 0)

        self.assertGreater(step1, 0, "checkpoint should have recorded progress before crash")
        self.assertGreater(eid1, 0, "checkpoint should have recorded event progress before crash")

        # Resume
        cmd_resume = [
            PYTHON,
            "-m",
            "dimensional_core.run_demo",
            "--state-dir",
            str(self.state_dir),
            "--resume",
            "--warps",
            "4",
            "--max-steps",
            "200",
            "--flush-every",
            "8",
            "--flush-interval-ms",
            "100",
            "--rotate-every-commits",
            "0",
            "--quiet",
        ]

        proc2 = subprocess.run(
            cmd_resume,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )

        self.assertEqual(
            proc2.returncode,
            0,
            msg=f"resume process failed\nSTDOUT:\n{proc2.stdout}\nSTDERR:\n{proc2.stderr}",
        )

        ip2 = self._load_json(self._instance_path())

        step2 = int(ip2.get("global_step", 0) or 0)
        eid2 = int(ip2.get("last_eid", 0) or 0)

        self.assertGreater(step2, step1, "resume should continue past the crash checkpoint")
        self.assertGreater(eid2, eid1, "resume should emit more events after crash recovery")

        # Phase A seam checks
        self.assertIn("warps_state", ip2, "canonical resume key missing")
        self.assertIsInstance(ip2["warps_state"], dict, "warps_state should be a dict")

        self.assertIn("graphs_state", ip2, "compatibility alias missing")
        self.assertIsInstance(ip2["graphs_state"], dict, "graphs_state should be a dict")

        # Basic consistency
        self.assertIn("instance", ip2)
        self.assertIsInstance(ip2["instance"], dict)


if __name__ == "__main__":
    unittest.main()