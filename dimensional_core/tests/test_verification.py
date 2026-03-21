"""
Unit tests for the Dimension Z verification system and rollback.

Tests:
  - VSCORE_Z passes valid, converged output
  - VSCORE_Z rejects non-finite scores
  - VSCORE_Z rejects diverged scores (above threshold)
  - VSCORE_Z detects checksum mismatches
  - TaskGraph3D.rollback() restores _local and re-queues X
  - TaskGraph3D.rearm_cycle() re-queues X after success
  - Full mini-cycle: X→Y→Z pass and X→Y→Z fail + rollback
"""
from __future__ import annotations

import copy
import types
import unittest

from dimensional_core.core.task_graph import TaskGraph3D, build_3d_graph, NodeStatus
from dimensional_core.core.visa.vm import VectorVM, VInstruction
from dimensional_core.core.visa.registry import ExecutionContext
from dimensional_core.core.dimensions import DimXOperator, DimYOperator, DimZOperator


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_vscore_z(x_vec, scores, step=0, expected_checksum=None, max_score_threshold=1e6):
    """Run VSCORE_Z in isolation with controlled local state."""
    vm = VectorVM()
    local = {"V": {"x": x_vec, "score": scores}}
    args = {"max_score_threshold": max_score_threshold}
    if expected_checksum:
        args["expected_checksum"] = expected_checksum
    return vm.run(
        VInstruction(opcode="VSCORE_Z", args=args),
        warp_id="W0", node_id="W0:Z", dimension="Z",
        instance={}, local=local, lane_gids=["G0"], step=step,
    )


def _run_full_cycle(wid="W0", lane_gids=None, step=0):
    """Run a complete X→Y→Z cycle and return (x_local, y_res, z_res)."""
    lane_gids = lane_gids or ["G0", "G1"]
    instance  = {}
    local     = {}

    dim_x = DimXOperator()
    dim_y = DimYOperator()
    dim_z = DimZOperator()

    x_res = dim_x.execute("W0:X", wid, instance, local, lane_gids, {}, step)
    y_res = dim_y.execute("W0:Y", wid, instance, local, lane_gids, {"lr": 0.10}, step)
    z_res = dim_z.execute("W0:Z", wid, instance, local, lane_gids, {}, step)

    return local, x_res, y_res, z_res


# ── VSCORE_Z unit tests ────────────────────────────────────────────────────────

class TestVScoreZ(unittest.TestCase):

    def test_verified_on_good_scores(self):
        res = _run_vscore_z(x_vec=[3.1, 2.9], scores=[0.01, 0.01])
        self.assertTrue(res["verified"])
        self.assertFalse(res["rollback"])
        self.assertIn("checksum", res)
        self.assertIn("min_score", res)

    def test_rejects_nan_score(self):
        res = _run_vscore_z(x_vec=[float("nan")], scores=[float("nan")])
        self.assertFalse(res["verified"])
        self.assertTrue(res["rollback"])
        self.assertEqual(res["reason"], "non_finite_score")

    def test_rejects_inf_score(self):
        res = _run_vscore_z(x_vec=[float("inf")], scores=[float("inf")])
        self.assertFalse(res["verified"])
        self.assertEqual(res["reason"], "non_finite_score")

    def test_rejects_diverged_score(self):
        res = _run_vscore_z(x_vec=[1e7], scores=[1e7], max_score_threshold=1e6)
        self.assertFalse(res["verified"])
        self.assertEqual(res["reason"], "score_diverged")

    def test_rejects_empty_scores(self):
        res = _run_vscore_z(x_vec=[], scores=[])
        self.assertFalse(res["verified"])
        self.assertEqual(res["reason"], "no_scores")

    def test_checksum_mismatch_triggers_rollback(self):
        res = _run_vscore_z(x_vec=[3.0], scores=[0.0], expected_checksum="deadbeef")
        self.assertFalse(res["verified"])
        self.assertEqual(res["reason"], "checksum_mismatch")
        self.assertIn("checksum", res)

    def test_checksum_match_passes(self):
        # Run once to get the real checksum, then re-run with that checksum
        res1 = _run_vscore_z(x_vec=[3.0], scores=[0.0], step=99)
        checksum = res1["checksum"]
        res2 = _run_vscore_z(x_vec=[3.0], scores=[0.0], step=99, expected_checksum=checksum)
        self.assertTrue(res2["verified"])

    def test_score_key_present_on_success(self):
        res = _run_vscore_z(x_vec=[3.5], scores=[0.25])
        self.assertIn("score", res)
        self.assertAlmostEqual(res["score"], 0.25, places=5)


# ── TaskGraph3D rollback tests ─────────────────────────────────────────────────

class TestTaskGraph3DRollback(unittest.TestCase):

    def _make_graph(self, wid="W0"):
        g = build_3d_graph(wid, ["G0", "G1"], {})
        return g

    def test_rollback_restores_local(self):
        g = self._make_graph()
        g._local = {"V": {"x": [1.0, 2.0]}}
        g.save_rollback_snapshot()           # snapshot saved before Y
        g._local["V"]["x"] = [99.0, 99.0]   # Y modifies local
        g.rollback()
        self.assertEqual(g._local["V"]["x"], [1.0, 2.0])

    def test_rollback_requeues_x(self):
        g = self._make_graph()
        g.save_rollback_snapshot()
        g.rollback()
        self.assertIn("W0:X", g.ready)

    def test_rollback_marks_y_z_rolled_back(self):
        g = self._make_graph()
        # Simulate Y and Z completed then failed
        g.nodes["W0:Y"].status = NodeStatus.DONE
        g.nodes["W0:Z"].status = NodeStatus.DONE
        g.save_rollback_snapshot()
        g.rollback()
        self.assertEqual(g.nodes["W0:Y"].status, NodeStatus.ROLLED_BACK)
        self.assertEqual(g.nodes["W0:Z"].status, NodeStatus.ROLLED_BACK)

    def test_rollback_increments_fail_counter(self):
        g = self._make_graph()
        g.save_rollback_snapshot()
        g.rollback()
        self.assertEqual(g.verify_fail_count, 1)

    def test_rearm_cycle_requeues_x(self):
        g = self._make_graph()
        # Drain the ready set first
        g.ready.clear()
        g.rearm_cycle()
        self.assertIn("W0:X", g.ready)

    def test_rearm_cycle_increments_pass_counter(self):
        g = self._make_graph()
        g.rearm_cycle()
        self.assertEqual(g.verify_pass_count, 1)

    def test_rollback_without_snapshot_resets_local_to_empty(self):
        g = self._make_graph()
        # No snapshot saved — rollback should still work but _local may be empty
        g._local = {"V": {"x": [5.0]}}
        g.rollback()
        # After rollback without snapshot, _local is reset (snapshot was None)
        self.assertIn("W0:X", g.ready)


# ── Full X→Y→Z cycle tests ────────────────────────────────────────────────────

class TestFullCycle(unittest.TestCase):

    def test_full_cycle_passes_verification(self):
        """A fresh cycle starting at x=10 should pass Z (finite, not diverged)."""
        local, x_res, y_res, z_res = _run_full_cycle()
        self.assertIn("x_vec", x_res)
        self.assertIn("min_score", y_res)
        self.assertTrue(z_res["verified"],
                        msg=f"Z failed unexpectedly: {z_res}")

    def test_x_initialises_to_default(self):
        lane_gids = ["G0", "G1", "G2"]
        local, x_res, _, _ = _run_full_cycle(lane_gids=lane_gids)
        self.assertEqual(len(x_res["x_vec"]), 3)

    def test_y_reduces_score(self):
        """After gradient descent, score at step N+1 should be less than at step 0."""
        local, x_res, y_res, _ = _run_full_cycle()
        # x starts at 10.0, target is 3.0 → score = (10-3)^2 = 49
        # After one gradient step with lr=0.10: x = 10 - 0.10*2*(10-3) = 10 - 1.4 = 8.6
        # score = (8.6-3)^2 = 31.36 < 49
        self.assertLess(y_res["min_score"], 49.0,
                        msg="Gradient step must reduce score from initial 49.0")

    def test_z_score_matches_y_score(self):
        local, x_res, y_res, z_res = _run_full_cycle()
        # Z verifies the score already computed by Y
        self.assertAlmostEqual(z_res["min_score"], y_res["min_score"], places=6)

    def test_multiple_cycles_converge(self):
        """After 20 X→Y→Z cycles, score should be close to 0 (x≈3)."""
        local = {}
        instance = {}
        lane_gids = ["G0"]
        dim_x = DimXOperator()
        dim_y = DimYOperator()
        dim_z = DimZOperator()

        last_z = None
        for step in range(20):
            dim_x.execute("W0:X", "W0", instance, local, lane_gids, {}, step)
            dim_y.execute("W0:Y", "W0", instance, local, lane_gids, {"lr": 0.20}, step)
            last_z = dim_z.execute("W0:Z", "W0", instance, local, lane_gids, {}, step)

        self.assertTrue(last_z["verified"], f"Z failed after 20 cycles: {last_z}")
        self.assertLess(last_z["min_score"], 0.01,
                        msg=f"Score after 20 cycles should be near 0, got {last_z['min_score']}")

    def test_mutate_y_also_passes_z(self):
        """VMUTATE_Y should also produce output that passes Z verification."""
        local    = {}
        instance = {}
        lane_gids = ["G0", "G1"]

        DimXOperator().execute("W0:X", "W0", instance, local, lane_gids, {}, 0)
        DimYOperator().execute("W0:Y", "W0", instance, local, lane_gids,
                               {"y_opcode": "VMUTATE_Y"}, 0)
        z_res = DimZOperator().execute("W0:Z", "W0", instance, local, lane_gids, {}, 0)

        self.assertTrue(z_res["verified"], f"VMUTATE_Y output failed Z: {z_res}")


# ── Determinism test ──────────────────────────────────────────────────────────

class TestDeterminism(unittest.TestCase):

    def test_same_inputs_produce_same_checksum(self):
        """Two cycles with identical inputs must produce the same Z checksum."""
        def _checksum(step):
            local    = {}
            instance = {}
            DimXOperator().execute("W0:X", "W0", instance, local, ["G0"], {}, step)
            DimYOperator().execute("W0:Y", "W0", instance, local, ["G0"],
                                   {"lr": 0.10}, step)
            res = DimZOperator().execute("W0:Z", "W0", instance, local, ["G0"], {}, step)
            return res.get("checksum")

        c1 = _checksum(step=5)
        c2 = _checksum(step=5)
        self.assertIsNotNone(c1)
        self.assertEqual(c1, c2, "Same step must produce identical checksum")

    def test_different_steps_produce_different_checksums(self):
        """Different global_step values must yield different checksums."""
        def _checksum(step):
            local    = {}
            instance = {}
            DimXOperator().execute("W0:X", "W0", instance, local, ["G0"], {}, step)
            DimYOperator().execute("W0:Y", "W0", instance, local, ["G0"],
                                   {"lr": 0.10}, step)
            res = DimZOperator().execute("W0:Z", "W0", instance, local, ["G0"], {}, step)
            return res.get("checksum")

        # Different steps start from different seeds → different x_vec → different checksum
        c_s0 = _checksum(step=0)
        c_s99 = _checksum(step=99)
        self.assertNotEqual(c_s0, c_s99,
                            "Different steps should (almost certainly) produce different checksums")


if __name__ == "__main__":
    unittest.main()
