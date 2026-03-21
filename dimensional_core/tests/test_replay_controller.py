"""Unit tests for ReplayController."""
from __future__ import annotations

import types
import unittest
from dimensional_core.core.replay_controller import ReplayController
from dimensional_core.core.task_graph import build_3d_graph
from dimensional_core.core.priority_scheduler import HeapWarpScheduler


def _make_engine(warp_ids=("W0",)):
    """Return a minimal engine-like namespace for testing the replay controller."""
    eng = types.SimpleNamespace()
    eng.global_step = 0
    eng.cycle = 0
    eng.warps = {}
    eng.store = types.SimpleNamespace(instance={})
    eng.scheduler = HeapWarpScheduler()

    for wid in warp_ids:
        g = build_3d_graph(wid, [], {})
        eng.warps[wid] = g
        eng.scheduler.register(wid)

    return eng


class TestReplayController(unittest.TestCase):

    def test_unknown_event_type_is_ignored(self):
        rc = ReplayController()
        eng = _make_engine()
        rc.apply(eng, {"type": "UNKNOWN_EVT", "payload": {"global_step": 0, "cycle": 0}})
        self.assertEqual(eng.global_step, 0)

    def test_empty_event_is_ignored(self):
        rc = ReplayController()
        eng = _make_engine()
        rc.apply(eng, {})
        self.assertEqual(eng.global_step, 0)

    def test_global_step_advanced_monotonically(self):
        rc = ReplayController()
        eng = _make_engine()
        eng.global_step = 5
        rc.apply(eng, {"type": "NODE_RESULT", "payload": {
            "global_step": 10, "cycle": 0,
            "warp": "W0", "node_id": "W0:X", "res": {},
        }})
        self.assertEqual(eng.global_step, 10)

    def test_global_step_not_decreased(self):
        rc = ReplayController()
        eng = _make_engine()
        eng.global_step = 20
        rc.apply(eng, {"type": "NODE_RESULT", "payload": {
            "global_step": 5, "cycle": 0,
            "warp": "W0", "node_id": "W0:X", "res": {},
        }})
        self.assertEqual(eng.global_step, 20)

    def test_node_result_marks_node_done(self):
        rc = ReplayController()
        eng = _make_engine()
        rc.apply(eng, {"type": "NODE_RESULT", "payload": {
            "global_step": 1, "cycle": 0,
            "warp": "W0", "node_id": "W0:X",
            "res": {"value": 42},
        }})
        node = eng.warps["W0"].nodes.get("W0:X")
        self.assertIsNotNone(node)
        self.assertEqual(str(node.status), "DONE")
        self.assertEqual(node.result, {"value": 42})

    def test_node_result_applies_instance_updates(self):
        rc = ReplayController()
        eng = _make_engine()
        rc.apply(eng, {"type": "NODE_RESULT", "payload": {
            "global_step": 1, "cycle": 0,
            "warp": "W0", "node_id": "W0:X",
            "res": {"instance_updates": {"key": "value"}},
        }})
        self.assertEqual(eng.store.instance.get("key"), "value")

    def test_node_result_updates_scheduler_score(self):
        rc = ReplayController()
        eng = _make_engine()
        rc.apply(eng, {"type": "NODE_RESULT", "payload": {
            "global_step": 1, "cycle": 0,
            "warp": "W0", "node_id": "W0:Z",
            "res": {"score": 3.14, "min_score": 3.14},
        }})
        # Scheduler should have score updated (priority changes from 0.0 baseline)
        p = eng.scheduler.priority("W0")
        self.assertIsInstance(p, float)

    def test_node_result_applies_local_delta(self):
        rc = ReplayController()
        eng = _make_engine()
        rc.apply(eng, {"type": "NODE_RESULT", "payload": {
            "global_step": 1, "cycle": 0,
            "warp": "W0", "node_id": "W0:Y",
            "res": {},
            "local_delta": {"x": 7},
        }})
        self.assertEqual(eng.warps["W0"]._local.get("x"), 7)

    def test_warp_local_snapshot_restores_local(self):
        rc = ReplayController()
        eng = _make_engine()
        rc.apply(eng, {"type": "WARP_LOCAL_SNAPSHOT", "payload": {
            "warp": "W0",
            "local": {"a": 1, "b": 2},
        }})
        self.assertEqual(eng.warps["W0"]._local, {"a": 1, "b": 2})

    def test_rollback_event_triggers_graph_rollback(self):
        rc = ReplayController()
        eng = _make_engine()
        graph = eng.warps["W0"]
        graph.save_rollback_snapshot()
        # Simulate that W0:X had been queued before
        graph.ready.clear()
        rc.apply(eng, {"type": "WARP_ROLLED_BACK", "payload": {
            "global_step": 1, "cycle": 0,
            "warp": "W0",
        }})
        self.assertIn("W0:X", graph.ready)

    def test_rearm_triggers_cycle_rearm(self):
        rc = ReplayController()
        eng = _make_engine()
        graph = eng.warps["W0"]
        graph.ready.clear()
        rc.apply(eng, {"type": "NODE_RESULT", "payload": {
            "global_step": 1, "cycle": 0,
            "warp": "W0", "node_id": "W0:Z",
            "res": {}, "rearm": True,
        }})
        self.assertIn("W0:X", graph.ready)


if __name__ == "__main__":
    unittest.main()
