"""Unit tests for HeapWarpScheduler."""
from __future__ import annotations

import unittest
from dimensional_core.core.priority_scheduler import HeapWarpScheduler


class TestHeapWarpScheduler(unittest.TestCase):

    def test_choose_returns_registered_warp(self):
        s = HeapWarpScheduler()
        s.register("W0")
        s.register("W1")
        self.assertIn(s.choose(), ("W0", "W1"))

    def test_choose_empty_returns_none(self):
        s = HeapWarpScheduler()
        self.assertIsNone(s.choose())

    def test_unregistered_warp_not_chosen(self):
        s = HeapWarpScheduler()
        s.register("W0")
        s.unregister("W0")
        self.assertIsNone(s.choose())

    def test_lower_score_wins(self):
        s = HeapWarpScheduler(age_bonus=0.0)
        s.register("W0")
        s.register("W1")
        s.update_score("W0", 10.0)
        s.update_score("W1", 1.0)
        self.assertEqual(s.choose(), "W1")

    def test_aging_prevents_starvation(self):
        s = HeapWarpScheduler(age_bonus=100.0)
        s.register("W0")
        s.register("W1")
        s.update_score("W0", 1000.0)
        s.update_score("W1", 1000.0)
        choices = [s.choose() for _ in range(20)]
        self.assertIn("W1", choices)

    def test_priority_returns_float(self):
        s = HeapWarpScheduler()
        s.register("W0")
        self.assertIsInstance(s.priority("W0"), float)

    def test_update_score_changes_priority(self):
        s = HeapWarpScheduler(age_bonus=0.0)
        s.register("W0")
        p_before = s.priority("W0")
        s.update_score("W0", 99.0)
        p_after = s.priority("W0")
        self.assertNotEqual(p_before, p_after)

    def test_compaction_preserves_behavior(self):
        s = HeapWarpScheduler()
        for i in range(5):
            s.register(f"W{i}")
        for _ in range(100):
            for i in range(5):
                s.update_score(f"W{i}", float(i))
        self.assertIsNotNone(s.choose())

    def test_register_twice_is_idempotent(self):
        s = HeapWarpScheduler()
        s.register("W0")
        s.register("W0")
        choices = [s.choose() for _ in range(3)]
        self.assertTrue(all(c == "W0" for c in choices))


if __name__ == "__main__":
    unittest.main()
