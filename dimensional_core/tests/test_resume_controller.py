"""Unit tests for ResumeController."""
from __future__ import annotations

import unittest
from dimensional_core.core.resume_controller import ResumeController


class TestResumeController(unittest.TestCase):

    def test_none_returns_defaults(self):
        rc = ResumeController()
        result = rc.resolve(None)
        self.assertEqual(result["global_step"], 0)
        self.assertEqual(result["cycle"], 0)
        self.assertEqual(result["last_eid"], 0)
        self.assertEqual(result["warps_state"], {})
        self.assertEqual(result["instance"], {})

    def test_empty_dict_returns_defaults(self):
        rc = ResumeController()
        result = rc.resolve({})
        self.assertEqual(result["global_step"], 0)
        self.assertEqual(result["cycle"], 0)

    def test_canonical_warps_state_used(self):
        rc = ResumeController()
        ip = {
            "global_step": 5,
            "cycle": 3,
            "last_eid": 10,
            "instance": {"x": 1},
            "warps_state": {"W0": {"local": {"val": 42}}},
        }
        result = rc.resolve(ip)
        self.assertEqual(result["global_step"], 5)
        self.assertEqual(result["cycle"], 3)
        self.assertEqual(result["last_eid"], 10)
        self.assertEqual(result["warps_state"], {"W0": {"local": {"val": 42}}})

    def test_legacy_graphs_state_fallback(self):
        rc = ResumeController()
        ip = {
            "global_step": 2,
            "cycle": 1,
            "last_eid": 4,
            "instance": {},
            "graphs_state": {"W0": {"local": {"x": 1}}},
        }
        result = rc.resolve(ip)
        self.assertEqual(result["warps_state"], {"W0": {"local": {"x": 1}}})

    def test_warps_state_preferred_over_graphs_state(self):
        rc = ResumeController()
        ip = {
            "global_step": 1,
            "cycle": 0,
            "last_eid": 1,
            "instance": {},
            "warps_state": {"W0": {"local": {"canonical": True}}},
            "graphs_state": {"W0": {"local": {"canonical": False}}},
        }
        result = rc.resolve(ip)
        self.assertTrue(result["warps_state"]["W0"]["local"]["canonical"])

    def test_instance_none_defaults_to_empty_dict(self):
        rc = ResumeController()
        result = rc.resolve({"global_step": 0, "cycle": 0, "last_eid": 0, "instance": None})
        self.assertEqual(result["instance"], {})

    def test_graphs_state_alias_matches_warps_state(self):
        rc = ResumeController()
        ip = {"global_step": 1, "cycle": 0, "last_eid": 1, "instance": {}, "warps_state": {"W0": {}}}
        result = rc.resolve(ip)
        self.assertEqual(result["warps_state"], result["graphs_state"])


if __name__ == "__main__":
    unittest.main()
