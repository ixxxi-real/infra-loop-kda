from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.k3 import campaign, diagnosis, tuning, util


class CampaignTests(unittest.TestCase):
    def test_promotion_requires_verified(self) -> None:
        record = campaign.new("c", "task", "base", "workload", "gate")
        record = campaign.transition(record, campaign.RUNNING)
        with self.assertRaises(util.ToolError):
            campaign.transition(record, campaign.PROMOTED)
        record = campaign.add_candidate(
            record,
            {
                "candidate_id": "x",
                "search_lane": "structural",
                "diagnosis_ref": "diagnosis.json",
                "limiter_class": "register_pressure",
                "gate_version": "gate",
            },
        )
        record = campaign.transition(record, campaign.CANDIDATE_READY)
        record = campaign.transition(record, campaign.VERIFIED, actor="evaluator")
        record = campaign.transition(record, campaign.PROMOTED, actor="orchestrator")
        self.assertEqual(record["status"], campaign.PROMOTED)
        self.assertEqual(len(record["history"]), 5)

    def test_save_load_preserves_identity(self) -> None:
        record = campaign.new("c", "task", "base", "workload", "gate")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            campaign.save(path, record)
            self.assertEqual(campaign.load(path)["base_commit"], "base")


class DiagnosisTests(unittest.TestCase):
    def test_normalizes_limiter_and_rejects_unknown(self) -> None:
        result = diagnosis.normalize(
            {"kernel": "k", "phase": "prefill", "limiter": "register_pressure"},
            profile_hash="abc",
        )
        self.assertEqual(result["limiter_class"], "register_pressure")
        self.assertEqual(result["profile_hash"], "abc")
        with self.assertRaises(util.ToolError):
            diagnosis.normalize({"kernel": "k", "phase": "p", "limiter": "guess"})


class TuningTests(unittest.TestCase):
    def test_expansion_is_sorted_and_hashed(self) -> None:
        manifest = {
            "schema_version": 1,
            "kernel": "k",
            "gate_version": "g",
            "seed": 1,
            "search_space": {"warps": [2, 4], "stages": [2, 3]},
        }
        trials = tuning.expand(manifest)
        self.assertEqual([item["trial"] for item in trials], [0, 1, 2, 3])
        self.assertEqual(len({item["config_hash"] for item in trials}), 4)


if __name__ == "__main__":
    unittest.main()
