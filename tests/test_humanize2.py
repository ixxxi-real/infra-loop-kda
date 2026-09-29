import unittest
from pathlib import Path

from tools.k3 import humanize2
from tools import k3ctl


class Humanize2ConfigTests(unittest.TestCase):
    def test_public_config_declares_a_separate_runtime_and_flow(self) -> None:
        path, value = k3ctl.resolve_config("config/project.example.json")
        self.assertEqual(k3ctl.validate(path, value), [])
        block = value["workflow"]["humanize2"]
        self.assertEqual(block["runtime_dir"], "external/humanize2")
        self.assertEqual(block["flow"], "flows/infra_loop_kda_flame_chase")
        self.assertEqual(value["dependencies"]["humanize2"]["repository"], "https://github.com/humanfia/humanize.git")
        self.assertEqual(
            value["dependencies"]["flowverse"]["commit"],
            "59546b7c572f06a7c71304952a4898fde5c2dc7e",
        )
        self.assertEqual(block["first_chaser"], "codex/gpt-6-astra:ultra")
        self.assertEqual(block["second_chaser"], "claude/claude-opus-5:max")
        self.assertEqual(block["cleaner"], "codex/gpt-6-astra:ultra")
        self.assertEqual(block["work_paths"], ["python", ".kda-task"])

    def test_plan_is_read_only_and_keeps_old_adapter_separate(self) -> None:
        path, value = k3ctl.resolve_config("config/project.example.json")
        result = humanize2.plan(Path(k3ctl.ROOT), path, value, "prefill-kda")
        self.assertEqual(result["schema"], "k3ctl/humanize2-plan/1")
        self.assertFalse(result["would_spawn"])
        self.assertTrue(result["workflow"]["flow"].endswith("/flows/infra_loop_kda_flame_chase"))
        self.assertEqual(
            result["invocation"]["env"]["CODEX_HOME"],
            str(Path.home() / ".codex-bak"),
        )
        self.assertIn(
            "second_chaser=claude/claude-opus-5:max",
            result["invocation"]["argv"],
        )
        self.assertEqual(result["workflow"]["work_paths"], ["python", ".kda-task"])
        self.assertEqual(result["workflow"]["reviewer_session_policy"], "fresh-per-turn")
        self.assertEqual(result["workflow"]["review_mode"], "advisory-only")
        self.assertEqual(result["workflow"]["protected_evidence_paths"], [".kda-task"])
        self.assertEqual(
            result["workflow"]["final_review"],
            {"required": True, "backend": "claude"},
        )
        self.assertTrue((Path(k3ctl.ROOT) / "projects/kimi-k3/prefill/runtime/humanize2/plan.json").is_file())


if __name__ == "__main__":
    unittest.main()
