import json
import tempfile
import unittest
from pathlib import Path

from tools import k3ctl


class K3CtlTests(unittest.TestCase):
    def test_public_example_is_valid(self) -> None:
        path, config = k3ctl.resolve_config("config/project.example.json")
        self.assertEqual(k3ctl.validate(path, config), [])

    def test_public_example_declares_workflow_dependencies_and_skills(self) -> None:
        _, config = k3ctl.resolve_config("config/project.example.json")
        self.assertEqual(set(config["dependencies"]), {"kda", "humanize"})
        self.assertEqual(
            set(config["skills"]), {"project", "kernelwiki", "ncu_report", "humanize"}
        )

    def test_local_config_requires_full_source_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "project.local.json"
            source = json.loads(Path(k3ctl.ROOT / "config/project.example.json").read_text())
            source["source"]["commit"] = "<not-pinned>"
            path.write_text(json.dumps(source), encoding="utf-8")
            errors = k3ctl.validate(path, source)
            self.assertTrue(any("40-character" in error for error in errors))

    def test_duplicate_task_ids_are_rejected(self) -> None:
        path, config = k3ctl.resolve_config("config/project.example.json")
        duplicate = dict(config)
        duplicate["tasks"] = list(config["tasks"]) * 2
        errors = k3ctl.validate(path, duplicate)
        self.assertTrue(any("duplicate task id" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
