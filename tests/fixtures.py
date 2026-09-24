"""Shared test fixtures.

Everything here is synthetic and lives under a temporary directory:

* no real GPU, SSH, model server or API is contacted;
* no network access;
* the mock ``claude``/``codex`` executables are shell scripts that record their
  argv and print canned output;
* Git commits are created **only** inside throwaway fixture repositories, never
  in this repository or in any external checkout.

A fixture is explicitly a fixture. Nothing produced here is evidence of a real
run, and the tests assert that the tooling says so.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

REAL_ROOT = Path(__file__).resolve().parents[1]

#: Identity used for fixture-only commits, so no global Git config is read
#: or modified.
GIT_IDENTITY = (
    "-c",
    "user.name=k3ctl fixture",
    "-c",
    "user.email=fixture@example.invalid",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "init.defaultBranch=main",
)


def git(args: List[str], cwd: Path, check: bool = True) -> str:
    """Run Git in a fixture repository with a hermetic environment."""
    env = dict(os.environ)
    env.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(cwd),
            "GIT_PAGER": "cat",
        }
    )
    completed = subprocess.run(
        ["git", *GIT_IDENTITY, *args],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            "fixture git %s failed: %s"
            % (" ".join(args), completed.stderr.decode("utf-8", "replace"))
        )
    return completed.stdout.decode("utf-8", "replace")


SOURCE_KERNEL = '''"""Fixture kernel module."""


def chunk_kda(q, k, v):
    """Baseline implementation used by the fixture harness."""
    return q, k, v
'''


def make_source_repo(base: Path, name: str = "source") -> Dict[str, Any]:
    """Create a throwaway source repository with two real commits.

    Returns the path plus both commit SHAs, so a test can distinguish a task's
    exact base from a different pin.
    """
    repo = base / name
    (repo / "python" / "kernels").mkdir(parents=True)
    git(["init", "--quiet", "."], cwd=repo)
    (repo / "python" / "kernels" / "kda.py").write_text(SOURCE_KERNEL, encoding="utf-8")
    (repo / "README.md").write_text("# fixture source\n", encoding="utf-8")
    git(["add", "-A"], cwd=repo)
    git(["commit", "--quiet", "-m", "base commit"], cwd=repo)
    base_commit = git(["rev-parse", "HEAD"], cwd=repo).strip()

    (repo / "README.md").write_text("# fixture source, moved on\n", encoding="utf-8")
    git(["add", "-A"], cwd=repo)
    git(["commit", "--quiet", "-m", "later commit"], cwd=repo)
    later_commit = git(["rev-parse", "HEAD"], cwd=repo).strip()

    return {
        "path": repo,
        "base_commit": base_commit,
        "later_commit": later_commit,
    }


PROJECT_GITIGNORE = """\
tasks/*/runtime/
external/.overlay/
evidence/exports/
.infra/
*.log
"""


def make_project(
    base: Path,
    source: Optional[Dict[str, Any]] = None,
    with_submodules: bool = True,
    name: str = "project",
) -> Dict[str, Any]:
    """Create a synthetic project root that satisfies root discovery.

    ``with_submodules=False`` reproduces a fresh checkout whose external
    dependencies were never initialized, which must be reported as missing
    rather than silently resolving to a parent repository.
    """
    root = base / name
    (root / "tools" / "k3").mkdir(parents=True)
    (root / "config").mkdir(parents=True)
    (root / "tasks").mkdir(parents=True)
    (root / "evidence" / "manifests").mkdir(parents=True)
    (root / "skills" / "kernel-optimization").mkdir(parents=True)

    # Root markers. The real implementation is imported from the real tree;
    # these only have to exist for discovery.
    (root / "tools" / "k3ctl.py").write_text("# fixture marker\n", encoding="utf-8")
    (root / ".gitignore").write_text(PROJECT_GITIGNORE, encoding="utf-8")
    (root / "skills" / "kernel-optimization" / "SKILL.md").write_text(
        "---\nname: kernel-optimization\nversion: 0.1.0\n---\n\nFixture skill.\n",
        encoding="utf-8",
    )

    # Root discovery requires every marker, including the public example config.
    (root / "config" / "project.example.json").write_text(
        json.dumps({"schema_version": 1, "project_id": "kimi-k3-kda"}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    if with_submodules:
        for relative, skill in (
            ("external/kda", "skills/KernelWiki/SKILL.md"),
            ("external/kda", "skills/ncu-report-skill/SKILL.md"),
            ("external/humanize", "skills/humanize/SKILL.md"),
        ):
            target = root / relative / skill
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "---\nname: %s\nversion: 1.0.0\n---\n\nFixture skill.\n"
                % Path(skill).parent.name,
                encoding="utf-8",
            )
        prompts = root / "external" / "kda" / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "basic-flow.md").write_text(
            "# Kernel Design Agents Basic Flow Prompt\n\nFixture prompt.\n",
            encoding="utf-8",
        )
        docs = root / "external" / "kda" / "docs"
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "agent-flow.md").write_text("# Agent Flow\n", encoding="utf-8")

        # A minimal but structurally complete Humanize plugin, so doctor sees a
        # real manifest, hook set and loop commands rather than reporting the
        # plugin missing. These are structural stand-ins; the genuinely pinned
        # plugin is exercised by tests/test_hook_chain.py.
        humanize = root / "external" / "humanize"
        (humanize / ".claude-plugin").mkdir(parents=True, exist_ok=True)
        (humanize / ".claude-plugin" / "plugin.json").write_text(
            json.dumps(
                {
                    "name": "humanize",
                    "version": "1.16.0",
                    "description": "Fixture stand-in for the pinned plugin.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (humanize / "hooks").mkdir(parents=True, exist_ok=True)
        (humanize / "hooks" / "hooks.json").write_text(
            json.dumps(
                {
                    "description": "Fixture hooks",
                    "hooks": {
                        "UserPromptSubmit": [],
                        "PreToolUse": [],
                        "PostToolUse": [],
                        "Stop": [],
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (humanize / "hooks" / "loop-codex-stop-hook.sh").write_text(
            "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
        )
        (humanize / "commands").mkdir(parents=True, exist_ok=True)
        for command in ("start-rlcr-loop", "cancel-rlcr-loop"):
            (humanize / "commands" / ("%s.md" % command)).write_text(
                "# %s\n\nFixture command.\n" % command, encoding="utf-8"
            )
        (humanize / "scripts").mkdir(parents=True, exist_ok=True)
        (humanize / "scripts" / "setup-rlcr-loop.sh").write_text(
            "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
        )

        # Each dependency is its own Git repository, as a real submodule is.
        for relative in ("external/kda", "external/humanize"):
            git(["init", "--quiet", "."], cwd=root / relative)
            git(["add", "-A"], cwd=root / relative)
            git(["commit", "--quiet", "-m", "fixture dependency"], cwd=root / relative)
    else:
        # Present but empty: the trap that must not resolve to the parent repo.
        (root / "external" / "kda").mkdir(parents=True, exist_ok=True)
        (root / "external" / "humanize").mkdir(parents=True, exist_ok=True)

    git(["init", "--quiet", "."], cwd=root)

    commits: Dict[str, str] = {}
    source_commit: Optional[str] = None
    if with_submodules:
        for relative in ("external/kda", "external/humanize"):
            commits[relative] = git(
                ["rev-parse", "HEAD"], cwd=root / relative
            ).strip()

        # external/sglang stands in for the pinned source runtime. When a source
        # repository is supplied it is cloned at the task base commit, so the
        # exact base is genuinely reachable; otherwise a standalone repo is
        # created so the dependency is present but unrelated.
        sglang = root / "external" / "sglang"
        if source is not None:
            git(
                [
                    "clone",
                    "--quiet",
                    "--no-hardlinks",
                    str(source["path"]),
                    str(sglang),
                ],
                cwd=root,
            )
            git(["checkout", "--quiet", source["base_commit"]], cwd=sglang)
        else:
            sglang.mkdir(parents=True, exist_ok=True)
            (sglang / "README.md").write_text("# fixture sglang\n", encoding="utf-8")
            git(["init", "--quiet", "."], cwd=sglang)
            git(["add", "-A"], cwd=sglang)
            git(["commit", "--quiet", "-m", "fixture source"], cwd=sglang)
        source_commit = git(["rev-parse", "HEAD"], cwd=sglang).strip()

    config = build_config(
        root=root,
        source=source,
        dependency_commits=commits,
        source_commit=source_commit,
    )
    config_path = root / "config" / "project.local.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    return {
        "root": root,
        "config_path": config_path,
        "config": config,
        "dependency_commits": commits,
    }


PLACEHOLDER_SHA = "0" * 40


def build_config(
    root: Path,
    source: Optional[Dict[str, Any]] = None,
    dependency_commits: Optional[Dict[str, str]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None,
    source_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a complete local configuration with no placeholders.

    ``source_commit`` is the commit ``external/sglang`` is actually checked out
    at, so ``doctor`` sees a consistent pin. It falls back to the source
    repository's base commit, then to a placeholder SHA.
    """
    commits = dependency_commits or {}
    if source_commit is None:
        source_commit = source["base_commit"] if source else PLACEHOLDER_SHA
    config: Dict[str, Any] = {
        "schema_version": 1,
        "project_id": "kimi-k3-kda",
        "model": {"id": "kimi-k3", "display_name": "Kimi K3"},
        "source": {
            "submodule": "external/sglang",
            "repository": "https://example.invalid/fixture/sglang.git",
            "branch": "main",
            "commit": source_commit,
        },
        "dependencies": {
            "kda": {
                "path": "external/kda",
                "repository": "https://example.invalid/fixture/kda.git",
                "commit": commits.get("external/kda", PLACEHOLDER_SHA),
            },
            "humanize": {
                "path": "external/humanize",
                "repository": "https://example.invalid/fixture/humanize.git",
                "commit": commits.get("external/humanize", PLACEHOLDER_SHA),
            },
        },
        "skills": {
            "project": "skills/kernel-optimization/SKILL.md",
            "kernelwiki": "external/kda/skills/KernelWiki/SKILL.md",
            "ncu_report": "external/kda/skills/ncu-report-skill/SKILL.md",
            "humanize": "external/humanize/skills/humanize/SKILL.md",
        },
        "workflow": {
            "writer": {
                "command": "claude",
                "model": "opus",
                "effort": "max",
                "permission_mode": "acceptEdits",
            },
            # An upstream-supported effort by default, so a fixture project is
            # startable without the overlay. Tests that specifically exercise
            # the non-upstream effort -- and the refusal to downgrade it
            # silently -- override this to "ultra".
            "reviewer": {
                "command": "codex",
                "model": "gpt-6-astra",
                "effort": "xhigh",
            },
            "humanize": {
                "plugin_dir": "external/humanize",
                "start_command": "/humanize:start-rlcr-loop",
            },
            "kda": {
                "prompt": "external/kda/prompts/basic-flow.md",
                "agent_flow": "external/kda/docs/agent-flow.md",
            },
        },
        "runner": {"adapter": "local-command", "environment": {}, "commands": {}},
        "no_commit_overlay": {
            "enabled": False,
            "workspace": "external/.overlay/humanize-no-commit",
        },
        "tasks": tasks if tasks is not None else [],
    }
    if source:
        config["source"]["local_path"] = str(source["path"])
    return config


def write_config(project: Dict[str, Any], config: Dict[str, Any]) -> None:
    project["config_path"].write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    project["config"] = config


HARNESS_CORRECTNESS = '''#!/usr/bin/env python3
"""Fixture correctness harness. Writes a report; never touches a GPU.

Behaviour is driven by environment variables so a test can produce passing
evidence, failing evidence, or no evidence at all:

``K3_FIXTURE_STATUS``   status written into the report (default ``passed``)
``K3_FIXTURE_CASES``    executed case count (default ``11``)
``K3_FIXTURE_NO_REPORT``  when set, exit without writing anything
``K3_FIXTURE_EXIT``     process exit code (default ``0``)
``K3_FIXTURE_SLEEP``    seconds to sleep before writing, for cancel tests
"""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    if os.environ.get("K3_FIXTURE_SLEEP"):
        time.sleep(float(os.environ["K3_FIXTURE_SLEEP"]))

    if os.environ.get("K3_FIXTURE_NO_REPORT"):
        return int(os.environ.get("K3_FIXTURE_EXIT", "0"))

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "fixture": True,
                "is_real_evidence": False,
                "status": os.environ.get("K3_FIXTURE_STATUS", "passed"),
                "source_root": args.source_root,
                "oracle_case_count": int(os.environ.get("K3_FIXTURE_CASES", "11")),
                "note": "Fixture output. Not evidence about any kernel.",
            },
            indent=2,
        )
        + "\\n",
        encoding="utf-8",
    )
    return int(os.environ.get("K3_FIXTURE_EXIT", "0"))


if __name__ == "__main__":
    raise SystemExit(main())
'''

HARNESS_BENCHMARK = '''#!/usr/bin/env python3
"""Fixture A/B harness writing a report *directory* with summary.json.

Mirrors the real harness contract: ``summary.json`` starts ``failed`` and is only
rewritten as ``passed`` once every phase completes, and it names the per-trial
reports it depends on.

``K3_FIXTURE_STATUS``      final summary status (default ``passed``)
``K3_FIXTURE_CASES``       case count recorded in the summary (default ``51``)
``K3_FIXTURE_NO_SUMMARY``  write trial reports but never the summary
``K3_FIXTURE_DROP_REPORT`` reference a trial report but do not write it
"""

import argparse
import json
import os
from pathlib import Path


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    if Path(args.baseline_root).resolve() == Path(args.candidate_root).resolve():
        raise SystemExit("baseline and candidate roots are identical")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trial_report = out / "trial-01.candidate.json"
    if not os.environ.get("K3_FIXTURE_DROP_REPORT"):
        write(trial_report, {"fixture": True, "status": "passed", "trial": 1})

    if os.environ.get("K3_FIXTURE_NO_SUMMARY"):
        return 0

    write(
        out / "summary.json",
        {
            "fixture": True,
            "is_real_evidence": False,
            "status": os.environ.get("K3_FIXTURE_STATUS", "passed"),
            "baseline_comparison_case_count": int(
                os.environ.get("K3_FIXTURE_CASES", "51")
            ),
            "trial_reports": [trial_report.name],
            "trials": args.trials,
            "samples": args.samples,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def add_task(
    project: Dict[str, Any],
    task_id: str,
    base_commit: str,
    kind: str = "prefill",
    status: str = "ready",
    workload_status: str = "resolved",
    with_harness: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Register a task package in the fixture project."""
    root: Path = project["root"]
    task_dir = root / "tasks" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "schema_version": 1,
        "task_id": task_id,
        "project_id": project["config"]["project_id"],
        "kind": kind,
        "status": status,
        "source": {
            "repository": "https://example.invalid/fixture/sglang.git",
            "branch": "main",
            "commit": base_commit,
        },
        "base_commit": base_commit,
        "workload_file": "workloads.resolved.json",
        "model_profile_file": "model-profile.json",
        "workload_status": workload_status,
        "optimization_status": "unresolved",
        "integration_status": "not_applied",
        "serving_validation": "not_performed",
        "contract_file": "contract.md",
        "plan_input_file": "plan-input.md",
    }
    if extra:
        record.update(extra)
    (task_dir / "task.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    (task_dir / "workloads.resolved.json").write_text(
        json.dumps({"schema_version": 1, "cases": [{"id": "case-0"}]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (task_dir / "model-profile.json").write_text(
        json.dumps({"schema_version": 1, "model": "kimi-k3"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (task_dir / "contract.md").write_text(
        "# Contract\n\nFixture contract.\n", encoding="utf-8"
    )
    (task_dir / "plan-input.md").write_text(
        "# Plan input\n\n## Goal\n\nFixture goal.\n\n"
        "## Acceptance criteria\n\n- AC-1: fixture criterion.\n"
        "- AC-2: another criterion.\n\n## Steps\n\n1. Inspect.\n2. Implement.\n",
        encoding="utf-8",
    )
    if with_harness:
        bench = task_dir / "bench"
        bench.mkdir(exist_ok=True)
        (bench / "correctness.py").write_text(HARNESS_CORRECTNESS, encoding="utf-8")
        (bench / "benchmark.py").write_text(HARNESS_BENCHMARK, encoding="utf-8")

    config = dict(project["config"])
    task_entries = list(config.get("tasks") or [])
    task_entries.append({"id": task_id, "kind": kind, "path": "tasks/%s" % task_id})
    config["tasks"] = task_entries
    config["active_task"] = task_id
    write_config(project, config)
    return task_dir


MOCK_CLAUDE = r'''#!/usr/bin/env bash
# Mock Claude CLI. Records argv and emits a stream-json transcript.
# It never contacts a model and never performs work.
set -euo pipefail
record="${K3_MOCK_RECORD:-/dev/null}"
{
  printf 'claude'
  for arg in "$@"; do printf ' %s' "$arg"; done
  printf '\n'
} >> "$record"

session_id="fixture-session"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-id) session_id="${2:-fixture-session}"; shift 2 ;;
    *) shift ;;
  esac
done

printf '{"type":"system","subtype":"init","session_id":"%s"}\n' "$session_id"
printf '{"type":"assistant","session_id":"%s"}\n' "$session_id"
if [[ -n "${K3_MOCK_CLAUDE_SLEEP:-}" ]]; then
  sleep "$K3_MOCK_CLAUDE_SLEEP"
fi
printf '{"type":"result","subtype":"success","is_error":false,"num_turns":1,"session_id":"%s"}\n' "$session_id"
exit "${K3_MOCK_CLAUDE_EXIT:-0}"
'''

MOCK_CODEX = r'''#!/usr/bin/env bash
# Mock Codex CLI. Records argv and emits canned review output.
# It never contacts a model.
set -euo pipefail
record="${K3_MOCK_RECORD:-/dev/null}"
{
  printf 'codex'
  for arg in "$@"; do printf ' %s' "$arg"; done
  printf '\n'
} >> "$record"

if [[ -n "${K3_MOCK_CODEX_OUTPUT:-}" ]]; then
  printf '%s\n' "$K3_MOCK_CODEX_OUTPUT"
fi
exit "${K3_MOCK_CODEX_EXIT:-0}"
'''

MOCK_JQ_NOTE = "jq is required by the hooks and is used unmocked when present."


def make_mock_bin(base: Path) -> Path:
    """Create a directory of mock agent CLIs and return it."""
    target = base / "mockbin"
    target.mkdir(parents=True, exist_ok=True)
    for name, body in (("claude", MOCK_CLAUDE), ("codex", MOCK_CODEX)):
        path = target / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    return target


class TempDirCase(unittest.TestCase):
    """Base case providing a temporary directory that is always cleaned up."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.base = Path(self._temp.name)
        self.addCleanup(self._temp.cleanup)

    def read_json(self, path: Path) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))
