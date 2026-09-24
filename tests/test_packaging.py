"""Packaging and entrypoint invariants.

These tests guard failures that only appear once the package is *installed* and
run from outside the source tree -- exactly where they are least likely to be
noticed:

* the console script imports ``tools.k3``, so omitting that package from the
  distribution produces a ``k3ctl`` that cannot import its own modules;
* root discovery must not happen at import time, or ``k3ctl --help`` crashes
  with a traceback wherever the project root cannot be found;
* discovery must be marker-based, so an installed script never binds to an
  unrelated directory that happens to be the current working directory.

No network access and no installation are required: the same code paths are
exercised in-tree via a subprocess with a foreign working directory.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import fixtures

REAL_ROOT = fixtures.REAL_ROOT
ENTRY_TARGET = "tools.k3ctl:main"
REQUIRED_PACKAGES = {"tools", "tools.k3"}


def _run(argv, cwd, env=None):
    environment = dict(os.environ)
    environment.pop("K3_PROJECT_ROOT", None)
    if env:
        environment.update(env)
    return subprocess.run(
        argv,
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )


class DeclaredPackagesTests(unittest.TestCase):
    """Both packaging manifests must ship every package the entrypoint needs."""

    def test_setup_py_declares_tools_and_tools_k3(self) -> None:
        source = (REAL_ROOT / "setup.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        packages = None
        entry_points = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "packages":
                        packages = ast.literal_eval(keyword.value)
                    elif keyword.arg == "entry_points":
                        entry_points = ast.literal_eval(keyword.value)
        self.assertIsNotNone(packages, "setup.py declares no packages")
        self.assertTrue(
            REQUIRED_PACKAGES.issubset(set(packages)),
            "setup.py must declare %s, found %s" % (REQUIRED_PACKAGES, packages),
        )
        self.assertIsNotNone(entry_points)
        scripts = entry_points.get("console_scripts") or []
        self.assertTrue(
            any(ENTRY_TARGET in entry for entry in scripts),
            "console_scripts must point at %s, found %s" % (ENTRY_TARGET, scripts),
        )

    def test_pyproject_declares_tools_and_tools_k3(self) -> None:
        text = (REAL_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for package in sorted(REQUIRED_PACKAGES):
            with self.subTest(package=package):
                self.assertIn(
                    '"%s"' % package,
                    text,
                    "pyproject.toml must declare the %s package" % package,
                )
        self.assertIn(ENTRY_TARGET, text)

    def test_every_module_is_inside_a_declared_package(self) -> None:
        """A new module must not land outside the distributed packages."""
        modules = sorted(
            str(path.relative_to(REAL_ROOT))
            for path in (REAL_ROOT / "tools").rglob("*.py")
            if "__pycache__" not in path.parts
        )
        self.assertTrue(modules)
        for relative in modules:
            with self.subTest(module=relative):
                parts = Path(relative).parts
                package = ".".join(parts[:-1]) if len(parts) > 1 else "tools"
                self.assertIn(
                    package,
                    REQUIRED_PACKAGES,
                    "%s lives in undeclared package %r" % (relative, package),
                )


class EntrypointTests(unittest.TestCase):
    """The entrypoint must be usable from outside the source tree."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = REAL_ROOT / "tools" / "k3ctl.py"

    def test_help_works_from_a_foreign_working_directory(self) -> None:
        """Root discovery at import time would make --help crash here."""
        with tempfile.TemporaryDirectory() as foreign:
            result = _run([sys.executable, str(self.script), "--help"], cwd=foreign)
        self.assertEqual(
            result.returncode,
            0,
            "k3ctl --help failed from a foreign cwd: %s"
            % result.stderr.decode("utf-8", "replace")[-800:],
        )
        self.assertIn(b"usage: k3ctl", result.stdout)

    def test_module_imports_without_a_discoverable_root(self) -> None:
        """Importing the dispatcher must never require a discoverable root."""
        with tempfile.TemporaryDirectory() as foreign:
            result = _run(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.path.insert(0, %r); "
                    "import tools.k3ctl; print('imported')" % str(REAL_ROOT),
                ],
                cwd=foreign,
            )
        self.assertEqual(
            result.returncode,
            0,
            "importing tools.k3ctl failed: %s"
            % result.stderr.decode("utf-8", "replace")[-800:],
        )
        self.assertIn(b"imported", result.stdout)

    def test_root_attribute_still_resolves_for_legacy_callers(self) -> None:
        result = _run(
            [
                sys.executable,
                "-c",
                "from tools import k3ctl; print(k3ctl.ROOT)",
            ],
            cwd=REAL_ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(
            result.stdout.decode().strip(), str(REAL_ROOT)
        )

    def test_unknown_module_attribute_still_raises(self) -> None:
        """The lazy attribute hook must not swallow genuine typos."""
        result = _run(
            [
                sys.executable,
                "-c",
                "from tools import k3ctl; k3ctl.NOT_A_REAL_ATTRIBUTE",
            ],
            cwd=REAL_ROOT,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"AttributeError", result.stderr)

    def test_explicit_root_is_honoured_from_a_foreign_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as foreign:
            result = _run(
                [
                    sys.executable,
                    str(self.script),
                    "validate",
                    "--config",
                    "config/project.example.json",
                ],
                cwd=foreign,
                env={"K3_PROJECT_ROOT": str(REAL_ROOT)},
            )
        self.assertEqual(
            result.returncode,
            0,
            result.stderr.decode("utf-8", "replace")[-800:],
        )
        payload = json.loads(result.stdout.decode("utf-8"))
        self.assertEqual(payload["status"], "passed")

    def test_unrelated_directory_is_refused_not_adopted(self) -> None:
        """An unrelated cwd must produce an actionable error, never a wrong root."""
        with tempfile.TemporaryDirectory() as foreign:
            # A directory that superficially resembles a checkout.
            (Path(foreign) / "tasks").mkdir()
            result = _run(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.path.insert(0, %r); "
                    "from tools.k3 import paths; print(paths.find_root())"
                    % str(REAL_ROOT),
                ],
                cwd=foreign,
                env={"K3_PROJECT_ROOT": foreign},
            )
        self.assertNotEqual(
            result.returncode, 0, "an unrelated directory was accepted as the root"
        )
        self.assertIn(b"not a infra-loop-kda checkout", result.stderr)


class ShellEntrypointTests(unittest.TestCase):
    """The shell wrappers must stay syntactically valid and thin."""

    def test_scripts_parse(self) -> None:
        from tools.k3 import toolchain

        check = toolchain.bash_check()
        if check["status"] != toolchain.OK:
            self.skipTest("bash >= 4 is unavailable: %s" % check["detail"])
        for name in ("doctor.sh", "install-skills.sh"):
            script = REAL_ROOT / "scripts" / name
            with self.subTest(script=name):
                self.assertTrue(script.is_file(), "%s is missing" % name)
                result = subprocess.run(
                    [str(check["path"]), "-n", str(script)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    "%s has a syntax error: %s"
                    % (name, result.stderr.decode("utf-8", "replace")),
                )

    def test_doctor_wrapper_delegates_to_the_cli(self) -> None:
        """One implementation, so the wrapper and CLI cannot disagree."""
        text = (REAL_ROOT / "scripts" / "doctor.sh").read_text(encoding="utf-8")
        self.assertIn("tools/k3ctl.py doctor", text)


class SkillsInstallerTests(unittest.TestCase):
    """The installer must never mutate a global home implicitly.

    Its destination is a user's home directory, so an accidental invocation --
    or a CI run -- must not write there. Every test below targets an isolated
    throwaway directory and additionally asserts the real ``CODEX_HOME`` was not
    created or modified.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from tools.k3 import toolchain

        check = toolchain.bash_check()
        if check["status"] != toolchain.OK:
            raise unittest.SkipTest("bash >= 4 unavailable: %s" % check["detail"])
        cls.bash = str(check["path"])
        cls.script = REAL_ROOT / "scripts" / "install-skills.sh"
        if not (REAL_ROOT / "external" / "kda" / "skills").is_dir():
            raise unittest.SkipTest("external/kda is not initialized")

    def _run_installer(self, *args, home=None):
        env = {}
        if home is not None:
            env["CODEX_HOME"] = str(home)
        return _run([self.bash, str(self.script), *args], cwd=REAL_ROOT, env=env)

    def test_default_invocation_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "codex-home"
            result = self._run_installer(home=home)
            self.assertEqual(
                result.returncode,
                0,
                result.stderr.decode("utf-8", "replace")[-500:],
            )
            self.assertIn(b"check mode", result.stdout)
            self.assertIn(b"Nothing was written", result.stdout)
            self.assertFalse(
                home.exists(),
                "check mode created the destination directory",
            )

    def test_apply_writes_only_to_the_explicit_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "codex-home"
            other = Path(temp) / "untouched"
            other.mkdir()
            result = self._run_installer("--apply", "--home", str(home))
            self.assertEqual(
                result.returncode,
                0,
                result.stderr.decode("utf-8", "replace")[-500:],
            )
            links = sorted(p.name for p in (home / "skills").iterdir())
            self.assertEqual(
                links, ["kernel-optimization", "kernelwiki", "ncu-report-skill"]
            )
            for name in links:
                self.assertTrue((home / "skills" / name).is_symlink())
            self.assertEqual(sorted(other.iterdir()), [])

    def test_explicit_target_directory_is_honoured(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "custom-skills"
            result = self._run_installer("--apply", "--target", str(target))
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertTrue((target / "kernel-optimization").is_symlink())

    def test_existing_non_symlink_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "skills"
            target.mkdir()
            precious = target / "kernel-optimization"
            precious.mkdir()
            (precious / "user-file.md").write_text("keep me\n", encoding="utf-8")

            result = self._run_installer("--apply", "--target", str(target))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"not a symlink", result.stderr)
            self.assertTrue((precious / "user-file.md").is_file())
            self.assertFalse(precious.is_symlink())

    def test_upstream_humanize_installer_is_not_run_implicitly(self) -> None:
        """It writes into the home and installs a native stop hook."""
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "codex-home"
            result = self._run_installer("--apply", "--home", str(home))
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertIn(b"not installed globally by this script", result.stdout)
            # Only the three project symlinks exist; no hook was installed.
            self.assertEqual(
                sorted(p.name for p in home.iterdir()),
                ["skills"],
                "the upstream installer appears to have run implicitly",
            )

    def test_apply_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "skills"
            first = self._run_installer("--apply", "--target", str(target))
            self.assertEqual(first.returncode, 0, first.stderr.decode())
            second = self._run_installer("--apply", "--target", str(target))
            self.assertEqual(second.returncode, 0, second.stderr.decode())
            check = self._run_installer("--target", str(target))
            self.assertIn(b"already linked", check.stdout)

    def test_unknown_argument_is_refused(self) -> None:
        result = self._run_installer("--not-a-real-flag")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"unknown argument", result.stderr)


if __name__ == "__main__":
    unittest.main()
