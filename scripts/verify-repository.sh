#!/usr/bin/env bash
# Read-only checks for a clean, publishable checkout.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$repo_root"

required_files=(
  README.md LICENSE Makefile pyproject.toml setup.py
  .gitmodules config/project.example.json tools/k3ctl.py
)
for path in "${required_files[@]}"; do
  [[ -e "$path" ]] || { echo "repository-check: missing $path" >&2; exit 1; }
done

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "repository-check: run inside a Git checkout" >&2
  exit 1
fi

if git diff --name-only --diff-filter=U | grep -q .; then
  echo "repository-check: unresolved merge conflicts" >&2
  exit 1
fi

# Public metadata must never depend on one developer's machine or GPU.
if git grep -n -E '/(Users|home)/|GPU-[0-9A-Fa-f-]{20,}' -- \
    ':!docs/getting-started.md' ':!README.md' ':!**/test_metrics.py' ':!external/**' >/dev/null 2>&1; then
  echo "repository-check: machine-specific absolute path or GPU UUID found" >&2
  exit 1
fi

if grep -Eq 'url = git@|url = ssh://' .gitmodules; then
  echo "repository-check: submodule URLs must be cloneable over HTTPS" >&2
  exit 1
fi

python_bin="${PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  if [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_bin="$repo_root/.venv/bin/python"
  else
    python_bin=python3
  fi
fi
"$python_bin" -m py_compile tools/k3ctl.py tools/k3/*.py

# The parent repository is English-only. Submodule contents are separate
# upstream repositories and are intentionally outside this check.
"$python_bin" - <<'PY'
import subprocess
import sys

paths = subprocess.check_output(["git", "ls-files", "-z"]).split(b"\0")
hits = []
for raw in paths:
    if not raw:
        continue
    path = raw.decode("utf-8")
    try:
        text = open(path, encoding="utf-8").read()
    except (UnicodeDecodeError, OSError):
        continue
    if any("\u3400" <= char <= "\u9fff" for char in text):
        hits.append(path)
if hits:
    print("repository-check: Chinese characters found in tracked files:", file=sys.stderr)
    print("\n".join(hits), file=sys.stderr)
    raise SystemExit(1)
PY

echo "repository-check: passed"
