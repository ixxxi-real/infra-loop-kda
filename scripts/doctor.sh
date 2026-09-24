#!/usr/bin/env bash
#
# Thin wrapper around `k3ctl doctor`.
#
# The inspection logic lives in one place (tools/k3/toolchain.py) so the shell
# entrypoint and the CLI can never disagree. This script exists because a fresh
# clone may have nothing installed yet: it runs the in-tree module directly.
#
# Read-only. No installation, no network access, no model or GPU invocation.
#
# Usage:
#   scripts/doctor.sh                       # basic metadata, works without submodules
#   scripts/doctor.sh --agent-profile       # also check bash>=4, jq, claude, codex
#   scripts/doctor.sh --json                # machine-readable
#   scripts/doctor.sh --config config/project.local.json --task prefill-kda
#
# Any argument is forwarded to `k3ctl doctor` unchanged.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-}"
if [[ -z "$python_bin" ]]; then
  if [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_bin="$repo_root/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
  else
    echo "doctor: python3 is not available; install Python 3.9 or newer" >&2
    exit 1
  fi
fi

exec "$python_bin" tools/k3ctl.py doctor "$@"
