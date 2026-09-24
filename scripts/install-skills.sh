#!/usr/bin/env bash
#
# Link this project's skills into a Codex/agent skills directory.
#
# SAFETY MODEL
#
# This script does nothing by default. Run without arguments it only *reports*
# what it would do, because the destination is a user's home directory and an
# accidental invocation should never mutate it.
#
#   scripts/install-skills.sh                 # check only: report, write nothing
#   scripts/install-skills.sh --apply         # create the symlinks
#   scripts/install-skills.sh --target DIR    # explicit skills directory
#   scripts/install-skills.sh --home DIR      # explicit Codex home (DIR/skills)
#
# The upstream Humanize installer is NOT run implicitly. It writes into the
# user's home and installs a native Codex stop hook, and this project drives
# Humanize through Claude's `--plugin-dir` instead, which needs no global
# installation at all. Request it explicitly if you want it:
#
#   scripts/install-skills.sh --apply --with-humanize-installer
#
# Existing symlinks are replaced; an existing *non*-symlink is never touched.
#
# Read-only apart from the symlinks it is explicitly asked to create. No network
# access, no package installation, no model or GPU invocation.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$repo_root"

apply="false"
run_humanize_installer="false"
target_dir=""
codex_home="${CODEX_HOME:-$HOME/.codex-bak}"

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]:-$0}" | sed 's/^#//; s/^ //'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      apply="true"
      shift
      ;;
    --check|--dry-run)
      apply="false"
      shift
      ;;
    --target)
      [[ -n "${2:-}" ]] || { echo "error: --target requires a directory" >&2; exit 2; }
      target_dir="$2"
      shift 2
      ;;
    --home)
      [[ -n "${2:-}" ]] || { echo "error: --home requires a directory" >&2; exit 2; }
      codex_home="$2"
      shift 2
      ;;
    --with-humanize-installer)
      run_humanize_installer="true"
      shift
      ;;
    -h|--help)
      usage 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      echo "run with --help for usage" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$target_dir" ]]; then
  target_dir="$codex_home/skills"
fi

# ---------------------------------------------------------------- preflight

missing=0
if [[ ! -f external/kda/skills/KernelWiki/SKILL.md ]] \
   || [[ ! -f external/kda/skills/ncu-report-skill/SKILL.md ]]; then
  echo "miss external/kda skills are not present" >&2
  echo "     run: git submodule update --init --recursive" >&2
  missing=1
fi
if [[ ! -f skills/kernel-optimization/SKILL.md ]]; then
  echo "miss skills/kernel-optimization/SKILL.md is not present" >&2
  missing=1
fi
if [[ "$missing" -ne 0 ]]; then
  # State the outcome explicitly on every exit path. A caller -- or a CI step --
  # must be able to confirm that nothing was written without having to infer it
  # from an exit code, and a fresh checkout without submodules reaches exactly
  # this branch.
  echo "Nothing was written: required skill sources are missing." >&2
  exit 1
fi

# name -> source path, as parallel arrays (portable to bash 3.2).
skill_names=(kernel-optimization kernelwiki ncu-report-skill)
skill_sources=(
  skills/kernel-optimization
  external/kda/skills/KernelWiki
  external/kda/skills/ncu-report-skill
)

# ------------------------------------------------------------------- report

if [[ "$apply" != "true" ]]; then
  printf 'check mode: nothing will be written.\n\n'
  printf 'destination : %s\n' "$target_dir"
  printf 'exists      : %s\n\n' "$([[ -d "$target_dir" ]] && echo yes || echo 'no, would be created')"
  printf 'would link:\n'
fi

status=0
index=0
while [[ "$index" -lt "${#skill_names[@]}" ]]; do
  name="${skill_names[$index]}"
  source="${skill_sources[$index]}"
  destination="$target_dir/$name"
  index=$((index + 1))

  if [[ -e "$destination" && ! -L "$destination" ]]; then
    printf 'BLOCKED %-18s %s exists and is not a symlink\n' "$name" "$destination" >&2
    printf '        move it aside deliberately, then re-run\n' >&2
    status=1
    continue
  fi

  if [[ "$apply" != "true" ]]; then
    current=""
    if [[ -L "$destination" ]]; then
      current="$(readlink "$destination" 2>/dev/null || true)"
    fi
    if [[ "$current" == "$repo_root/$source" ]]; then
      printf '  ok      %-18s already linked -> %s\n' "$name" "$source"
    elif [[ -n "$current" ]]; then
      printf '  relink  %-18s %s -> %s\n' "$name" "$current" "$source"
    else
      printf '  create  %-18s -> %s\n' "$name" "$source"
    fi
    continue
  fi

  mkdir -p "$target_dir"
  ln -sfn "$repo_root/$source" "$destination"
  printf 'linked  %-18s %s -> %s\n' "$name" "$destination" "$source"
done

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

# ------------------------------------------------- optional upstream installer

if [[ "$run_humanize_installer" == "true" ]]; then
  if [[ ! -x external/humanize/scripts/install-skills-codex.sh ]]; then
    echo "miss external/humanize installer is not present" >&2
    echo "     run: git submodule update --init --recursive" >&2
    exit 1
  fi
  if [[ "$apply" != "true" ]]; then
    printf '\nwould also run the upstream Humanize installer:\n'
    printf '  external/humanize/scripts/install-skills-codex.sh\n'
    printf '  (writes into the Codex home and installs a native stop hook)\n'
  else
    printf '\nrunning the upstream Humanize installer (explicitly requested)\n'
    # Humanize owns its own installation: it also installs a native Codex stop
    # hook and runtime support files. That behaviour stays in the upstream tool.
    CODEX_HOME="$codex_home" external/humanize/scripts/install-skills-codex.sh
  fi
fi

# ------------------------------------------------------------------ epilogue

if [[ "$apply" != "true" ]]; then
  cat <<EOF

Nothing was written. To apply:
  scripts/install-skills.sh --apply

To try it against a throwaway directory first:
  scripts/install-skills.sh --apply --home "\$(pwd)/.codex-test"
EOF
  exit 0
fi

cat <<EOF

Skills linked in $target_dir:
  kimi-k3-kda       project contract and validation rules
  kernelwiki        KDA kernel references (from external/kda)
  ncu-report-skill  Nsight Compute analysis (from external/kda)

Humanize is not installed globally by this script. This project loads the
pinned plugin directly with Claude's --plugin-dir, so no global installation
is required. Pass --with-humanize-installer if you specifically want the
upstream installer, which also installs a native Codex stop hook.
EOF
