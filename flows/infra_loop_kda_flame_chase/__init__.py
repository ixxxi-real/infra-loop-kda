"""Protected Flame Chase flow used by Infra Loop-KDA.

This is a small adapter around the pinned Humanize2/Flowverse implementation.
The upstream flow remains responsible for fresh sessions, steering, budgets and
history cleanup.  The adapter adds one project invariant: the task control and
evidence directory is snapshotted outside the candidate tree around every
cleanup epoch and restored afterwards.  A cleaner can therefore summarize the
workspace, but cannot delete or rewrite the candidate ledger, benchmark logs,
failure records or profiler artifacts.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

from hmz.flows import AgentCollection, EnvCollection, FlowContext, Outworlder, flow


_ROOT = Path(__file__).resolve().parents[2]
_UPSTREAM = _ROOT / "external" / "flowverse" / "flows" / "flame_chase_agent_cleanup"
if not _UPSTREAM.is_dir():
    raise ImportError("pinned Flowverse flame_chase_agent_cleanup is missing: %s" % _UPSTREAM)

# Flowverse's implementation intentionally keeps its helper package as a plain
# sibling named ``_flame_chase_agent_cleanup``.  Loading that package from the
# pinned checkout lets this adapter track the exact upstream flow without
# copying its implementation into the project.
sys.path.insert(0, str(_UPSTREAM))
try:
    from _flame_chase_agent_cleanup import Config, Worker, Workspace, drive
    import _flame_chase_agent_cleanup.loop as _upstream_loop
finally:
    sys.path.remove(str(_UPSTREAM))


async def _snapshot_protected(env: Any) -> str | None:
    """Copy protected task evidence to a temporary directory outside the repo."""
    command = """
set -eu
backup=$(mktemp -d "${TMPDIR:-/tmp}/infra-loop-kda-evidence.XXXXXX")
if [ -e .kda-task ] || [ -L .kda-task ]; then
  cp -PRp -- .kda-task "$backup/"
else
  : > "$backup/.missing"
fi
printf '%s' "$backup"
"""
    done, output, error = await env.exec(command)
    if done:
        raise RuntimeError("could not snapshot protected evidence: %s" % error.strip())
    return output.strip() or None


async def _restore_protected(env: Any, backup: str | None) -> None:
    """Restore the protected directory even when the cleaner removed it."""
    if not backup:
        return
    quoted_backup = shlex.quote(backup)
    command = f"""
set -eu
backup={quoted_backup}
if [ -f "$backup/.missing" ]; then
  rm -rf -- .kda-task
else
  restored=$(mktemp -d "${{TMPDIR:-/tmp}}/infra-loop-kda-restore.XXXXXX")
  cp -PRp -- "$backup/.kda-task" "$restored/kda-task"
  rm -rf -- .kda-task
  mv -- "$restored/kda-task" .kda-task
  rmdir -- "$restored"
fi
rm -rf -- "$backup"
"""
    done, _, error = await env.exec(command)
    if done:
        raise RuntimeError("could not restore protected evidence: %s" % error.strip())


async def _protected_clean_epoch(
    cleaner: Any,
    held: Config,
    env: Any,
    manifest: set[str],
    store: Any,
    epoch: int,
) -> None:
    backup = await _snapshot_protected(env)
    try:
        await _ORIGINAL_CLEAN_EPOCH(cleaner, held, env, manifest, store, epoch)
    finally:
        await _restore_protected(env, backup)


_ORIGINAL_CLEAN_EPOCH = _upstream_loop.clean_epoch
_upstream_loop.clean_epoch = _protected_clean_epoch


class Agents(AgentCollection):
    first_chaser: Worker
    second_chaser: Worker
    cleaner: Worker
    human: Outworlder


class Envs(EnvCollection):
    workspace: Workspace


@flow(agents=Agents, envs=Envs, params=Config, description=__doc__, resumable=True)
async def infra_loop_kda_flame_chase(
    task: str, *, agents: Agents, envs: Envs, params: Config, ctx: FlowContext
) -> None:
    await drive(
        "infra_loop_kda_flame_chase",
        (agents["first_chaser"], agents["second_chaser"]),
        agents["cleaner"],
        agents["human"],
        task,
        params,
        ctx,
        envs["workspace"],
    )
