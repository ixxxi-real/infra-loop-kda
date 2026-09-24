"""Focused, stdlib-only modules behind the ``k3ctl`` control-plane CLI.

Module map:

``paths``       project root discovery and safe path handling
``util``        hashing, JSON IO, subprocess, locks, process identity
``gitq``        read-only Git queries (never mutates a repo, index or ref)
``config``      project/task configuration loading and validation
``toolchain``   offline toolchain inspection (the ``doctor`` command)
``lifecycle``   task states and the gates that guard process launch
``taskfactory`` task scaffolding (``task-create``)
``workspace``   isolated source workspace preparation
``overlay``     audited no-commit Humanize compatibility overlay
``agent``       Claude + Humanize loop adapter
``runner``      runner adapter protocol and the local-command adapter
``supervise``   exit-status supervisor for detached runs
``export``      delivery and evidence export
``cli``         argument parsing and command dispatch
"""

__all__ = [
    "agent",
    "cli",
    "config",
    "export",
    "gitq",
    "lifecycle",
    "overlay",
    "paths",
    "runner",
    "supervise",
    "taskfactory",
    "toolchain",
    "util",
    "workspace",
]
