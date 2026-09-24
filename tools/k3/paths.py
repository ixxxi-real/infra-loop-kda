#!/usr/bin/env python3
"""Project root discovery and safe path handling.

The project root is located by markers instead of a fixed number of parent
directories. An installed ``k3ctl`` console script therefore cannot silently
bind to an unrelated tree, and ``K3_PROJECT_ROOT`` gives an explicit override
for foreign working directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

#: Files and directories that together identify this repository.
ROOT_MARKERS: Sequence[str] = (
    "tools/k3ctl.py",
    "config/project.example.json",
    "tasks",
)


class PathError(ValueError):
    """Raised for unsafe, escaping or unresolvable paths."""


def looks_like_root(path: Path) -> bool:
    """Return True when *path* contains every root marker."""
    try:
        return all((path / marker).exists() for marker in ROOT_MARKERS)
    except OSError:
        return False


def _ancestors(path: Path) -> Iterator[Path]:
    yield path
    for parent in path.parents:
        yield parent


def find_root(
    start: Optional[Path] = None,
    env: Optional[dict] = None,
) -> Path:
    """Locate the project root.

    Order: ``K3_PROJECT_ROOT``, then upwards from *start*, then upwards from
    this module, then upwards from the current directory.
    """
    environ = os.environ if env is None else env
    explicit = environ.get("K3_PROJECT_ROOT")
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_dir():
            raise PathError("K3_PROJECT_ROOT does not exist: %s" % explicit)
        candidate = candidate.resolve()
        if not looks_like_root(candidate):
            raise PathError(
                "K3_PROJECT_ROOT is not a infra-loop-kda checkout: %s" % candidate
            )
        return candidate

    seeds: list[Path] = []
    if start is not None:
        seeds.append(Path(start))
    seeds.append(Path(__file__).resolve().parent)
    seeds.append(Path.cwd())

    for seed in seeds:
        try:
            resolved = seed.resolve()
        except OSError:
            continue
        for candidate in _ancestors(resolved):
            if looks_like_root(candidate):
                return candidate

    raise PathError(
        "cannot locate the infra-loop-kda root; set K3_PROJECT_ROOT "
        "(expected markers: %s)" % ", ".join(ROOT_MARKERS)
    )


def contains_symlink(root: Path, relative: str) -> Optional[str]:
    """Return the first symlinked component under *root*, or None."""
    current = root
    for part in Path(relative).parts:
        current = current / part
        try:
            if current.is_symlink():
                return str(current)
        except OSError:
            return str(current)
    return None


def safe_relative(raw: str) -> Path:
    """Validate a relative path with no upward traversal."""
    if not raw:
        raise PathError("empty path")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise PathError("path must be relative: %s" % raw)
    if any(part == ".." for part in candidate.parts):
        raise PathError("path must not traverse upwards: %s" % raw)
    if "\x00" in raw:
        raise PathError("path contains a NUL byte")
    return candidate


def safe_join(root: Path, raw: str, allow_symlink: bool = False) -> Path:
    """Join *raw* under *root*, refusing escapes and (by default) symlinks."""
    relative = safe_relative(raw)
    root_resolved = root.resolve()
    resolved = Path(os.path.normpath(str(root_resolved / relative)))
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise PathError("path escapes the project root: %s" % raw)
    if not allow_symlink:
        found = contains_symlink(root_resolved, str(relative))
        if found is not None:
            raise PathError("path traverses a symbolic link: %s" % found)
    return resolved


def relative_to_root(root: Path, path: Path) -> str:
    """Render *path* relative to *root* when possible, else absolute."""
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path)
