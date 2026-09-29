"""Deterministic parameter-space expansion for the tuner lane.

This module only enumerates configurations.  Compilation and benchmarking stay
with the project-specific bench harness, so an LLM cannot silently change the
acceptance gate while exploring parameters.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List

from . import util


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not str(manifest.get("kernel") or "").strip():
        errors.append("kernel is required")
    space = manifest.get("search_space")
    if not isinstance(space, dict) or not space:
        errors.append("search_space must be a non-empty object")
    else:
        for key, values in space.items():
            if not isinstance(values, list) or not values:
                errors.append("search_space.%s must be a non-empty list" % key)
    if not str(manifest.get("gate_version") or "").strip():
        errors.append("gate_version is required")
    if not isinstance(manifest.get("seed"), int):
        errors.append("seed must be an integer")
    return errors


def expand(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    errors = validate_manifest(manifest)
    if errors:
        raise util.ToolError("invalid tuning manifest: %s" % "; ".join(errors))
    keys = sorted(manifest["search_space"])
    values = [manifest["search_space"][key] for key in keys]
    result: List[Dict[str, Any]] = []
    for index, combination in enumerate(itertools.product(*values)):
        config = dict(zip(keys, combination))
        result.append(
            {
                "trial": index,
                "seed": manifest["seed"],
                "kernel": manifest["kernel"],
                "gate_version": manifest["gate_version"],
                "config": config,
                "config_hash": util.sha256_json(config),
            }
        )
    return result


def load(path: Path) -> Dict[str, Any]:
    manifest = util.read_json(path)
    if not isinstance(manifest, dict):
        raise util.ToolError("tuning manifest must be a JSON object: %s" % path)
    errors = validate_manifest(manifest)
    if errors:
        raise util.ToolError("invalid tuning manifest: %s" % "; ".join(errors))
    return manifest
