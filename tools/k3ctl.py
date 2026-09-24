#!/usr/bin/env python3
"""Canonical control-plane CLI for the Kimi K3 KDA lab.

This module is the single entrypoint. It stays deliberately thin: argument
parsing and command implementations live in the focused modules under
``tools/k3/``.

It also remains the stable import surface. ``ROOT``, ``ValidationError``,
``SHA_RE``, ``load_json``, ``resolve_config``, ``is_example``,
``find_placeholders`` and ``validate`` keep their original names and behaviour
so existing callers and tests continue to work unchanged.

The tool never launches GPU jobs, contacts a gateway, installs anything, or
writes to this repository's Git index. Those belong to an environment-specific
runner and to the operator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):  # direct execution: python3 tools/k3ctl.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.k3 import cli as _cli
from tools.k3 import config as _config
from tools.k3 import paths as _paths

# --------------------------------------------------------- compatibility API

ValidationError = _config.ValidationError
SHA_RE = _config.SHA_RE


def _root() -> Path:
    return _paths.find_root()


def __getattr__(name: str) -> Any:
    """Resolve ``ROOT`` lazily, on first access.

    ``ROOT`` used to be computed at import time. That made the module
    unimportable wherever the project root could not be discovered -- so an
    installed ``k3ctl --help`` crashed with a traceback before argparse ever
    ran, instead of printing usage.

    PEP 562 module ``__getattr__`` keeps the historical ``k3ctl.ROOT``
    attribute working for existing callers while deferring discovery until
    something actually needs it.
    """
    if name == "ROOT":
        return _root()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object, raising :class:`ValidationError` on any problem."""
    return _config.load_json(path)


def resolve_config(raw: str) -> Tuple[Path, Dict[str, Any]]:
    """Resolve a configuration path relative to the project root and load it."""
    return _config.resolve_config(raw, root=_root())


def is_example(path: Path) -> bool:
    return _config.is_example(path)


def find_placeholders(value: Any, prefix: str = "") -> List[str]:
    """Return paths whose string values still contain a template placeholder."""
    return _config.find_placeholders(value, prefix)


def validate(config_path: Path, config: Dict[str, Any]) -> List[str]:
    """Validate a project configuration and return a list of error strings."""
    return _config.validate(config_path, config, root=_root())


def build_parser():
    return _cli.build_parser()


def main(argv: Optional[List[str]] = None) -> int:
    return _cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
