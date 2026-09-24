#!/usr/bin/env python3
"""Stdlib-only helpers: hashing, JSON IO, subprocess, locks, process identity."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import subprocess
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CHUNK = 1 << 20


class ToolError(RuntimeError):
    """Actionable, user-facing failure."""


# ---------------------------------------------------------------- time / ids


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def new_uuid() -> str:
    return str(_uuid.uuid4())


# ------------------------------------------------------------------ hashing


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash a JSON-serialisable value with a stable key order."""
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


#: Digest prefixes that keep entry kinds from colliding.
#:
#: A symlink is recorded as a hash of its *target string*. Without a prefix, a
#: symlink to "x" and a regular file containing "x" would produce the same
#: digest, so replacing one with the other would be invisible.
SYMLINK_PREFIX = "symlink\0"
SPECIAL_PREFIX = "special\0"


def tree_hashes(root: Path, skip: Sequence[str] = (".git",)) -> Dict[str, str]:
    """Map relative path -> digest for every entry under *root*.

    Regular files are hashed by content. Symlinks are hashed by their target
    string, so a retargeted link changes the digest: silently skipping them
    would make a changed source symlink invisible to baseline verification.
    Real source trees contain them -- SGLang tracks three -- so they are
    recorded, not rejected.

    Other special entries (FIFOs, sockets, devices) are recorded by kind so
    their presence is never invisible either.
    """
    result: Dict[str, str] = {}
    skip_set = set(skip)
    root = Path(root)
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        # Symlinked directories arrive in dirnames; record them as links here
        # because os.walk (followlinks=False) will not descend into them.
        kept: list = []
        for name in sorted(dirnames):
            if name in skip_set:
                continue
            full = Path(current) / name
            if full.is_symlink():
                relative = str(full.relative_to(root))
                result[relative] = sha256_text(
                    SYMLINK_PREFIX + os.readlink(str(full))
                )
                continue
            kept.append(name)
        dirnames[:] = kept

        for name in sorted(filenames):
            full = Path(current) / name
            relative = str(full.relative_to(root))
            if full.is_symlink():
                try:
                    target = os.readlink(str(full))
                except OSError:
                    target = "<unreadable>"
                result[relative] = sha256_text(SYMLINK_PREFIX + target)
                continue
            if full.is_file():
                result[relative] = sha256_file(full)
                continue
            # FIFO, socket, device, or a dangling entry.
            result[relative] = sha256_text(SPECIAL_PREFIX + name)
    return result


def tree_digest(hashes: Dict[str, str]) -> str:
    """Single digest over a path->hash mapping."""
    lines = "".join(
        "%s\0%s\n" % (path, hashes[path]) for path in sorted(hashes)
    )
    return sha256_text(lines)


# --------------------------------------------------------------------- json


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ToolError("missing JSON file: %s" % path) from exc
    except json.JSONDecodeError as exc:
        raise ToolError("invalid JSON in %s: %s" % (path, exc)) from exc


def write_json(path: Path, value: Any) -> None:
    """Write pretty JSON atomically."""
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    write_text(path, text)


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-%s" % os.getpid())
    temp.write_text(text, encoding="utf-8")
    os.replace(str(temp), str(path))


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- subprocess


def run(
    argv: Sequence[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    stdin_data: Optional[bytes] = None,
) -> Tuple[int, str, str]:
    """Run *argv* without a shell. Returns (rc, stdout, stderr).

    ``stdin_data`` feeds bytes to the process. It exists so NUL-separated
    pathspecs can be passed to Git without going through a shell or an argument
    list, which keeps paths containing spaces, newlines or non-ASCII bytes
    intact.
    """
    if cwd is not None and not Path(cwd).is_dir():
        return 127, "", "working directory does not exist: %s" % cwd
    try:
        completed = subprocess.run(  # noqa: S603 - argv list, no shell
            list(argv),
            cwd=str(cwd) if cwd else None,
            env=env,
            input=stdin_data,
            stdin=None if stdin_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except FileNotFoundError:
        return 127, "", "command not found: %s" % argv[0]
    except subprocess.TimeoutExpired:
        return 124, "", "timed out after %ss: %s" % (timeout, argv[0])
    return (
        completed.returncode,
        completed.stdout.decode("utf-8", "replace"),
        completed.stderr.decode("utf-8", "replace"),
    )


def git_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Environment for read-only Git: no optional locks, no pager, no prompts."""
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_PAGER"] = "cat"
    env["GIT_TERMINAL_PROMPT"] = "0"
    if extra:
        env.update(extra)
    return env


def git(
    args: Sequence[str],
    cwd: Optional[Path] = None,
    timeout: Optional[float] = 30.0,
) -> Tuple[int, str, str]:
    """Run a read-only Git command with optional locks disabled."""
    return run(["git", *args], cwd=cwd, env=git_env(), timeout=timeout)


def which(command: str) -> Optional[str]:
    return shutil.which(command)


# --------------------------------------------------------------------- locks


class LockError(ToolError):
    pass


class FileLock:
    """Exclusive advisory lock backed by ``O_CREAT|O_EXCL``.

    The lock file records the owning pid and a caller-supplied identity so a
    stale lock can be reported precisely instead of being silently stolen.
    """

    def __init__(self, path: Path, payload: Optional[Dict[str, Any]] = None):
        self.path = Path(path)
        self.payload = dict(payload or {})
        self._held = False

    def holder(self) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(self.payload)
        record.setdefault("pid", os.getpid())
        record.setdefault("acquired_at", utcnow())
        data = (json.dumps(record, indent=2) + "\n").encode("utf-8")
        try:
            handle = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise LockError("cannot create lock %s: %s" % (self.path, exc)) from exc
            existing = self.holder() or {}
            raise LockError(
                "task is locked by pid %s since %s (%s); "
                "stop it or remove the lock deliberately"
                % (
                    existing.get("pid", "?"),
                    existing.get("acquired_at", "?"),
                    self.path,
                )
            )
        try:
            os.write(handle, data)
        finally:
            os.close(handle)
        self._held = True

    def release(self) -> None:
        if not self._held:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        self._held = False

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


# --------------------------------------------------------- process identity


def process_identity(pid: int) -> Optional[Dict[str, str]]:
    """Return a stable identity for *pid*, or None when it is not running.

    ``lstart`` pins the exact start time so a recycled pid cannot be mistaken
    for the original process.
    """
    rc, out, _ = run(["ps", "-o", "lstart=,command=", "-p", str(int(pid))], timeout=10)
    if rc != 0:
        return None
    line = out.strip()
    if not line:
        return None
    # lstart is a fixed-width 5-field date, e.g. "Sun Sep 21 16:01:02 2026".
    parts = line.split(None, 5)
    if len(parts) < 6:
        return {"lstart": "", "command": line}
    return {"lstart": " ".join(parts[:5]), "command": parts[5]}


def process_matches(pid: int, recorded: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Check that *pid* is still the process described by *recorded*.

    Both the start time and the command must be recorded and non-empty. A
    partially populated record never matches, so an incomplete identity can
    never authorise signalling an unrelated pid.
    """
    if not recorded:
        return False, "no recorded process identity"
    expected_start = str(recorded.get("lstart") or "").strip()
    expected_command = str(recorded.get("command") or "").strip()
    if not expected_start or not expected_command:
        return False, (
            "recorded process identity is incomplete (lstart=%r, command=%r); "
            "refusing to match any pid" % (expected_start, expected_command)
        )
    current = process_identity(pid)
    if current is None:
        return False, "pid %s is not running" % pid
    current_start = str(current.get("lstart") or "").strip()
    current_command = str(current.get("command") or "").strip()
    if not current_start or not current_command:
        return False, (
            "cannot read a complete identity for pid %s; refusing to match" % pid
        )
    if current_start != expected_start:
        return False, (
            "pid %s start time differs (recorded %r, current %r); "
            "the pid was recycled" % (pid, expected_start, current_start)
        )
    if current_command != expected_command:
        return False, (
            "pid %s command differs (recorded %r, current %r)"
            % (pid, expected_command, current_command)
        )
    return True, "pid %s matches the recorded identity" % pid


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
