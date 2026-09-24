#!/usr/bin/env python3
"""Run one command, forward termination to its process tree, and record how it ended.

``k3ctl run start`` launches work in a detached process, so the CLI is not the
parent and cannot reap it. Without this wrapper:

* a finished run is indistinguishable from a killed one -- the pid is simply
  gone, and "the pid is not running" would read as a neutral ``exited``;
* cancelling would signal only this wrapper, leaving the real child orphaned;
  and
* the measurement harnesses spawn their own workers (``benchmark.py`` and
  ``precision/diagnose.py`` both run ``_worker.py`` subprocesses), so signalling
  only the direct child would leave those grandchildren running -- still holding
  a GPU -- after a cancel reported success.

Design notes that matter:

* The child is started with ``start_new_session=True`` so it leads its own
  process group, and the whole group is signalled on cancellation. The
  supervisor deliberately stays *outside* that group so it survives to reap the
  child and write a durable terminal status.
* The signal handler does **nothing but set a flag**. It must not wait, signal,
  or perform I/O: it runs on the main thread, and the main thread is inside
  ``Popen`` bookkeeping that holds an internal wait lock. A handler that called
  ``poll()``/``wait()`` there could block, and a handler that ran its own
  escalation sleep would delay reaping even a cooperative child by the full
  escalation timeout. All of that work belongs in the polling loop below.

Usage::

    python3 -m tools.k3.supervise <status-file> <argv...>
    python3 /path/to/supervise.py <status-file> <argv...>

stdout and stderr are inherited, so the caller's redirections keep working.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

#: Signals forwarded to the child's process group rather than killing the
#: supervisor outright.
FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)

#: Seconds to wait after forwarding a termination signal before escalating to
#: ``SIGKILL``. A child that ignores ``SIGTERM`` must not leave the run without
#: a durable status, and must not leave workers behind.
ESCALATE_AFTER = 20.0

#: How long to wait for a process group to drain after ``SIGKILL``.
GROUP_DRAIN_TIMEOUT = 5.0

#: Main-loop poll interval. Also the worst-case latency between a signal
#: arriving and the cancellation being acted on.
POLL_INTERVAL = 0.1

#: Set by the signal handler. Nothing else belongs in handler context.
_CANCEL: Dict[str, Any] = {"signum": None, "monotonic": None}


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: str, payload: Dict[str, Any]) -> None:
    """Write the status file atomically so a reader never sees a partial one."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=directory, prefix=".status-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=False)
            stream.write("\n")
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def _child_identity(pid: int) -> Optional[Dict[str, str]]:
    """Capture the child's start time and command, for later verification."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["ps", "-o", "lstart=,command=", "-p", str(int(pid))],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    line = completed.stdout.decode("utf-8", "replace").strip()
    if not line:
        return None
    parts = line.split(None, 5)
    if len(parts) < 6:
        return {"lstart": "", "command": line}
    return {"lstart": " ".join(parts[:5]), "command": parts[5]}


def _handler(signum, _frame):  # type: ignore[no-untyped-def]
    """Record the cancellation request and return immediately.

    Deliberately minimal: no waiting, no signalling, no I/O. See the module
    docstring for why any of those would be wrong here.
    """
    if _CANCEL["signum"] is None:
        _CANCEL["signum"] = int(signum)
        _CANCEL["monotonic"] = time.monotonic()


def _install_handlers() -> None:
    for signum in FORWARDED:
        try:
            signal.signal(signum, _handler)
        except (OSError, ValueError):
            continue


def _killpg(pgid: Optional[int], signum: int) -> bool:
    """Signal a whole process group. Returns True when the signal was sent."""
    if pgid is None:
        return False
    try:
        os.killpg(pgid, signum)
        return True
    except (OSError, ProcessLookupError):
        return False


def _group_alive(pgid: Optional[int]) -> bool:
    """True when any process remains in *pgid*."""
    if pgid is None:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _drain_group(pgid: Optional[int], kill: bool) -> Tuple[bool, str]:
    """Ensure no descendants remain in the child's process group.

    On cancellation the group is killed, because leftover workers would still
    hold a GPU. On a normal exit the group is only *inspected*: surviving
    descendants are reported rather than killed, so a leak is visible instead of
    being silently cleaned up.
    """
    if pgid is None:
        return True, "no process group recorded"
    if not _group_alive(pgid):
        return True, "process group is empty"
    if not kill:
        return False, (
            "descendants remain in the child's process group after a normal "
            "exit; they were reported, not killed"
        )
    _killpg(pgid, signal.SIGKILL)
    deadline = time.monotonic() + GROUP_DRAIN_TIMEOUT
    while time.monotonic() < deadline:
        if not _group_alive(pgid):
            return True, "process group drained after SIGKILL"
        time.sleep(POLL_INTERVAL)
    return False, "process group still has members after SIGKILL"


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        sys.stderr.write("usage: supervise.py <status-file> <command> [args...]\n")
        return 2

    status_path = argv[0]
    command = list(argv[1:])

    record: Dict[str, Any] = {
        "schema": "k3ctl/run-status/3",
        "state": "starting",
        "argv": command,
        "cwd": os.getcwd(),
        "supervisor_pid": os.getpid(),
        "child_pid": None,
        "child_pgid": None,
        "child_identity": None,
        "started_at": _utcnow(),
        "finished_at": None,
        "exit_code": None,
        "signal": None,
        "received_signal": None,
        "exited_normally": None,
        "cancelled": False,
        "escalated_to_sigkill": False,
        "descendants_cleared": None,
        "descendants_detail": None,
    }
    _write(status_path, record)

    # Install handlers before spawning, so a signal arriving immediately after
    # the fork is still recorded rather than killing the supervisor.
    _install_handlers()

    try:
        child = subprocess.Popen(  # noqa: S603 - argv list, no shell
            command,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        record.update(
            {
                "state": "failed-to-spawn",
                "finished_at": _utcnow(),
                "exit_code": 127,
                "error": str(exc),
                "exited_normally": False,
            }
        )
        _write(status_path, record)
        return 127
    except OSError as exc:
        record.update(
            {
                "state": "failed-to-spawn",
                "finished_at": _utcnow(),
                "exit_code": 126,
                "error": str(exc),
                "exited_normally": False,
            }
        )
        _write(status_path, record)
        return 126

    # The child leads its own process group because of start_new_session.
    try:
        pgid: Optional[int] = os.getpgid(child.pid)
    except (OSError, ProcessLookupError):
        pgid = child.pid

    record.update(
        {
            "state": "running",
            "child_pid": child.pid,
            "child_pgid": pgid,
            "child_identity": _child_identity(child.pid),
        }
    )
    _write(status_path, record)

    # Polling loop. All cancellation work happens here, never in the handler.
    cancel_sent = False
    escalated = False
    escalate_at: Optional[float] = None
    returncode: Optional[int] = None

    while True:
        returncode = child.poll()
        if returncode is not None:
            break

        signum = _CANCEL["signum"]
        if signum is not None and not cancel_sent:
            cancel_sent = True
            record.update(
                {
                    "state": "cancelling",
                    "received_signal": signum,
                    "cancel_requested_at": _utcnow(),
                    "cancelled": True,
                    "signalled_process_group": _killpg(pgid, signum),
                }
            )
            _write(status_path, record)
            escalate_at = time.monotonic() + ESCALATE_AFTER
        elif (
            cancel_sent
            and not escalated
            and escalate_at is not None
            and time.monotonic() >= escalate_at
        ):
            escalated = True
            record["escalated_to_sigkill"] = _killpg(pgid, signal.SIGKILL)
            _write(status_path, record)

        time.sleep(POLL_INTERVAL)

    # The child has been reaped by poll(). Now account for its descendants.
    cleared, detail = _drain_group(pgid, kill=cancel_sent)

    record["finished_at"] = _utcnow()
    record["descendants_cleared"] = cleared
    record["descendants_detail"] = detail

    if returncode is not None and returncode < 0:
        record.update(
            {
                "state": "cancelled" if cancel_sent else "signalled",
                "signal": -returncode,
                "exit_code": returncode,
                "exited_normally": False,
            }
        )
    else:
        record.update(
            {
                "state": "cancelled" if cancel_sent else "exited",
                "exit_code": returncode,
                "exited_normally": not cancel_sent,
            }
        )
    record["note"] = (
        "An exit code of 0 means the command ran to completion. It is not an "
        "evidence or gate result. A cancelled run is never a completion. When "
        "descendants_cleared is false, worker processes may have outlived the "
        "run."
    )
    _write(status_path, record)

    if returncode is None:
        return 1
    return returncode if returncode >= 0 else 128 + (-returncode)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
