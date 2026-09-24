#!/usr/bin/env python3
"""Read-only Git queries.

Every helper runs with ``GIT_OPTIONAL_LOCKS=0`` and never mutates a
repository, an index, or a ref.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from . import util

GITLINK_MODE = "160000"


def toplevel(path: Path) -> Optional[str]:
    """Return the working-tree root Git reports for *path*."""
    rc, out, _ = util.git(["rev-parse", "--show-toplevel"], cwd=path)
    if rc != 0:
        return None
    return out.strip() or None


def is_own_root(path: Path) -> bool:
    """True only when *path* is itself the root of a Git working tree.

    An empty or uninitialized submodule directory otherwise resolves to the
    parent repository, which would make a missing dependency look present.
    """
    top = toplevel(path)
    if top is None:
        return False
    try:
        return Path(top).resolve() == Path(path).resolve()
    except OSError:
        return False


def git_dir(path: Path) -> Optional[str]:
    rc, out, _ = util.git(["rev-parse", "--absolute-git-dir"], cwd=path)
    if rc != 0:
        return None
    return out.strip() or None


def head_exists(path: Path) -> bool:
    rc, _, _ = util.git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=path)
    return rc == 0


def head_commit(path: Path) -> Optional[str]:
    rc, out, _ = util.git(["rev-parse", "HEAD"], cwd=path)
    if rc != 0:
        return None
    return out.strip() or None


def current_branch(path: Path) -> Optional[str]:
    rc, out, _ = util.git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=path)
    if rc != 0:
        return None
    return out.strip() or None


def has_commit(path: Path, sha: str) -> bool:
    """True when *sha* resolves to a commit object inside *path*."""
    if not sha:
        return False
    rc, _, _ = util.git(["cat-file", "-e", "%s^{commit}" % sha], cwd=path)
    return rc == 0


def porcelain(path: Path, untracked: bool = True) -> Optional[List[str]]:
    """Return ``git status --porcelain`` lines, or None when unavailable."""
    args = ["status", "--porcelain"]
    args.append("--untracked-files=all" if untracked else "--untracked-files=no")
    rc, out, _ = util.git(args, cwd=path)
    if rc != 0:
        return None
    return [line for line in out.splitlines() if line.strip()]


def is_dirty(path: Path, untracked: bool = True) -> Optional[bool]:
    lines = porcelain(path, untracked=untracked)
    if lines is None:
        return None
    return bool(lines)


def is_sparse(path: Path) -> bool:
    rc, out, _ = util.git(["config", "--get", "core.sparseCheckout"], cwd=path)
    return rc == 0 and out.strip().lower() == "true"


def tracked_file_count(path: Path) -> Optional[int]:
    rc, out, _ = util.git(["ls-files"], cwd=path)
    if rc != 0:
        return None
    return len([line for line in out.splitlines() if line.strip()])


def present_file_count(path: Path) -> Optional[int]:
    """Count tracked files actually materialised in the working tree."""
    rc, out, _ = util.git(["ls-files", "--", "."], cwd=path)
    if rc != 0:
        return None
    present = 0
    root = Path(path)
    for line in out.splitlines():
        name = line.strip()
        if name and (root / name).exists():
            present += 1
    return present


def recorded_gitlinks(root: Path) -> Dict[str, str]:
    """Map submodule path -> gitlink SHA recorded in the index."""
    rc, out, _ = util.git(["ls-files", "--stage"], cwd=root)
    if rc != 0:
        return {}
    links: Dict[str, str] = {}
    for line in out.splitlines():
        if not line.startswith(GITLINK_MODE + " "):
            continue
        meta, _, name = line.partition("\t")
        fields = meta.split()
        if len(fields) >= 2:
            links[name.strip()] = fields[1]
    return links


def staged_listing(root: Path) -> Optional[str]:
    rc, out, _ = util.git(["ls-files", "--stage"], cwd=root)
    if rc != 0:
        return None
    return out


def staged_paths(root: Path) -> List[str]:
    rc, out, _ = util.git(["ls-files"], cwd=root)
    if rc != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]


def untracked_paths(root: Path) -> List[str]:
    rc, out, _ = util.git(
        ["ls-files", "--others", "--exclude-standard"], cwd=root
    )
    if rc != 0:
        return []
    return [line for line in out.splitlines() if line.strip()]


def is_ignored(root: Path, relative: str) -> bool:
    rc, _, _ = util.git(["check-ignore", "-q", "--", relative], cwd=root)
    return rc == 0


def is_tracked(root: Path, relative: str) -> bool:
    rc, _, _ = util.git(
        ["ls-files", "--error-unmatch", "--", relative], cwd=root
    )
    return rc == 0


def diff_staged_vs_worktree(
    root: Path, paths: Optional[List[str]] = None
) -> str:
    """Diff the working tree against the index, including binary contents.

    This is the correct comparison for a repository with an unborn HEAD: there
    is no commit to diff against, but the staged content is authoritative.

    A Git failure raises. Returning an empty string on error would produce an
    apparently successful but empty bundle, which is worse than no bundle.
    """
    args = ["--no-pager", "diff", "--no-color", "--no-ext-diff", "--binary"]
    if paths:
        args.append("--")
        args.extend(paths)
    rc, out, err = util.git(args, cwd=root, timeout=300)
    if rc != 0:
        raise util.ToolError(
            "git diff failed in %s (rc=%d): %s\n"
            "Refusing to emit an empty bundle that would look like 'no changes'."
            % (root, rc, err.strip() or "no stderr")
        )
    return out


#: Entry kinds that ``git add`` can represent. Anything else (FIFO, socket,
#: device) is reported as skipped rather than crashing the export.
def _is_addable(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        if path.is_file():
            return True
    except OSError:
        return False
    return False


def diff_including_untracked(
    root: Path, untracked: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Diff the working tree against the index, *including* untracked files.

    ``git diff`` alone is not sufficient for delivery from this repository. The
    index holds the user's staged baseline, and new work arrives as **untracked**
    files -- so a plain worktree-vs-index diff lists modifications to existing
    files but omits every new file entirely. Applying such a patch would
    reconstruct none of the new code.

    The fix is to diff against a *temporary copy* of the index with the untracked
    paths registered via ``git add --intent-to-add``. That makes Git itself emit
    proper ``/dev/null -> b/path`` additions with full content, correct modes,
    symlinks as symlinks, and binary hunks where needed.

    The real index is never touched: ``GIT_INDEX_FILE`` points at a copy in a
    temporary directory, and the copy is discarded afterwards.

    Returns the diff text plus an inventory describing exactly what was
    included and what was skipped.
    """
    import shutil
    import tempfile

    root = Path(root)
    if untracked is None:
        untracked = untracked_paths_z(root)

    addable: List[str] = []
    skipped: List[Dict[str, str]] = []
    for relative in untracked:
        candidate = root / relative
        if _is_addable(candidate):
            addable.append(relative)
        else:
            skipped.append(
                {
                    "path": relative,
                    "reason": "not a regular file or symlink; Git cannot "
                    "represent it in a patch",
                }
            )

    real_index = root / ".git" / "index"
    args_base = ["--no-pager", "-c", "core.quotePath=false"]

    with tempfile.TemporaryDirectory() as temp:
        temp_index = Path(temp) / "index"
        index_copied = False
        if real_index.is_file():
            # copy2, not copyfile: the index's own mtime must be preserved.
            #
            # Git re-reads a file's content only when its index entry looks
            # "racily clean" -- entry mtime >= the index file's own mtime.
            # copyfile stamps the copy with the current time, which is newer than
            # every entry, so Git trusts the cached stat data and skips the
            # content check. A tracked file edited within the same second without
            # changing size is then reported as unchanged and omitted from the
            # patch entirely, silently losing that work from the bundle.
            shutil.copy2(str(real_index), str(temp_index))
            index_copied = True

        env = util.git_env({"GIT_INDEX_FILE": str(temp_index)})

        if addable:
            # NUL-separated pathspecs, so paths containing spaces, newlines or
            # non-ASCII bytes are handled exactly as Git reported them.
            payload = "\0".join(addable)
            rc, _, err = util.run(
                [
                    "git",
                    *args_base,
                    "add",
                    "--intent-to-add",
                    "--pathspec-from-file=-",
                    "--pathspec-file-nul",
                ],
                cwd=root,
                env=env,
                timeout=300,
                stdin_data=payload.encode("utf-8"),
            )
            if rc != 0:
                raise util.ToolError(
                    "cannot register untracked files for the export diff "
                    "(rc=%d): %s\n"
                    "The real index was not modified." % (rc, err.strip())
                )

        rc, out, err = util.run(
            [
                "git",
                *args_base,
                "diff",
                "--no-color",
                "--no-ext-diff",
                "--binary",
            ],
            cwd=root,
            env=env,
            timeout=600,
        )
        if rc != 0:
            raise util.ToolError(
                "git diff failed while building the export patch (rc=%d): %s\n"
                "Refusing to emit an empty bundle that would look like "
                "'no changes'." % (rc, err.strip() or "no stderr")
            )

    # Confirm the real index is byte-identical: the whole point of the temp copy.
    return {
        "diff": out,
        "included_untracked": sorted(addable),
        "skipped_untracked": skipped,
        "index_copied": index_copied,
        "method": "temp-index intent-to-add",
    }


def untracked_paths_z(root: Path) -> List[str]:
    """Untracked, non-ignored paths read NUL-separated.

    ``-z`` is required: newline-separated output mangles paths containing
    newlines or non-ASCII bytes, which Git would otherwise quote.
    """
    rc, out, err = util.git(
        ["ls-files", "--others", "--exclude-standard", "-z"], cwd=root, timeout=120
    )
    if rc != 0:
        raise util.ToolError(
            "git ls-files (untracked) failed in %s (rc=%d): %s"
            % (root, rc, err.strip())
        )
    return [item for item in out.split("\0") if item]


def staged_entries(root: Path) -> List[Dict[str, str]]:
    """Staged index entries read NUL-separated: mode, object, stage, path.

    In a repository with an unborn HEAD this is the authoritative base
    identity: there is no commit to name, but every staged blob has an object
    id.
    """
    rc, out, err = util.git(["ls-files", "--stage", "-z"], cwd=root, timeout=120)
    if rc != 0:
        raise util.ToolError(
            "git ls-files --stage failed in %s (rc=%d): %s" % (root, rc, err.strip())
        )
    entries: List[Dict[str, str]] = []
    for record in out.split("\0"):
        if not record:
            continue
        meta, _, name = record.partition("\t")
        fields = meta.split()
        if len(fields) < 3 or not name:
            continue
        entries.append(
            {
                "mode": fields[0],
                "object": fields[1],
                "stage": fields[2],
                "path": name,
            }
        )
    return entries


def archive_commit(source: Path, sha: str, destination: Path) -> None:
    """Materialise *sha* from *source* into *destination* via ``git archive``.

    ``git archive`` writes only to the destination directory; the source
    repository, its index and its checkout are untouched.
    """
    import tarfile
    import tempfile

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        bundle = Path(temp) / "archive.tar"
        rc, _, err = util.run(
            ["git", "archive", "--format=tar", "-o", str(bundle), sha],
            cwd=source,
            env=util.git_env(),
            timeout=600,
        )
        if rc != 0:
            raise util.ToolError(
                "git archive failed for %s in %s: %s" % (sha, source, err.strip())
            )
        with tarfile.open(bundle) as handle:
            _safe_extract(handle, destination)


def _link_escapes(member_name: str, link_target: str) -> bool:
    """True when a link target would resolve outside the archive root.

    Real source trees legitimately contain relative symlinks (SGLang tracks
    several). Those are preserved. Absolute targets, or relative targets that
    climb above the archive root, are refused because they would let an archive
    reach host paths.
    """
    import posixpath

    if posixpath.isabs(link_target):
        return True
    base = posixpath.dirname(member_name)
    resolved = posixpath.normpath(posixpath.join(base, link_target))
    return resolved == ".." or resolved.startswith("../")


def _safe_extract(handle, destination: Path) -> None:
    """Extract a tar archive, refusing path escapes and special files.

    In-root symlinks and hardlinks are preserved so the extracted tree matches
    the commit exactly; escaping or absolute links, devices and FIFOs are
    refused.
    """
    import posixpath

    root = destination.resolve()
    for member in handle.getmembers():
        normalised = posixpath.normpath(member.name)
        if posixpath.isabs(normalised) or normalised == ".." or normalised.startswith("../"):
            raise util.ToolError(
                "archive member escapes the destination: %s" % member.name
            )
        if member.isdev() or member.isfifo():
            raise util.ToolError(
                "archive member is a special file, refusing: %s" % member.name
            )
        if (member.issym() or member.islnk()) and _link_escapes(
            normalised, member.linkname
        ):
            raise util.ToolError(
                "archive member %s links outside the archive root (%s), refusing"
                % (member.name, member.linkname)
            )
    handle.extractall(str(destination))
