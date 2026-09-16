"""Skip, rather than error, when a check needs research history this checkout does not have.

Several evidence tests verify a claim against a specific commit of the research repository: that a
historical blob is unchanged, that a receipt matches the tree it recorded. A public release snapshot
deliberately carries content without history, so those commits are absent there. The check is not
failing in that case - it cannot run, and saying so is different from saying the evidence is wrong.

Inside the research repository the same guard is inert: the commits exist and the checks run.
"""
from pathlib import Path
import subprocess
import unittest


def git_output(arguments, cwd):
    try:
        return subprocess.check_output(["git", *arguments], cwd=str(cwd), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def inside_git_work_tree(root):
    return git_output(["rev-parse", "--is-inside-work-tree"], root) == "true"


def has_commit(root, commit):
    return git_output(["cat-file", "-t", commit], root) == "commit"


def commit_count(root):
    value = git_output(["rev-list", "--count", "HEAD"], root)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def is_public_snapshot(root):
    """Recognize the explicit release marker even after snapshot follow-up commits exist."""
    marker = Path(root) / "RELEASE_PROVENANCE.md"
    if not marker.is_file():
        return False
    try:
        text = marker.read_text(encoding="utf-8")
    except OSError:
        return False
    return (
        "This directory is a **snapshot**, not a clone" in text
        and "none of its history" in text
        and "**None.** The research repository's history was not rewritten" in text
    )


def require_research_history(root, *commits):
    """Raise SkipTest when this checkout cannot see the history the check is about.

    The initial release had one commit, but ordinary public maintenance adds descendants without
    importing research ancestors. ``RELEASE_PROVENANCE.md`` is therefore the durable snapshot
    marker; commit count is retained only for the original one-commit form. Named commits are still
    checked in research clones, which catches a shallow clone that has some history but not this one.
    """
    if not inside_git_work_tree(root):
        raise unittest.SkipTest(
            "not inside a Git work tree: this check verifies research history, which a release "
            "snapshot does not carry")
    if is_public_snapshot(root):
        raise unittest.SkipTest(
            "this is the clean public content snapshot: research ancestor objects are deliberately "
            "absent even though public follow-up commits may exist")
    if commit_count(root) < 2:
        raise unittest.SkipTest(
            "this checkout has a single commit and no ancestors: it is a content snapshot, and the "
            "research history this check verifies against is not present")
    missing = [commit for commit in commits if not has_commit(root, commit)]
    if missing:
        raise unittest.SkipTest(
            "research history not present in this checkout (missing %s); a release snapshot carries "
            "content without history" % ", ".join(sorted(missing)[:3]))


def require_local_evidence(*paths):
    """Skip when evidence this check needs is not in this checkout.

    Some result bundles are deliberately excluded from the repository - `.gitignore` calls them
    "immutable local receipts with raw traces" - and some training checkpoints were never tracked.
    A check that reads them can only pass on the machine that produced them. In any other clone it
    should say that plainly rather than erroring, and the release manifest names the same gap.
    """
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise unittest.SkipTest(
            "evidence not present in this checkout (%s); it is a local bundle or an untracked "
            "artifact, not part of the repository" % ", ".join(sorted(missing)[:2]))
