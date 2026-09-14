"""Fail-closed guard against silently importing the wrong ``aerial_gym`` source tree.

Background: an editable install's ``.pth``/finder can hard-code an absolute path to one
worktree (the "primary" source tree). If a script is launched from a *different* worktree
(e.g. a git worktree checked out for an experiment) and that finder's meta_path entry wins
over the normal ``sys.path``-based resolution, Python will import ``aerial_gym`` from the
primary tree while everything else (config files, logs, source-manifest hashing) refers to
the worktree the operator actually intended to run. The receipt produced by such a run is
internally consistent but describes code that was never executed.

This module resolves where ``aerial_gym`` actually came from and, when the caller opts in
via ``NAVRL_REQUIRE_SOURCE_ROOT`` (or an explicit ``required_root`` argument), raises loudly
if that does not match the expected source root. It is deliberately dependency-free (stdlib
only) so it can be imported and unit-tested on a machine with no torch and no Isaac Gym.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Optional


def resolve_origin() -> dict:
    """Resolve where the ``aerial_gym`` package actually imports from.

    If the package is already imported, the live module object's ``__file__`` is the answer:
    it names the file that is actually executing. ``importlib.util.find_spec`` is only the
    fallback for the not-yet-imported case, because it answers the different question of
    where the package *would* resolve from now. This function never imports ``isaacgym``
    or ``torch`` itself. Note the guard requires an EXACT package-directory match; a parent
    or child directory is rejected.

    Returns:
        dict with keys:
            "module": always "aerial_gym"
            "origin": absolute resolved path to aerial_gym/__init__.py (str), or None if
                the spec has no file origin (e.g. a namespace package)
            "package_dir": absolute resolved path to the parent directory of that
                __init__.py (str), or None if "origin" is None
            "finder": "editable", "path", or "unknown" -- a best-effort label for which
                kind of meta_path/path finder produced the spec
    """
    loaded = sys.modules.get("aerial_gym")
    if loaded is not None:
        # Already imported: the only truthful answer is where the live module object came
        # from. find_spec would re-resolve and could name a different tree than the one whose
        # code is actually executing -- exactly the failure this guard exists to catch.
        spec = getattr(loaded, "__spec__", None)
        live_file = getattr(loaded, "__file__", None)
    else:
        spec = importlib.util.find_spec("aerial_gym")
        live_file = None

    origin: Optional[str] = None
    package_dir: Optional[str] = None
    finder = "unknown"

    origin_source = live_file or (spec.origin if spec is not None else None)
    if origin_source:
        origin_path = Path(origin_source).resolve()
        origin = str(origin_path)
        package_dir = str(origin_path.parent)

    if spec is not None:
        loader = spec.loader
        loader_module = type(loader).__module__ if loader is not None else ""
        loader_name = type(loader).__name__ if loader is not None else ""
        if "editable" in loader_module.lower() or "editable" in loader_name.lower():
            finder = "editable"
        elif loader_module.startswith("importlib.machinery") or loader_name in (
            "SourceFileLoader",
            "ExtensionFileLoader",
            "SourcelessFileLoader",
        ):
            finder = "path"

    return {
        "module": "aerial_gym",
        "origin": origin,
        "package_dir": package_dir,
        "finder": finder,
    }


def assert_runtime_origin(required_root: Optional[str] = None) -> dict:
    """Fail closed if ``aerial_gym`` did not import from the expected source root.

    Args:
        required_root: directory that should directly contain the ``aerial_gym`` package
            (i.e. ``<required_root>/aerial_gym`` is expected to be the package dir). If not
            given, defaults to ``os.environ.get("NAVRL_REQUIRE_SOURCE_ROOT", "").strip()``.
            When this resolves to an empty string, the guard is inert: it returns
            ``resolve_origin()`` merged with ``{"enforced": False}`` and never raises, so
            runs that do not opt in are unaffected.

    Returns:
        dict: the ``resolve_origin()`` fields plus:
            "enforced": bool -- whether a required root was checked
            "origin_sha256": str -- present only when enforced and the check passed; the
                sha256 hex digest of the resolved aerial_gym/__init__.py file's bytes.

    Raises:
        RuntimeError: when enforced and the actual package directory does not match
            ``<required_root>/aerial_gym``. The message includes both the required and the
            actual path, names the editable-install finder as the likely cause, and tells
            the caller to set ``PYTHONPATH=<required_root>``.
    """
    if required_root is None:
        required_root = os.environ.get("NAVRL_REQUIRE_SOURCE_ROOT", "").strip()

    info = resolve_origin()

    if not required_root:
        info["enforced"] = False
        return info

    expected_root = Path(required_root).resolve()
    expected_package_dir = (expected_root / "aerial_gym").resolve()

    actual_package_dir_str = info.get("package_dir")
    actual_package_dir = (
        Path(actual_package_dir_str).resolve() if actual_package_dir_str else None
    )

    matches = (
        actual_package_dir is not None and actual_package_dir == expected_package_dir
    )

    if not matches:
        raise RuntimeError(
            "aerial_gym import-origin mismatch: refusing to run.\n"
            f"  required (NAVRL_REQUIRE_SOURCE_ROOT): {expected_root}\n"
            f"  expected package dir:                 {expected_package_dir}\n"
            f"  actual origin:                        {info.get('origin')}\n"
            f"  actual package dir:                   {actual_package_dir}\n"
            "This usually means an editable install's meta_path finder "
            "(e.g. a __editable__.<pkg>-*.pth / *_finder.py generated by pip install -e) "
            "hard-codes a different worktree's absolute path and is shadowing the "
            "aerial_gym package you intended to run from PYTHONPATH/sys.path.\n"
            f"Fix: re-run with PYTHONPATH={expected_root} (prepended so it is searched "
            "before the editable-install finder), or unset NAVRL_REQUIRE_SOURCE_ROOT if "
            "this run is intentionally using a different source tree."
        )

    origin_path = Path(info["origin"]).resolve()
    origin_bytes = origin_path.read_bytes()
    info["origin_sha256"] = hashlib.sha256(origin_bytes).hexdigest()
    info["enforced"] = True
    return info
