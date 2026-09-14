#!/usr/bin/env python3
"""Generate the dashboard's NAVRL_* parameter catalogue from the source, not from prose.

Writes `docs/status/data/parameters.json` describing every environment knob
this project reads: its current default, where it is declared, what the code comment says about it,
which launcher scripts set it, and -- joined from `data/experiments.json` -- whether it has ever
been the subject of a controlled ablation.

Why AST and not a regex. Five constructs in `navrl_task_config.py` alone defeat line-based
matching: calls spanning multiple lines (`NAVRL_DENSITY_THRESHOLD_SCHEDULE`), non-literal defaults
(`_env_float("NAVRL_DENSITY_THRESHOLD_START", success_threshold)`), post-call transforms that
change the effective default (`os.environ.get("NAVRL_ROBOT", "").strip() or "navrl_quad"` -- the
literal default is `""`, the effective one is `"navrl_quad"`), nested class scope that decides the
attribute path, and the same name read at two sites in one file (`NAVRL_LIDAR_RANGE`).

Why tokenize as well. The comments ARE the catalogue's value -- the paragraph above
`latency_ego_motion_fix` is the reasoning behind a 40 pp reversal -- and comments are not in the
AST. They are harvested as the contiguous run of comment lines directly above each assignment.

authoritative vs echo. A name read through a typed helper, or through `os.environ.get` at
module/class-body scope, configures behaviour. A name read by `os.environ.get` inside a function
body is almost always a receipt/manifest stamp re-reading its own configuration to write it into a
provenance record (see `NavRLTask.get_env_state`). Only the former is catalogued; the latter is
carried as a collapsed appendix so the page can state what it deliberately leaves out.

Run: PYTHONNOUSERSITE=1 python3 tools/generate_parameter_catalog.py
"""

import argparse
import ast
import json
import os
import re
import subprocess
import tokenize
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "docs/status/data/parameters.json"
EXPERIMENTS_JSON = ROOT / "docs/status/data/experiments.json"
LAUNCHER_DIR = ROOT / "aerial_gym/rl_training/rl_games"
SCAN_DIRS = ["aerial_gym", "tools"]

TYPED_HELPERS = {"_env_int": "int", "_env_float": "float", "_env_bool": "bool"}

# Project helpers that take the environment as an argument instead of touching os.environ, so the
# name is not in args[0]. Without this table the entire speed governor -- riskcap, the brake and
# reaction constants, every margin -- would be invisible to the catalogue, which would be the worst
# possible omission given how much of the control-risk work turns on those knobs.
#   helper name -> (arg index of the env-var name, arg index of the default, declared type)
INJECTED_HELPERS = {
    "_finite_float": (1, 2, "float"),
}
NAME_RE = re.compile(r"^(NAVRL_|AERIAL_|NUM_ENVS$|TASK$|FILE$)")

# Longest-prefix grouping. Order matters only for ties; `_group_for` picks the longest match.
GROUPS = [
    ("sensor", "센서 기하", ["NAVRL_LIDAR_", "NAVRL_CAM_", "NAVRL_APP_", "NAVRL_DETECTOR_"]),
    ("perception", "인지 파이프라인", [
        "NAVRL_PERCEPTION", "NAVRL_DETECTION_", "NAVRL_POSE_", "NAVRL_LATENCY_",
        "NAVRL_TARGET_MASK", "NAVRL_MAX_OBSTACLES", "NAVRL_OBSTACLE_", "NAVRL_CORRIDOR_",
        "NAVRL_VISION",
    ]),
    ("control", "제어 · 거버너", [
        "NAVRL_SPEED_GOVERNOR", "NAVRL_MAX_VELOCITY", "NAVRL_YAW_RATE", "NAVRL_ALT_HOLD",
        "NAVRL_MAX_TILT", "NAVRL_TILT_COMP",
    ]),
    ("curriculum", "커리큘럼", [
        "NAVRL_DENSITY_", "NAVRL_K_", "NAVRL_FOV_CURRICULUM", "NAVRL_TARGET_SPEED",
        "NAVRL_TARGET_PATTERN",
    ]),
    ("arena", "아레나 · 장애물", [
        "NAVRL_ARENA_", "NAVRL_BAR_", "NAVRL_PLACEMENT_", "NAVRL_NUM_BARS", "NAVRL_MAX_BARS",
        "NAVRL_FIXED_BARS",
    ]),
    ("training", "학습 · PPO", [
        "NAVRL_PPO_", "NAVRL_LEARNING_RATE", "NAVRL_ENTROPY_", "NAVRL_ACTION_", "NAVRL_LATENT_",
        "NAVRL_REFLECTION_", "NAVRL_LATERAL_", "NAVRL_EPISODE_LEN",
    ]),
    ("protocol", "평가 · 재현성", [
        "NAVRL_SEED", "NAVRL_EVAL_", "NAVRL_V2_", "NAVRL_GENERAL_", "NAVRL_ROBOT",
        "AERIAL_", "NAVRL_RUN_",
    ]),
]

UNIT_SUFFIX = [
    ("_MPS2", "m/s²"), ("_MPS", "m/s"), ("_M", "m"), ("_DEG", "°"), ("_S", "s"),
    ("_HZ", "Hz"), ("_EPOCHS", "epoch"), ("_EPS", "episode"), ("_N", "N"),
]
# Suffix inference is a heuristic; these are the names it gets wrong or cannot see.
UNIT_OVERRIDE = {
    "NAVRL_LIDAR_RANGE": "m",
    "NAVRL_MAX_VELOCITY": "m/s",
    "NAVRL_YAW_RATE_MAX": "rad/s",
    "NAVRL_ALT_HOLD_VMAX": "m/s",
    "NAVRL_DETECTION_LATENCY_S": "s",
    "NAVRL_ARENA_XY": "m",
    "NAVRL_ARENA_Z": "m",
    "NAVRL_SEED": None,
    "NAVRL_EPISODE_LEN_STEPS": "step",
}

# Knobs whose literal default is not what the program actually ends up with, because the call is
# wrapped in `.strip() or "x"` / `.lower()`. Recorded by hand because inferring it would mean
# evaluating arbitrary expressions.
EFFECTIVE_DEFAULT = {
    "NAVRL_ROBOT": "navrl_quad",
    "NAVRL_LATENCY_OBSTACLE_FIX": "off",
    "NAVRL_PLACEMENT_MODE": "random",
    "NAVRL_BAR_POOL": "bars",
    "AERIAL_GYM_SIM_NAME": "base_sim",
}

# Changing any of these changes the 898-D observation vector, so a checkpoint trained under one
# value cannot be evaluated under another. Surfaced as a warning badge, not as a tunable.
FROZEN_BY_CONTRACT = {
    "NAVRL_LIDAR_HBEAMS", "NAVRL_LIDAR_VBEAMS", "NAVRL_MAX_OBSTACLES", "NAVRL_CORRIDOR_TOKENS",
    "NAVRL_VISION", "NAVRL_PERCEPTION", "NAVRL_OBSTACLE_SECTORS", "NAVRL_HISTORY_STEPS",
}


def _group_for(name):
    best, best_len = "other", -1
    for gid, _label, prefixes in GROUPS:
        for p in prefixes:
            if name.startswith(p) and len(p) > best_len:
                best, best_len = gid, len(p)
    return best


def _unit_for(name):
    if name in UNIT_OVERRIDE:
        return UNIT_OVERRIDE[name]
    for suffix, unit in UNIT_SUFFIX:
        if name.endswith(suffix):
            return unit
    return None


def _literal(node):
    """(value, is_literal). ast.literal_eval on a non-literal raises, which is the signal we want."""
    try:
        return ast.literal_eval(node), True
    except (ValueError, SyntaxError, TypeError):
        return None, False


class Harvester(ast.NodeVisitor):
    """Collect env-var reads with their enclosing class path and function depth."""

    def __init__(self, rel_path, source):
        self.rel = rel_path
        self.source = source
        self.class_stack = []
        self.func_depth = 0
        self.sites = []          # dicts, one per read
        self.assign_line = {}    # id(Call node) -> (assign target path, assign lineno, expr)

    def _expr(self, node):
        if node is None:
            return None
        unparse = getattr(ast, "unparse", None)
        if unparse is not None:
            return unparse(node)
        # Isaac Gym's supported environment is Python 3.8, before ast.unparse existed.  Source
        # segments preserve the exact expression and are sufficient for this documentation field.
        return ast.get_source_segment(self.source, node)

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def _visit_func(self, node):
        self.func_depth += 1
        self.generic_visit(node)
        self.func_depth -= 1

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_Assign(self, node):
        # Remember the assignment wrapping each Call so we can recover `... or "navrl_quad"` and
        # the attribute name the knob lands on.
        target = None
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Call):
                self.assign_line[id(sub)] = (target, node.lineno, node.value)
        self.generic_visit(node)

    def visit_Call(self, node):
        fn = node.func
        helper = None
        name_i, default_i, declared_type = 0, 1, "str"
        # A read is authoritative when it CONFIGURES behaviour, as opposed to a receipt writer
        # re-reading its own settings to stamp them into a provenance record.
        forced_authoritative = False

        if isinstance(fn, ast.Name) and fn.id in TYPED_HELPERS:
            helper, declared_type, forced_authoritative = fn.id, TYPED_HELPERS[fn.id], True
        elif isinstance(fn, ast.Name) and fn.id in INJECTED_HELPERS:
            name_i, default_i, declared_type = INJECTED_HELPERS[fn.id]
            helper, forced_authoritative = fn.id, True
        elif isinstance(fn, ast.Attribute) and fn.attr == "get":
            base = fn.value
            if isinstance(base, ast.Attribute) and base.attr == "environ":
                helper = "os.environ.get"
            elif isinstance(base, ast.Name) and base.id == "environ":
                # Dependency-injected os.environ (SpeedGovernorConfig.from_environ). Always a
                # config reader by construction: the caller passes the environment in precisely so
                # the parse can be tested, which receipt stamps never do.
                helper, forced_authoritative = "environ.get", True

        if helper and len(node.args) > name_i:
            name, ok = _literal(node.args[name_i])
            if ok and isinstance(name, str) and NAME_RE.match(name):
                has_default = len(node.args) > default_i
                default, dlit = (_literal(node.args[default_i]) if has_default else (None, True))
                target, aline, aexpr = self.assign_line.get(id(node), (None, node.lineno, None))
                self.sites.append({
                    "name": name,
                    "file": self.rel,
                    "line": node.lineno,
                    "assign_line": aline,
                    "helper": helper,
                    "type": declared_type,
                    "default": default,
                    "default_literal": dlit,
                    "default_expr": (self._expr(node.args[default_i])
                                     if (has_default and not dlit) else None),
                    "scope": ".".join(self.class_stack) or "<module>",
                    "attr": ".".join(self.class_stack + [target]) if target else None,
                    "authoritative": forced_authoritative or self.func_depth == 0,
                    "assign_expr": self._expr(aexpr),
                })
        self.generic_visit(node)


def harvest_comments(path):
    """line number of a statement -> the comment block directly above it (and any trailing one)."""
    above, trailing = defaultdict(list), {}
    try:
        with open(path, "rb") as fh:
            toks = list(tokenize.tokenize(fh.readline))
    except (tokenize.TokenError, SyntaxError):
        return {}, {}

    run = []            # (lineno, text) of the current contiguous comment block
    last_code_line = 0
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            text = tok.string.lstrip("#").strip()
            if tok.start[1] == 0 or not _line_has_code_before(toks, tok):
                # own-line comment: extends or starts a block
                if run and tok.start[0] != run[-1][0] + 1:
                    run = []
                run.append((tok.start[0], text))
            else:
                trailing[tok.start[0]] = text
        elif tok.type in (tokenize.NL, tokenize.COMMENT):
            continue
        elif tok.type in (tokenize.NEWLINE,):
            last_code_line = tok.start[0]
            run = []
        elif tok.type not in (tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
                              tokenize.ENDMARKER, tokenize.NL):
            if run and tok.start[0] == run[-1][0] + 1:
                above[tok.start[0]] = [t for _, t in run]
                run = []
    return above, trailing


def _line_has_code_before(toks, target):
    for tok in toks:
        if tok.start[0] == target.start[0] and tok.start[1] < target.start[1] \
           and tok.type not in (tokenize.INDENT, tokenize.NL):
            return True
    return False


def scan_launchers():
    """name -> [{file, value}] harvested from `export NAME=...` in the campaign launchers."""
    out = defaultdict(list)
    if not LAUNCHER_DIR.is_dir():
        return out
    pat = re.compile(r'^\s*export\s+([A-Z][A-Z0-9_]*)=(.*)$')
    for sh in sorted(LAUNCHER_DIR.glob("*.sh")):
        for line in sh.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pat.match(line)
            if not m or not NAME_RE.match(m.group(1)):
                continue
            value = m.group(2).strip().strip('"').strip("'")
            out[m.group(1)].append({
                "file": str(sh.relative_to(ROOT)), "value": value[:120],
            })
    return out


def scan_launcher_only():
    """Knobs the shell launchers consume themselves, e.g. `${NAVRL_V2_DENSITIES:-130 160 190}`.

    These never reach a Python `os.environ.get`, so the AST scan cannot see them -- the launcher
    expands them into a per-cell value (NAVRL_V2_DENSITIES -> NAVRL_NUM_BARS for each cell). They
    are genuine experiment knobs all the same: two of the canonical campaigns are indexed by them.
    Leaving them out would make the parameters page quietly incomplete exactly where the sweep
    campaigns live.
    """
    out = defaultdict(list)
    if not LAUNCHER_DIR.is_dir():
        return out
    # ${NAME:-default}, ${NAME}, "$NAME"
    pat = re.compile(r'\$\{(NAVRL_[A-Z0-9_]+)(?::-([^}]*))?\}|\$(NAVRL_[A-Z0-9_]+)\b')
    for sh in sorted(LAUNCHER_DIR.glob("*.sh")):
        rel = str(sh.relative_to(ROOT))
        for i, line in enumerate(sh.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for m in pat.finditer(line):
                name = m.group(1) or m.group(3)
                default = (m.group(2) or "").strip()
                out[name].append({"file": rel, "line": i, "default": default})
    return out


def load_experiment_join():
    """knob name -> [experiment ids]. Tolerates the file not existing yet (Phase 3 before 4)."""
    join = defaultdict(list)
    if not EXPERIMENTS_JSON.exists():
        return join, {}
    data = json.loads(EXPERIMENTS_JSON.read_text(encoding="utf-8"))
    by_id = {}
    for exp in data.get("experiments", []):
        by_id[exp["id"]] = exp
        for knob in ([exp.get("knob")] if exp.get("knob") else []) + list(exp.get("knob_extra") or []):
            join[knob].append(exp["id"])
    return join, by_id


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def build():
    sites = []
    files_scanned = 0
    for d in SCAN_DIRS:
        for py in sorted((ROOT / d).rglob("*.py")):
            rel = str(py.relative_to(ROOT))
            if "__pycache__" in rel or "/source_snapshot/" in rel:
                continue
            try:
                source = py.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except SyntaxError:
                continue
            files_scanned += 1
            h = Harvester(rel, source)
            h.visit(tree)
            if not h.sites:
                continue
            above, trailing = harvest_comments(py)
            for s in h.sites:
                block = above.get(s["assign_line"], [])
                tail = trailing.get(s["assign_line"])
                if tail:
                    block = block + [tail]
                s["doc_lines"] = block
            sites.extend(h.sites)

    launchers = scan_launchers()
    launcher_only = scan_launcher_only()
    join, _ = load_experiment_join()

    by_name = defaultdict(list)
    for s in sites:
        by_name[s["name"]].append(s)

    params, echoes = [], []
    for name in sorted(by_name):
        group = [s for s in by_name[name] if s["authoritative"]]
        echo = [s for s in by_name[name] if not s["authoritative"]]
        if not group:
            echoes.append({
                "name": name,
                "sites": [{"file": s["file"], "line": s["line"], "scope": s["scope"]} for s in echo],
            })
            continue

        # Prefer the declaration that carries a comment, then the typed one, then the first.
        primary = max(group, key=lambda s: (len(s.get("doc_lines") or []),
                                            s["helper"] in TYPED_HELPERS,
                                            s["default"] is not None))
        doc_lines = primary.get("doc_lines") or []
        doc = "\n".join(doc_lines)
        first_sentence = re.split(r"(?<=[.。])\s|(?<=다)\.\s", doc.replace("\n", " "), maxsplit=1)[0]

        eff = EFFECTIVE_DEFAULT.get(name, primary["default"])
        exp_ids = join.get(name, [])
        params.append({
            "name": name,
            "group": _group_for(name),
            "type": primary["type"],
            "default": primary["default"],
            "default_expr": primary["default_expr"],
            "effective_default": eff,
            "effective_differs": eff != primary["default"],
            "unit": _unit_for(name),
            "attr": primary["attr"],
            "declarations": [
                {"file": s["file"], "line": s["line"], "scope": s["scope"],
                 "type": s["type"], "default": s["default"]}
                for s in sorted(group, key=lambda s: (s["file"], s["line"]))
            ],
            # Declared in more than one place. navrl_task_config.py warns in prose that these MUST
            # stay in sync; nothing enforces it, so the page flags them.
            "mirrors": len(group) > 1,
            "echoes": [{"file": s["file"], "line": s["line"], "scope": s["scope"]} for s in echo],
            "doc": doc,
            "doc_short": (first_sentence[:200] or None),
            "launchers": launchers.get(name, []),
            "experiments": exp_ids,
            "ablated": bool(exp_ids),
            # Set by launchers but never the subject of a controlled A/B. The honest third state
            # between "never touched" and "ablated".
            "swept_but_not_ablated": bool(launchers.get(name)) and not exp_ids,
            "frozen_by_contract": name in FROZEN_BY_CONTRACT,
            "origin": "python",
        })

    # Launcher-only knobs: consumed by the .sh itself, never by a Python os.environ.get.
    known = {p["name"] for p in params}
    for name in sorted(launcher_only):
        if name in known:
            continue
        sites = launcher_only[name]
        defaults = [s["default"] for s in sites if s["default"]]
        exp_ids = join.get(name, [])
        params.append({
            "name": name,
            "group": _group_for(name),
            "type": "str",
            "default": defaults[0] if defaults else None,
            "default_expr": None,
            "effective_default": defaults[0] if defaults else None,
            "effective_differs": False,
            "unit": _unit_for(name),
            "attr": None,
            "declarations": [{"file": s["file"], "line": s["line"], "scope": "shell launcher",
                              "type": "str", "default": s["default"] or None}
                             for s in sites[:12]],
            "mirrors": False,
            "echoes": [],
            "doc": "런처 스크립트가 직접 소비하는 값이다. 파이썬 코드는 이 이름을 읽지 않고, "
                   "런처가 셀마다 개별 knob으로 펼쳐서 전달한다.",
            "doc_short": "런처 전용 — 셀별 knob으로 펼쳐진다.",
            "launchers": launchers.get(name, []),
            "experiments": exp_ids,
            "ablated": bool(exp_ids),
            "swept_but_not_ablated": not exp_ids,
            "frozen_by_contract": False,
            "origin": "launcher",
        })

    params.sort(key=lambda p: p["name"])

    return {
        "schema": "motar.parameters/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "tools/generate_parameter_catalog.py",
        "source_commit": git_commit(),
        "counts": {
            "authoritative": len(params),
            "echo_only": len(echoes),
            "files_scanned": files_scanned,
            "launchers_scanned": len(list(LAUNCHER_DIR.glob("*.sh"))) if LAUNCHER_DIR.is_dir() else 0,
            "ablated": sum(1 for p in params if p["ablated"]),
            "mirrors": sum(1 for p in params if p["mirrors"]),
            "launcher_only": sum(1 for p in params if p["origin"] == "launcher"),
        },
        "groups": [{"id": gid, "label_ko": label} for gid, label, _ in GROUPS]
                  + [{"id": "other", "label_ko": "기타"}],
        "parameters": params,
        "echo_only": echoes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="print the summary without writing (for CI / spot checks)")
    args = ap.parse_args()

    data = build()
    c = data["counts"]
    print(f"authoritative {c['authoritative']} · echo-only {c['echo_only']} · "
          f"files {c['files_scanned']} · launchers {c['launchers_scanned']} · "
          f"ablated {c['ablated']} · mirrors {c['mirrors']}")
    if args.check:
        return 0

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {OUT_JSON.relative_to(ROOT)} ({OUT_JSON.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
