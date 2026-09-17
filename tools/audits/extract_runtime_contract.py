#!/usr/bin/env python3
"""Extract the current simulator contract statically and diff it against a checkpoint.

Static on purpose. Importing the task config pulls in isaacgym and opens a CUDA
context, which is the wrong dependency for an audit artifact that must be
reproducible on a machine with no GPU. Everything here is read from source text
and from the checkpoint's own ``env_state``.

Usage:
  extract_runtime_contract.py --out docs/audits/current_runtime_contract_2026-09-17.json
  extract_runtime_contract.py --checkpoint-env-state STATE.json --diff-out DIFF.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TASK_CONFIG = ROOT / "aerial_gym/config/task_config/navrl_task_config.py"
NAVRL_TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
TARGET_MOTION = ROOT / "aerial_gym/task/navrl_task/target_motion.py"
ROUTE_PLANNER = ROOT / "aerial_gym/task/navrl_task/target_route_planner.py"

# cfg key -> (source file, module-level constant or config attribute)
# Only keys whose current value can be established from source text without
# instantiating the task are listed. Anything else is reported NOT_ATTESTABLE
# rather than guessed.
ENVVAR_CALL = re.compile(
    r'_env_(float|int|bool|str)\(\s*"(?P<env>[A-Z0-9_]+)"\s*,\s*(?P<default>[^)]+?)\s*\)'
)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_assignments(path: Path) -> dict:
    """Module- and class-level simple assignments, with their literal or env default."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}

    def literal(node):
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")
            elif isinstance(child, ast.Assign) and len(child.targets) == 1:
                target = child.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                name = prefix + target.id
                value = literal(child.value)
                source = ast.get_source_segment(path.read_text(encoding="utf-8"), child.value) or ""
                env = ENVVAR_CALL.search(source)
                found[name] = {
                    "value": value,
                    "env_var": env.group("env") if env else None,
                    "env_default": env.group("default").strip() if env else None,
                    "expression": source.strip()[:160],
                    "line": child.lineno,
                }
            elif isinstance(child, (ast.If, ast.Try)):
                visit(child, prefix)

    visit(tree)
    return found


def module_constants(path: Path, names) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    out[target.id] = {"value": ast.literal_eval(node.value), "line": node.lineno}
                except Exception:
                    out[target.id] = {
                        "value": None,
                        "line": node.lineno,
                        "expression": ast.get_source_segment(
                            path.read_text(encoding="utf-8"), node.value
                        ),
                    }
    return out


def env_state_emitters(path: Path) -> dict:
    """Every ``cfg_*`` key get_env_state() writes, with the expression behind it."""
    text = path.read_text(encoding="utf-8")
    out = {}
    for match in re.finditer(r'"(cfg_[a-z0-9_]+)"\s*:\s*([^,\n]+)', text):
        out.setdefault(match.group(1), []).append(
            {"expression": match.group(2).strip()[:140],
             "line": text[: match.start()].count("\n") + 1}
        )
    for match in re.finditer(r'\(\s*"(cfg_[a-z0-9_]+)"\s*,\s*([^,\n]+)', text):
        out.setdefault(match.group(1), []).append(
            {"expression": match.group(2).strip()[:140],
             "line": text[: match.start()].count("\n") + 1}
        )
    return out


def declared_key_to_env(path: Path) -> dict:
    """The code's OWN declared (cfg_key, value, ENV_VAR) attestation triples.

    navrl_task.py builds these tuples so a restored checkpoint can be compared
    against the running environment. Reading them is strictly better than
    guessing a mapping from the key name: it is the mapping the simulator
    itself uses.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 3:
            continue
        key_node, value_node, env_node = node.elts
        if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)
                and key_node.value.startswith("cfg_")):
            continue
        if not (isinstance(env_node, ast.Constant) and isinstance(env_node.value, str)):
            continue
        env = env_node.value.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", env):
            continue  # a prose label, not an environment variable
        out[key_node.value] = {
            "env_var": env,
            "expression": (ast.get_source_segment(text, value_node) or "")[:140],
            "line": node.lineno,
        }
    return out


ENV_DEFAULT = re.compile(
    r'(?:_env_(?:float|int|bool|str)|os\.environ\.get)\(\s*"(?P<env>[A-Z][A-Z0-9_]*)"\s*,\s*'
    r'(?P<default>"[^"]*"|\'[^\']*\'|[-+0-9eE.]+|True|False)'
)


def env_defaults(paths) -> dict:
    """ENV_VAR -> literal default, wherever the default is written in source."""
    out = {}
    for path in paths:
        for match in ENV_DEFAULT.finditer(path.read_text(encoding="utf-8")):
            env, raw = match.group("env"), match.group("default")
            try:
                value = ast.literal_eval(raw)
            except Exception:
                value = raw
            out.setdefault(env, {"value": value, "source": str(path.relative_to(ROOT))})
    return out


def drift_guard_keys(path: Path):
    """Keys checked by the 'warn, never override' config-drift guard."""
    text = path.read_text(encoding="utf-8")
    anchor = text.find("Loud config-drift guard")
    if anchor < 0:
        return {"found": False, "keys": [], "fails_closed": None}
    block = text[anchor: anchor + 4000]
    keys = re.findall(r'\(\s*"(cfg_[a-z0-9_]+)"', block)
    return {
        "found": True,
        "keys": sorted(set(keys)),
        "fails_closed": False,
        "behaviour": "logger.warning; execution continues",
        "line": text[:anchor].count("\n") + 1,
    }


def build_snapshot() -> dict:
    config = config_assignments(TASK_CONFIG)
    motion = module_constants(
        TARGET_MOTION,
        {
            "HEADING_VALID_SPEED_MPS", "HEADING_VALID_SPEED_KEY",
            "HEADING_VALID_SPEED_ATTESTED", "HEADING_VALID_SPEED_ASSUMED",
            "TURN_ANGLES_DEG", "BOUNDED_TURN_ANGLES_DEG", "HEADING_CONTINUITY_RAD",
            "TARGET_BEHAVIOR_LEVELS",
        },
    )
    route = module_constants(
        ROUTE_PLANNER,
        {"TARGET_ROUTE_MODE_OFF", "TARGET_ROUTE_MODE_GLOBAL_ASTAR", "TARGET_ROUTE_MODES"},
    )
    interesting = {
        name: entry for name, entry in config.items()
        if any(token in name.lower() for token in (
            "velocity", "yaw", "tilt", "success", "radius", "speed", "target",
            "lidar", "fov", "obstacle", "episode", "dt", "physics", "arena",
            "placement", "bar", "governor", "detector", "noise", "dropout",
            "goal", "margin", "action", "reward",
        ))
    }
    return {
        "kind": "navrl_current_runtime_contract_snapshot",
        "method": "static source extraction (no task instantiation, no CUDA context)",
        "git_commit": git("rev-parse", "HEAD"),
        "git_tree_clean": git("status", "--porcelain") == "",
        "sources": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "lines": len(
                path.read_text(encoding="utf-8").splitlines())}
            for path in (TASK_CONFIG, NAVRL_TASK, TARGET_MOTION, ROUTE_PLANNER)
        },
        "task_config_defaults": interesting,
        "target_motion_constants": motion,
        "route_planner_constants": route,
        "env_state_emitted_keys": sorted(env_state_emitters(NAVRL_TASK)),
        "declared_key_to_env": declared_key_to_env(NAVRL_TASK),
        "env_var_defaults": env_defaults(
            [TASK_CONFIG, NAVRL_TASK, TARGET_MOTION, ROUTE_PLANNER]
        ),
        "config_drift_guard": drift_guard_keys(NAVRL_TASK),
    }


def classify(key, checkpoint_value, snapshot):
    """Classify one checkpoint key against the current runtime source."""
    config = snapshot["task_config_defaults"]
    declared = snapshot["declared_key_to_env"]
    defaults = snapshot["env_var_defaults"]

    # Prefer the simulator's own declared key -> env-var attestation mapping.
    entry, current, env_var, source = None, None, None, None
    if key in declared and declared[key]["env_var"] in defaults:
        env_var = declared[key]["env_var"]
        current = defaults[env_var]["value"]
        source = f"{defaults[env_var]['source']} (declared at navrl_task.py:{declared[key]['line']})"
    else:
        attribute = key[len("cfg_"):] if key.startswith("cfg_") else key
        candidates = [name for name in config if name.split(".")[-1] == attribute]
        if not candidates:
            return {
                "status": "NOT_ATTESTABLE",
                "checkpoint": checkpoint_value,
                "reason": (
                    "no declared env-var triple and no single source-level default "
                    "maps to this key; its runtime value is computed or nested"
                ),
                "declared_env_var": declared.get(key, {}).get("env_var"),
            }
        entry = config[candidates[0]]
        current = entry["value"]
        env_var = entry["env_var"]
        source = f"{TASK_CONFIG.relative_to(ROOT)}:{entry['line']}"
        if env_var:
            try:
                current = ast.literal_eval(entry["env_default"])
            except Exception:
                current = entry["env_default"]
    # An empty-string default is a sentinel meaning "unset, resolve elsewhere",
    # not a value. Calling that a changed default would inflate the diff with
    # differences that do not exist.
    if isinstance(current, str) and current.strip() == "":
        return {
            "status": "NOT_ATTESTABLE",
            "checkpoint": checkpoint_value,
            "env_var": env_var,
            "source": source,
            "reason": "env-var default is an empty sentinel; effective value is computed elsewhere",
        }
    same = False
    try:
        same = abs(float(current) - float(checkpoint_value)) <= 1e-9
    except (TypeError, ValueError):
        same = str(current).strip() == str(checkpoint_value).strip()
    if same:
        status = "IDENTICAL" if not env_var else "DEFAULT_PRESERVED"
    else:
        status = "CHANGED_BUT_OPT_IN" if env_var else "CHANGED_FOR_CURRENT_DEFAULT"
    return {
        "status": status,
        "checkpoint": checkpoint_value,
        "current_default": current,
        "env_var": env_var,
        "source": source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--checkpoint-env-state")
    parser.add_argument("--diff-out")
    args = parser.parse_args()

    snapshot = build_snapshot()
    if args.out:
        Path(args.out).write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.out}")

    if args.checkpoint_env_state:
        env = json.loads(Path(args.checkpoint_env_state).read_text())
        rows, counts = {}, {}
        for key, value in sorted(env.items()):
            if not key.startswith("cfg_"):
                continue
            row = classify(key, value, snapshot)
            rows[key] = row
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        guard = snapshot["config_drift_guard"]
        diff = {
            "kind": "navrl_training_contract_vs_current_runtime",
            "git_commit": snapshot["git_commit"],
            "checkpoint_cfg_keys": len(rows),
            "counts": counts,
            "guarded_keys": guard["keys"],
            "guard_fails_closed": guard["fails_closed"],
            "rows": rows,
        }
        if args.diff_out:
            Path(args.diff_out).write_text(json.dumps(diff, indent=1, sort_keys=True) + "\n")
            print(f"wrote {args.diff_out}")
        for status in sorted(counts):
            print(f"  {status:32s} {counts[status]}")
        changed = [k for k, r in rows.items()
                   if r["status"].startswith("CHANGED")]
        if changed:
            print("\nchanged keys:")
            for key in changed:
                row = rows[key]
                guarded = "GUARDED(warn-only)" if key in guard["keys"] else "UNGUARDED"
                print(f"  {key:34s} ckpt={row['checkpoint']!s:<22s} "
                      f"now={row['current_default']!s:<12s} {guarded}")


if __name__ == "__main__":
    main()
