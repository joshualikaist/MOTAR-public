#!/usr/bin/env python3
"""Generate the machine-readable platform + sensor spec from URDFs and robot configs.

Writes `docs/status/data/platform.json`: airframe mass,
inertia, thrust, collision geometry and the derived flight envelope for every NavRL robot, plus the
sensor contract, plus the measured flight-envelope gate if it has been run.

Everything here is DERIVED. A hand-typed spec table on a research dashboard is a liability: the
2026-08-13 audit found the legacy URDF had been carrying a 0.150 m plate's inertia and Aerial Gym's
stock 0.25 kg for three weeks while the collision box said 0.28 m and the motor arms said 0.13 m,
precisely because no reader ever recomputed those numbers against each other. This script does the
recomputation on every run, so the page cannot drift from the simulator.

The derived quantities use the same formulas as `tests/test_navrl_ref5in_platform.py`; that test is
the authority and this is the presentation layer.

Run: PYTHONNOUSERSITE=1 python3 tools/generate_platform_spec.py
"""

import json
import math
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "docs/status/data/platform.json"
URDF_DIR = ROOT / "resources/robots/quad"
ENVELOPE = ROOT / "results/navrl_ref_platform_verification/flight_envelope.json"

G = 9.81

# The two robots the NavRL task can be built with, and the perception payload they would have to
# carry for real. See docs/archive/sim_vs_hardware_gap_2026-08.md.
ROBOTS = [
    {
        "key": "navrl_quad",
        "label": "legacy · navrl_quad",
        "config": "aerial_gym.config.robot_config.navrl_quad_config",
        "cls": "NavRLQuadCfg",
        "note": "모든 기존 체크포인트가 학습된 기체. Aerial Gym 기본 질량·추력을 상속했다.",
    },
    {
        "key": "navrl_ref5in_quad",
        "label": "ref5in · navrl_ref5in_quad",
        "config": "aerial_gym.config.robot_config.navrl_ref5in_quad_config",
        "cls": "NavRLRef5inQuadCfg",
        "note": "내부 정합성을 맞춘 5인치급 시뮬레이션 후보. 실기 검증된 기체가 아니다.",
    },
    {
        "key": "navrl_ref5in_v2_quad",
        "label": "fresh · navrl_ref5in_v2_quad",
        "config": "aerial_gym.config.robot_config.navrl_ref5in_v2_quad_config",
        "cls": "NavRLRef5inV2QuadCfg",
        "note": "prop-tip span을 바깥쪽으로 반올림한 fresh-only geometry correction.",
    },
]

PAYLOAD = [
    {
        "part": "온보드 연산",
        "model": "Jetson Orin NX SOM",
        "grams": 28.0,
        "note": "SOM only; carrier, cooling, storage and DC-DC excluded",
    },
    {"part": "360° LiDAR", "model": "Livox Mid-360", "grams": 265.0},
    {"part": "깊이 카메라", "model": "Intel RealSense D435i", "grams": 72.0},
    {
        "part": "비행 제어기",
        "model": "Pixhawk 6C Mini (revision not frozen)",
        "grams_min": 39.2,
        "grams_max": 46.8,
        "note": "official revision range; exact installed revision pending",
    },
]


def _stub_package():
    """Import the robot configs without pulling in isaacgym (aerial_gym/__init__ imports it)."""
    if "aerial_gym" in sys.modules:
        return
    pkg = types.ModuleType("aerial_gym")
    pkg.__path__ = [str(ROOT / "aerial_gym")]
    pkg.AERIAL_GYM_DIRECTORY = str(ROOT)
    sys.modules["aerial_gym"] = pkg
    sys.path.insert(0, str(ROOT))


def _urdf_facts(filename):
    root = ET.parse(URDF_DIR / filename).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    origins = {}
    for j in root.findall("joint"):
        xyz = j.find("origin").get("xyz").split()
        origins[j.find("child").get("link")] = tuple(float(v) for v in xyz)

    def mass(link):
        return float(link.find("inertial/mass").get("value"))

    base = links["base_link"]
    binert = base.find("inertial/inertia")
    ixx = float(binert.get("ixx"))
    izz = float(binert.get("izz"))
    # Rotor links carry mass with a zero inertia tensor, so they contribute by parallel axis only.
    for name, link in links.items():
        if name == "base_link":
            continue
        m = mass(link)
        if m == 0.0:
            continue
        x, y, z = origins[name]
        ixx += m * (y * y + z * z)
        izz += m * (x * x + y * y)

    box = [float(v) for v in base.find("collision/geometry/box").get("size").split()]
    return {
        "urdf": filename,
        "mass_kg": round(sum(mass(l) for l in links.values()), 6),
        "base_mass_kg": mass(base),
        "ixx_assembled": ixx,
        "izz_assembled": izz,
        "collision_box_m": box,
    }


def _config_facts(mod_name, cls_name):
    import importlib
    cfg = getattr(importlib.import_module(mod_name), cls_name)
    ca = cfg.control_allocator_config
    mm = ca.motor_model_config
    arms = {round(abs(v), 9) for v in ca.allocation_matrix[3] + ca.allocation_matrix[4]}
    return {
        "config": f"{mod_name}.{cls_name}",
        "arm_m": (arms.pop() if len(arms) == 1 else None),
        "max_thrust_n": float(mm.max_thrust),
        "motor_tau_s": float(mm.motor_time_constant_increasing_min),
        "thrust_k_min": float(mm.motor_thrust_constant_min),
        "thrust_k_max": float(mm.motor_thrust_constant_max),
        "thrust_to_torque_ratio": float(mm.thrust_to_torque_ratio),
        "num_motors": int(ca.num_motors),
    }


def _derive(u, c, max_tilt_deg=45.0):
    """Envelope quantities. Roll uses constant-total-thrust pure roll: two motors to zero, two to
    twice hover, so tau = arm * 2 * (2*hover). Reported as an ALGEBRAIC bound, not a measurement --
    the 10 Hz control loop never gets near it."""
    m = u["mass_kg"]
    total_thrust = c["max_thrust_n"] * c["num_motors"]
    tw = total_thrust / (m * G)
    hover_per_motor = m * G / c["num_motors"]
    roll_alpha = (c["arm_m"] * 4.0 * hover_per_motor / u["ixx_assembled"]) if c["arm_m"] else None
    k_mid = 0.5 * (c["thrust_k_min"] + c["thrust_k_max"])
    return {
        "thrust_total_n": round(total_thrust, 4),
        "thrust_to_weight": round(tw, 4),
        "hover_per_motor_n": round(hover_per_motor, 4),
        "climb_accel_mps2": round((tw - 1.0) * G, 3),
        "horizontal_accel_mps2": round(G * math.tan(math.radians(max_tilt_deg)), 3),
        "roll_alpha_radps2": round(roll_alpha, 1) if roll_alpha else None,
        "motor_diagonal_m": round(2 * c["arm_m"] * math.sqrt(2.0), 4) if c["arm_m"] else None,
        "box_diagonal_m": round(u["collision_box_m"][0] * math.sqrt(2.0), 4),
        "implied_max_rpm": round(math.sqrt(c["max_thrust_n"] / k_mid) * 60.0) if k_mid else None,
    }


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def build():
    _stub_package()
    robots = []
    for spec in ROBOTS:
        cfgf = _config_facts(spec["config"], spec["cls"])
        import importlib
        cfg = getattr(importlib.import_module(spec["config"]), spec["cls"])
        urdff = _urdf_facts(cfg.robot_asset.file)
        robots.append({
            **spec, **urdff, **cfgf,
            "derived": _derive(urdff, cfgf),
        })

    envelope = None
    if ENVELOPE.exists():
        raw = json.loads(ENVELOPE.read_text(encoding="utf-8"))
        envelope = {
            "verdict": raw.get("verdict"),
            "seed": raw.get("seed"),
            "num_envs": raw.get("num_envs"),
            "bars": raw.get("bars"),
            "checks": raw.get("checks"),
            "arms": {k: raw[k] for k in ("legacy", "ref5in") if k in raw},
            "source": str(ENVELOPE.relative_to(ROOT)),
        }

    payload_min_g = sum(p.get("grams", p.get("grams_min", 0.0)) for p in PAYLOAD)
    payload_max_g = sum(p.get("grams", p.get("grams_max", 0.0)) for p in PAYLOAD)
    return {
        "schema": "motar.platform/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "tools/generate_platform_spec.py",
        "source_commit": git_commit(),
        "robots": robots,
        "payload": {
            "parts": PAYLOAD,
            "complete": False,
            "partial_grams_min": payload_min_g,
            "partial_grams_max": payload_max_g,
            "excluded": [
                "compute carrier/cooling/storage/DC-DC",
                "wiring and mounts",
                "frame, motors, ESCs, propellers, battery",
            ],
            "claim_guard": (
                "This is an incomplete named-part subtotal, not payload mass or aircraft AUW."
            ),
        },
        "envelope": envelope,
    }


def main():
    data = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    for r in data["robots"]:
        d = r["derived"]
        print(f"{r['key']:20s} {r['mass_kg']:.3f} kg · T/W {d['thrust_to_weight']:.3f} · "
              f"box {r['collision_box_m'][0]:.3f}×{r['collision_box_m'][2]:.3f} m · "
              f"arm {r['arm_m']:.4f} m · roll α {d['roll_alpha_radps2']} rad/s²")
    payload = data["payload"]
    print(f"incomplete named-part subtotal {payload['partial_grams_min']:.1f}–"
          f"{payload['partial_grams_max']:.1f} g · envelope "
          f"{(data['envelope'] or {}).get('verdict', 'not run')}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
