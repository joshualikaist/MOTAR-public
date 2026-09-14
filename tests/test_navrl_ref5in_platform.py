"""Self-consistency tests for the NavRL 5-inch candidate platform (WORKLOG 2026-08-13).

CPU-only: parses the URDFs as XML and reads the robot configs as plain Python. No Isaac Gym, no
GPU, no simulator import.

What this guards. The legacy `quad_navrl_collide.urdf` mixes a 0.28 m collision proxy, 0.13 m
motor-arm coordinates, and stock mass/inertia. Nothing at runtime cross-checks the URDF inertial
block, joint origins and allocation matrix, so these tests pin their numerical relationship.

What this does NOT prove. Passing these checks does not establish a buildable hardware BOM,
packaging/CoM, actuator thermal margin, or task equivalence. The candidate keeps a 0.12 m collision
height versus legacy's 0.08 m: level XY literals match, but tilted bar-contact geometry changes.

Run: PYTHONNOUSERSITE=1 python tests/test_navrl_ref5in_platform.py
"""

import math
from pathlib import Path
import sys
import types
import unittest
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The real `aerial_gym/__init__.py` imports isaacgym and then every task/env/control module. Stand
# in a bare package with the same __path__ so the config submodules resolve normally -- they are
# plain declarations and need nothing from the simulator.
_pkg = sys.modules.get("aerial_gym")
if _pkg is None:
    _pkg = types.ModuleType("aerial_gym")
    sys.modules["aerial_gym"] = _pkg
# Other CPU-only NavRL tests may already have installed a minimal `aerial_gym` stand-in without a
# package path.  `setdefault` left that non-package in place, so this test passed by itself but
# failed under `unittest discover`.  Enrich the existing stand-in instead of replacing it (which
# could invalidate modules imported by an earlier test).
if not hasattr(_pkg, "__path__"):
    _pkg.__path__ = [str(REPO / "aerial_gym")]
if not hasattr(_pkg, "AERIAL_GYM_DIRECTORY"):
    _pkg.AERIAL_GYM_DIRECTORY = str(REPO)

from aerial_gym.config.robot_config.navrl_quad_config import NavRLQuadCfg  # noqa: E402
from aerial_gym.config.robot_config.navrl_ref5in_quad_config import (  # noqa: E402
    NavRLRef5inQuadCfg,
)

G = 9.81
PROP_RADIUS_5IN = 0.0635  # 127 mm diameter / 2

# With two maximum-footprint 0.8 m squares whose centres are exactly 1.6 m apart, the
# smallest AABB corner gap occurs at a 45-degree centre vector. This is much tighter than the
# incorrect axis-only 0.8 m estimate and is only a coarse static fit bound.
MIN_THEORETICAL_CORNER_GAP_M = math.sqrt(2.0) * (1.6 / math.sqrt(2.0) - 0.8)


def _urdf(name):
    return ET.parse(REPO / "resources/robots/quad" / name).getroot()


def _links(root):
    return {link.get("name"): link for link in root.findall("link")}


def _joint_origins(root):
    out = {}
    for joint in root.findall("joint"):
        xyz = joint.find("origin").get("xyz").split()
        out[joint.find("child").get("link")] = tuple(float(v) for v in xyz)
    return out


def _mass(link):
    return float(link.find("inertial/mass").get("value"))


def _inertia(link):
    node = link.find("inertial/inertia")
    return {k: float(node.get(k)) for k in ("ixx", "iyy", "izz")}


def _collision_box(link):
    return tuple(float(v) for v in link.find("collision/geometry/box").get("size").split())


def _assembled_inertia(root):
    """Total inertia about the base origin: base_link's own tensor plus the rotor links by parallel
    axis (their own tensors are zero, so each contributes m*(y^2+z^2) / m*(x^2+y^2))."""
    links, origins = _links(root), _joint_origins(root)
    base = _inertia(links["base_link"])
    ixx, izz = base["ixx"], base["izz"]
    for name, link in links.items():
        if name == "base_link":
            continue
        m = _mass(link)
        if m == 0.0:
            continue
        x, y, z = origins[name]
        ixx += m * (y * y + z * z)
        izz += m * (x * x + y * y)
    return ixx, izz


def _total_mass(root):
    return sum(_mass(link) for link in _links(root).values())


def _arm_from_allocation(cfg):
    """Recover the motor arm length from the roll/pitch rows of the allocation matrix."""
    roll, pitch = cfg.control_allocator_config.allocation_matrix[3:5]
    arms = {round(abs(v), 9) for v in roll + pitch}
    assert len(arms) == 1, f"non-square arm layout: {arms}"
    return arms.pop()


LEGACY = _urdf("quad_navrl_collide.urdf")
REF = _urdf("quad_navrl_ref5in.urdf")
TARGET = ET.parse(
    REPO / "resources/models/environment_assets/objects/navrl_target_drone.urdf"
).getroot()


class TestGeometryAgreesWithItself(unittest.TestCase):
    """Cross-check the candidate's declared geometry without claiming hardware equivalence."""

    def test_allocation_matrix_matches_urdf_motor_origins(self):
        arm = _arm_from_allocation(NavRLRef5inQuadCfg)
        origins = _joint_origins(REF)
        for i in range(4):
            x, y, z = origins[f"motor_{i}"]
            self.assertAlmostEqual(abs(x), arm, places=7, msg=f"motor_{i} x vs allocation")
            self.assertAlmostEqual(abs(y), arm, places=7, msg=f"motor_{i} y vs allocation")
            self.assertEqual(z, 0.0)

    def test_arm_visuals_sit_at_the_midpoint_of_their_arms(self):
        origins = _joint_origins(REF)
        for i in range(4):
            motor = origins[f"motor_{i}"]
            arm_visual = origins[f"arm_motor_{i}"]
            for m, a in zip(motor[:2], arm_visual[:2]):
                self.assertAlmostEqual(a, m / 2.0, places=6)

    def test_arm_visual_length_equals_the_motor_radius(self):
        arm = _arm_from_allocation(NavRLRef5inQuadCfg)
        radius = arm * math.sqrt(2.0)
        for link in _links(REF).values():
            cyl = link.find("visual/geometry/cylinder")
            if cyl is not None and link.get("name").startswith("arm_motor"):
                self.assertAlmostEqual(float(cyl.get("length")), radius, places=3)

    def test_level_collision_box_approximates_prop_disk_axis_aligned_bounds(self):
        """At level attitude, the 0.28 m square is within 1% of the prop disks' x/y AABB.

        It is still a box proxy: it slightly under-covers axial tips and over-covers the corners.
        This assertion must not be read as an exact swept-prop collision model.
        """
        arm = _arm_from_allocation(NavRLRef5inQuadCfg)
        w, d, _h = _collision_box(_links(REF)["base_link"])
        derived = 2.0 * (arm + PROP_RADIUS_5IN)
        self.assertAlmostEqual(derived, 0.2826, places=4)
        self.assertLess(abs(derived - w) / derived, 0.01, "box must be within 1% of the prop tips")
        self.assertEqual(w, d)

    def test_level_xy_box_literals_match_legacy(self):
        """Only the level XY literals match; the next test pins the tilted-contact difference."""
        lw, ld, _ = _collision_box(_links(REF)["base_link"])
        gw, gd, _ = _collision_box(_links(LEGACY)["base_link"])
        self.assertEqual((lw, ld), (gw, gd))

    def test_taller_box_changes_tilted_bar_contact_envelope(self):
        """A full-height bar sees the oriented 3-D box, not only its level XY literal.

        At 45 degrees pitch the horizontal support width is w*cos(theta)+h*sin(theta), so retaining
        0.12 m candidate height is an intentional whole-platform task-geometry change.
        """
        rw, _rd, rh = _collision_box(_links(REF)["base_link"])
        lw, _ld, lh = _collision_box(_links(LEGACY)["base_link"])
        theta = math.radians(45.0)
        ref_support = rw * math.cos(theta) + rh * math.sin(theta)
        legacy_support = lw * math.cos(theta) + lh * math.sin(theta)
        self.assertAlmostEqual(ref_support, 0.2828, places=4)
        self.assertAlmostEqual(legacy_support, 0.2546, places=4)
        self.assertAlmostEqual(ref_support - legacy_support, 0.0283, places=4)
        self.assertGreater(ref_support, legacy_support)

    def test_box_height_is_the_declared_candidate_envelope(self):
        """Pin the 0.12 m modelling choice; this is not a payload packing validation."""
        _w, _d, h = _collision_box(_links(REF)["base_link"])
        self.assertEqual(h, 0.12)

    def test_conservative_3d_box_diagonal_clears_theoretical_corner_gap(self):
        """A coarse geometric sanity bound, not a reachability or dynamic-flight proof."""
        w, d, h = _collision_box(_links(REF)["base_link"])
        self.assertLess(math.sqrt(w * w + d * d + h * h), MIN_THEORETICAL_CORNER_GAP_M)


class TestMassAndInertiaAgree(unittest.TestCase):

    def test_total_mass_is_the_declared_candidate_1200_g(self):
        self.assertAlmostEqual(_total_mass(REF), 1.200, places=3)

    def test_physical_target_rigid_body_matches_ref5in_equivalent_contract(self):
        target_base = _links(TARGET)["base_link"]
        ref_ixx, ref_izz = _assembled_inertia(REF)
        self.assertAlmostEqual(_mass(target_base), _total_mass(REF), places=6)
        self.assertAlmostEqual(_inertia(target_base)["ixx"], ref_ixx, places=6)
        self.assertAlmostEqual(_inertia(target_base)["iyy"], ref_ixx, places=6)
        self.assertAlmostEqual(_inertia(target_base)["izz"], ref_izz, places=6)
        self.assertEqual(
            _collision_box(target_base), _collision_box(_links(REF)["base_link"])
        )

    def test_mass_budget_exposes_conditional_compute_and_integration_allowance(self):
        """Document the candidate arithmetic without treating an incomplete BOM as buildability.

        The base subtotal is frame+motors+props+ESC+battery.  The avionics subtotal uses the stated
        Mid-360 and D435i masses plus the published Pixhawk 6C Mini model range.  Everything not in
        those subtotals -- complete Orin assembly, carrier/storage/cooling, DC/DC, wiring, mounts,
        fasteners and reserve -- must fit in the remainder and still needs a measured BOM.
        """
        base_vehicle_kg = (120 + 132 + 20 + 40 + 240) / 1000.0
        sensors_min_kg = (265 + 72 + 39.2) / 1000.0
        sensors_max_kg = (265 + 72 + 46.8) / 1000.0
        allowance_max = _total_mass(REF) - base_vehicle_kg - sensors_min_kg
        allowance_min = _total_mass(REF) - base_vehicle_kg - sensors_max_kg
        self.assertAlmostEqual(base_vehicle_kg, 0.552, places=6)
        self.assertAlmostEqual(allowance_min, 0.2642, places=4)
        self.assertAlmostEqual(allowance_max, 0.2718, places=4)
        self.assertGreater(allowance_min, 0.0)

    def test_inertia_matches_the_documented_box_model(self):
        """base_link is derived as a 0.15 x 0.15 x 0.12 m box; recompute it rather than trusting the
        literal, since a stale inertia literal is exactly the legacy defect."""
        base = _links(REF)["base_link"]
        m, w, h = _mass(base), 0.15, 0.12
        self.assertAlmostEqual(_inertia(base)["ixx"], m * (w * w + h * h) / 12.0, places=6)
        self.assertAlmostEqual(_inertia(base)["iyy"], m * (w * w + h * h) / 12.0, places=6)
        self.assertAlmostEqual(_inertia(base)["izz"], m * (w * w + w * w) / 12.0, places=6)

    def test_assembled_inertia_stays_inside_the_declared_sanity_band(self):
        """Catch gross stale literals; this heuristic band is not CAD or measured validation."""
        ixx, izz = _assembled_inertia(REF)
        m = _total_mass(REF)
        w, _d, h = _collision_box(_links(REF)["base_link"])
        uniform_ixx = m * (w * w + h * h) / 12.0
        uniform_izz = m * (w * w + w * w) / 12.0
        # Preregistered broad sanity band for a centrally concentrated candidate model.
        self.assertGreater(ixx, 0.25 * uniform_ixx)
        self.assertLessEqual(ixx, uniform_ixx)
        self.assertGreater(izz, 0.25 * uniform_izz)
        self.assertLessEqual(izz, uniform_izz)

    def test_izz_exceeds_ixx_as_it_must_for_a_flat_multirotor(self):
        ixx, izz = _assembled_inertia(REF)
        self.assertGreater(izz, ixx)


class TestNominalFlightEnvelopeAssumptions(unittest.TestCase):
    """Pin nominal parameter relationships; runtime response is a separate GPU gate."""

    def _tw(self, root, cfg):
        thrust = 4.0 * cfg.control_allocator_config.motor_model_config.max_thrust
        return thrust / (_total_mass(root) * G)

    def test_nominal_thrust_to_weight_cap_is_unchanged(self):
        self.assertAlmostEqual(
            self._tw(REF, NavRLRef5inQuadCfg), self._tw(LEGACY, NavRLQuadCfg), places=3
        )

    def test_nominal_vertical_acceleration_ceiling_is_unchanged(self):
        ref = (self._tw(REF, NavRLRef5inQuadCfg) - 1.0) * G
        legacy = (self._tw(LEGACY, NavRLQuadCfg) - 1.0) * G
        self.assertAlmostEqual(ref, legacy, places=2)
        self.assertAlmostEqual(ref, 22.19, places=1)

    def test_configured_motor_time_constant_is_unchanged(self):
        """This checks a simulator setting, not a sourced or bench-measured 40 ms response."""
        ref = NavRLRef5inQuadCfg.control_allocator_config.motor_model_config
        legacy = NavRLQuadCfg.control_allocator_config.motor_model_config
        for attr in (
            "motor_time_constant_increasing_min",
            "motor_time_constant_increasing_max",
            "motor_time_constant_decreasing_min",
            "motor_time_constant_decreasing_max",
        ):
            self.assertEqual(getattr(ref, attr), getattr(legacy, attr), attr)

    def test_thrust_constant_is_a_fixed_coordinate_calibration(self):
        """Pin one nominal k coordinate without mislabelling it motor-strength randomization.

        In the unconstrained ``use_rps`` update, k is used to map thrust to rotor speed and back and
        algebraically cancels from the normalized first-order transient.  The candidate therefore
        fixes min=max; whether the implied 28,023 RPM is achievable still requires exact
        motor/prop/voltage bench data and is not established by this CPU test.
        """
        mm = NavRLRef5inQuadCfg.control_allocator_config.motor_model_config
        self.assertEqual(mm.motor_thrust_constant_min, mm.motor_thrust_constant_max)
        self.assertAlmostEqual(mm.motor_thrust_constant_min, 4.401e-5, places=10)
        rpm = math.sqrt(mm.max_thrust / mm.motor_thrust_constant_min) * 60.0
        self.assertAlmostEqual(rpm, 28023.0, delta=2.0)

    def test_hover_thrust_roll_authority_ratio_is_the_documented_0_59x(self):
        """Constant-total-thrust (=mg) roll differential, not maximum angular acceleration.

        Two motors go to zero and two to twice hover, giving tau=arm*4*hover.  Actual closed-loop
        response also depends on inherited Lee gains, motor dynamics and allocator clipping and is
        measured separately by `verify_navrl_ref_platform.py`.
        """

        def alpha(root, cfg):
            m = _total_mass(root)
            arm = _arm_from_allocation(cfg)
            hover = m * G / 4.0
            ixx, _izz = _assembled_inertia(root)
            return arm * 4.0 * hover / ixx

        legacy_alpha = alpha(LEGACY, NavRLQuadCfg)
        ref_alpha = alpha(REF, NavRLRef5inQuadCfg)
        ratio = ref_alpha / legacy_alpha
        self.assertAlmostEqual(legacy_alpha, 377.3, places=1)
        self.assertAlmostEqual(ref_alpha, 221.1, places=1)
        self.assertAlmostEqual(ratio, 0.586, places=2)


class TestConfigWiring(unittest.TestCase):

    def test_ref_config_points_at_the_ref_urdf(self):
        self.assertEqual(NavRLRef5inQuadCfg.robot_asset.file, "quad_navrl_ref5in.urdf")
        self.assertEqual(NavRLQuadCfg.robot_asset.file, "quad_navrl_collide.urdf")

    def test_sensor_and_spawn_contract_is_inherited_untouched(self):
        """The observation contract must not move with the airframe."""
        self.assertIs(NavRLRef5inQuadCfg.sensor_config.lidar_config,
                      NavRLQuadCfg.sensor_config.lidar_config)
        self.assertEqual(NavRLRef5inQuadCfg.sensor_config.enable_camera,
                         NavRLQuadCfg.sensor_config.enable_camera)
        self.assertEqual(NavRLRef5inQuadCfg.init_config.min_init_state,
                         NavRLQuadCfg.init_config.min_init_state)
        self.assertEqual(NavRLRef5inQuadCfg.init_config.max_init_state,
                         NavRLQuadCfg.init_config.max_init_state)

    def test_registered_under_its_own_name(self):
        source = (REPO / "aerial_gym/robots/__init__.py").read_text()
        self.assertIn('robot_registry.register("navrl_ref5in_quad", BaseMultirotor,'
                      ' NavRLRef5inQuadCfg)', source)

    def test_task_config_default_is_still_the_legacy_body(self):
        """Switching the default would silently invalidate every existing checkpoint's provenance;
        the reference platform must be opted into with NAVRL_ROBOT."""
        source = (REPO / "aerial_gym/config/task_config/navrl_task_config.py").read_text()
        self.assertIn('robot_name = os.environ.get("NAVRL_ROBOT", "").strip() or "navrl_quad"',
                      source)


class TestLegacyDefectIsPinned(unittest.TestCase):
    """Reproduces the 2026-08-13 diagnosis so it stays checkable, and fails loudly if someone edits
    the legacy URDF instead of migrating to the reference platform (which would break the frozen
    ep25000+riskcap checkpoint's provenance)."""

    def test_legacy_motor_span_is_not_covered_by_its_collision_proxy_with_5in_props(self):
        """At the declared +/-0.13 m coordinates, even 5-inch disks exceed the 0.28 m XY box.

        A 368 mm motor diagonal is compatible with several layouts; this test does not infer which
        prop a real frame must use. It pins only the literal geometry mismatch in this model.
        """
        arm = _arm_from_allocation(NavRLQuadCfg)
        self.assertAlmostEqual(arm, 0.13, places=6)
        self.assertAlmostEqual(2.0 * arm * math.sqrt(2.0), 0.3677, places=3)
        w, _d, _h = _collision_box(_links(LEGACY)["base_link"])
        self.assertLess(w, 2.0 * (arm + PROP_RADIUS_5IN))
        self.assertAlmostEqual(2.0 * (arm + 0.0889), 0.4378, places=3)  # conditional 7" AABB

    def test_legacy_mass_is_below_the_two_sensor_nominal_subtotal(self):
        """A conservative inconsistency that does not depend on an incomplete 472 g BOM.

        The stated Mid-360 (265 g) and D435i (72 g) nominal masses alone total 337 g, already above
        the entire 250 g legacy actor; compute, autopilot, carrier, power and mounts are excluded.
        """
        self.assertAlmostEqual(_total_mass(LEGACY), 0.250, places=3)
        self.assertLess(_total_mass(LEGACY), 0.337)

    def test_legacy_base_inertia_literal_is_a_150_mm_plate(self):
        """The third, subtler one: the base_link inertia is untouched Aerial Gym stock, and solving
        ixx = m*w^2/12 with izz = 2*ixx (h = 0) gives a zero-thickness 0.150 m square plate. That
        matches neither the 0.28 m box nor the 0.368 m motor span."""
        base = _links(LEGACY)["base_link"]
        m, inertia = _mass(base), _inertia(base)
        self.assertAlmostEqual(math.sqrt(12.0 * inertia["ixx"] / m), 0.150, places=3)
        self.assertAlmostEqual(inertia["izz"], 2.0 * inertia["ixx"], places=9)  # h = 0

    def test_legacy_assembled_inertia_is_nevertheless_plausible(self):
        """Why the stale literal went unnoticed for three weeks, recorded so the write-up does not
        overstate it. Adding the four rotor masses at +/-0.13 m doubles ixx by parallel axis, and
        the assembled 8.45e-4 lands at 0.48x a uniform 0.28 m box. The legacy dynamics are therefore
        numerically survivable, while its 250 g total remains inconsistent with the stated sensor
        pair's nominal mass subtotal."""
        ixx, _izz = _assembled_inertia(LEGACY)
        m = _total_mass(LEGACY)
        w, _d, h = _collision_box(_links(LEGACY)["base_link"])
        uniform = m * (w * w + h * h) / 12.0
        self.assertAlmostEqual(ixx, 8.45e-4, places=6)
        self.assertTrue(0.4 < ixx / uniform < 0.55, f"{ixx / uniform:.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
