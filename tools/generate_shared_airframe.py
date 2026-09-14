#!/usr/bin/env python3
"""Generate the shared quadrotor visual, so one definition can serve every asset that needs it.

Most dimensions are read from something that already exists: motor positions and the arm and
motor cylinders come from quad_navrl_ref5in_v2.urdf via read_airframe_layout, and the propeller
radius from PROP_RADIUS_5IN_M via read_prop_radius. Two are NOT read, and an earlier version of
this docstring wrongly said they were. BODY_BOX_M is transcribed from a COMMENT in that URDF,
because an inertia approximation stated in prose cannot be parsed; check_body_box_matches_urdf
re-reads that comment so a transcription that goes stale fails loudly instead of silently. And
PROP_THICKNESS_M is chosen here outright: a propeller has no thickness anywhere in this repo.

Scope and invariants are in docs/v1_shared_airframe_contract.md.

Writing the URDF is all this does. It changes no configuration and swaps no asset.
"""
import argparse
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ElementTree

ROOT = Path(__file__).resolve().parents[1]
INTERCEPTOR = ROOT / "resources/robots/quad/quad_navrl_ref5in_v2.urdf"
TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"

# Part colours. The detector currently flat-fills target pixels with its configured colour, so
# these do not reach any image yet; V2 is what makes them matter. Neutral greys on purpose: a
# target that is the only red object in the scene can be found by colour alone.
MATERIALS = (("AirframeBody", (0.18, 0.18, 0.20, 1.0)),
             ("AirframeArm", (0.15, 0.15, 0.16, 1.0)),
             ("AirframeMotor", (0.45, 0.45, 0.47, 1.0)),
             ("AirframeProp", (0.72, 0.72, 0.75, 1.0)))
# CHOSEN here, not read: no propeller thickness is stated anywhere in this repository. Thin
# enough to sit inside the body height, thick enough not to vanish at a shallow viewing angle.
PROP_THICKNESS_M = 0.006
# TRANSCRIBED from a comment in the interceptor URDF, which states its inertia approximation in
# prose. check_body_box_matches_urdf verifies the transcription against that comment.
BODY_BOX_M = (0.15, 0.15, 0.12)


def read_prop_radius(path=TASK):
    """Take the propeller radius from the task constant instead of restating it here."""
    for line in Path(path).read_text().splitlines():
        if line.startswith("PROP_RADIUS_5IN_M"):
            return float(line.split("=")[1].split("#")[0].strip())
    raise ValueError(f"PROP_RADIUS_5IN_M is not defined in {path}")


def check_body_box_matches_urdf(path=INTERCEPTOR, box=BODY_BOX_M):
    """Fail if the URDF comment BODY_BOX_M was transcribed from no longer states that box.

    The base inertia approximation lives in prose, so it cannot be parsed the way a joint origin
    can. Searching the text for the transcribed numbers is weaker than reading a tag, but it does
    turn a silently stale copy into a loud failure, which is the part that matters.
    """
    text = Path(path).read_text()
    stated = " x ".join(("%g" % value) for value in box)
    if stated not in text:
        raise ValueError(f"{path} no longer documents a {stated} m base box; BODY_BOX_M is stale")
    return stated


def read_airframe_layout(path=INTERCEPTOR):
    """Motor positions and the arm/motor cylinder sizes, straight out of the interceptor URDF."""
    root = ElementTree.parse(path).getroot()
    origins, cylinders = {}, {}
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        origin = joint.find("origin")
        origins[child] = [float(v) for v in (origin.get("xyz") if origin is not None else "0 0 0").split()]
    for link in root.findall("link"):
        shape = link.find("visual/geometry/cylinder")
        if shape is not None:
            cylinders[link.get("name")] = (float(shape.get("radius")), float(shape.get("length")))
    motors = [origins[f"motor_{i}"] for i in range(4)]
    arms = [cylinders[f"arm_motor_{i}"] for i in range(4)]
    hubs = [cylinders[f"motor_{i}"] for i in range(4)]
    if len({tuple(a) for a in arms}) != 1 or len({tuple(h) for h in hubs}) != 1:
        raise ValueError("the four arms and the four motors must each be identical")
    if len({round(math.hypot(x, y), 9) for x, y, _ in motors}) != 1:
        raise ValueError("the four motors must sit on one circle")
    return motors, arms[0], hubs[0]


def visual_elements(motors, arm, hub, prop_radius):
    """One (name, geometry, origin, material) tuple per visual, in a fixed order."""
    elements = [("body", ("box", {"size": "%s %s %s" % BODY_BOX_M}), (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 "AirframeBody")]
    for index, (x, y, _) in enumerate(motors):
        yaw = math.atan2(y, x)
        # A cylinder points along +Z; pitch by 90 degrees to lay it flat, then yaw it outward.
        elements.append((f"arm_{index}",
                         ("cylinder", {"radius": repr(arm[0]), "length": repr(arm[1])}),
                         (x / 2.0, y / 2.0, 0.0, 0.0, math.pi / 2.0, yaw), "AirframeArm"))
    for index, (x, y, _) in enumerate(motors):
        elements.append((f"motor_{index}",
                         ("cylinder", {"radius": repr(hub[0]), "length": repr(hub[1])}),
                         (x, y, 0.0, 0.0, 0.0, 0.0), "AirframeMotor"))
    for index, (x, y, _) in enumerate(motors):
        # Sitting just above the motor hub, the way a propeller does.
        elements.append((f"prop_{index}",
                         ("cylinder", {"radius": repr(prop_radius), "length": repr(PROP_THICKNESS_M)}),
                         (x, y, hub[1] / 2.0 + PROP_THICKNESS_M / 2.0, 0.0, 0.0, 0.0), "AirframeProp"))
    return elements


def render_part_links(elements, indent="  "):
    """A massless link per visual after the first, each fixed to base_link.

    Not a style choice. aerial_gym/assets/warp_asset.py indexes its per-LINK name list with a
    per-MESH counter, so it assumes one mesh per link; thirteen visuals in one link made it raise
    IndexError before a single frame rendered. The interceptor URDF already has this shape, so it
    is known to load. collapse_fixed_joints is set on this asset class, and the added links carry
    zero mass and no collision, so contact and dynamics stay entirely with base_link.
    """
    lines = []
    for name, geometry, origin, material in elements[1:]:
        lines.append(f'{indent}<link name="{name}">')
        lines.append(render_visual_xml([(name, geometry, (0.0,) * 6, material)], indent + "  "))
        lines.append(f"{indent}  <inertial>")
        lines.append(f'{indent}    <mass value="0.0"/>')
        lines.append(f'{indent}    <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>')
        lines.append(f"{indent}  </inertial>")
        lines.append(f"{indent}</link>")
        lines.append(f'{indent}<joint name="base_link_to_{name}" type="fixed">')
        lines.append(f'{indent}  <parent link="base_link"/>')
        lines.append(f'{indent}  <child link="{name}"/>')
        lines.append(f'{indent}  <origin xyz="{origin[0]!r} {origin[1]!r} {origin[2]!r}" '
                     f'rpy="{origin[3]!r} {origin[4]!r} {origin[5]!r}"/>')
        lines.append(f"{indent}</joint>")
    return "\n".join(lines)


def render_visual_xml(elements, indent="    "):
    lines = []
    for name, (kind, attributes), origin, material in elements:
        attribute_text = " ".join(f'{key}="{value}"' for key, value in attributes.items())
        lines.append(f'{indent}<visual name="{name}">')
        lines.append(f"{indent}  <geometry>")
        lines.append(f"{indent}    <{kind} {attribute_text}/>")
        lines.append(f"{indent}  </geometry>")
        lines.append(f'{indent}  <origin xyz="{origin[0]!r} {origin[1]!r} {origin[2]!r}" '
                     f'rpy="{origin[3]!r} {origin[4]!r} {origin[5]!r}"/>')
        lines.append(f'{indent}  <material name="{material}"/>')
        lines.append(f"{indent}</visual>")
    return "\n".join(lines)


def element_text(element, indent="  ", level=0):
    """Serialise one element with a stable two-space indent, on any Python version.

    This used to call ElementTree.indent, which arrived in 3.9, behind a hasattr guard that
    silently did nothing on 3.8. The generator then emitted the source file's own whitespace on
    3.8 and normalised whitespace on 3.13, so `--check` reported IDENTICAL under one interpreter
    and DIFFERS under another, and the byte-exactness invariant depended on who ran it.
    """
    pad = indent * level
    attributes = "".join(f' {key}="{value}"' for key, value in element.items())
    children = list(element)
    if not children:
        return f"{pad}<{element.tag}{attributes}/>"
    inner = "\n".join(element_text(child, indent, level + 1) for child in children)
    return f"{pad}<{element.tag}{attributes}>\n{inner}\n{pad}</{element.tag}>"


def build_target_urdf(source, robot_name):
    """Keep the source's inertial and collision verbatim; replace only the visual geometry."""
    root = ElementTree.parse(source).getroot()
    links = root.findall("link")
    if len(links) != 1 or links[0].get("name") != "base_link":
        raise ValueError(f"{source} must be a single base_link asset to derive from")
    inertial, collision = links[0].find("inertial"), links[0].find("collision")
    if inertial is None or collision is None:
        raise ValueError(f"{source} needs both <inertial> and <collision> to preserve")
    check_body_box_matches_urdf()
    motors, arm, hub = read_airframe_layout()
    prop_radius = read_prop_radius()
    elements = visual_elements(motors, arm, hub, prop_radius)
    span = 2.0 * (math.hypot(*motors[0][:2]) / math.sqrt(2.0) + prop_radius)
    body = ["<?xml version='1.0' encoding='UTF-8'?>",
            "<!-- GENERATED by tools/generate_shared_airframe.py; edit that, not this file.",
            "",
            "     The shared quadrotor visual: a body box, four arms, four motors and four",
            "     propeller discs. Every dimension is read from an existing source, none is new:",
            "     motor positions and the arm and motor cylinders from quad_navrl_ref5in_v2.urdf,",
            "     the propeller radius from PROP_RADIUS_5IN_M in navrl_task.py, and the body box",
            "     from the inertia approximation that URDF documents.",
            "",
            f"     Propeller-tip span is {span:.7f} m, inside the {collision.find('geometry/box').get('size').split()[0]} m collision box, so apparent",
            "     width changes by a fraction of a percent rather than by the 28% that copying the",
            "     interceptor's own visual would have cost.",
            "",
            "     Inertial and collision are carried over from navrl_target_drone_v2.urdf unchanged.",
            "     The simulator reads collision only, so this file changes no physics. Colours do",
            "     not reach any image until the detector's flat fill is replaced. -->",
            f'<robot name="{robot_name}">',
            '  <link name="base_link">',
            element_text(inertial, level=2),
            render_visual_xml([elements[0]]),
            element_text(collision, level=2),
            "  </link>",
            render_part_links(elements)]
    for name, colour in MATERIALS:
        body.append(f'  <material name="{name}">')
        body.append('    <color rgba="%s %s %s %s"/>' % colour)
        body.append("  </material>")
    body.append("</robot>")
    return "\n".join(body) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "resources/models/environment_assets/objects/navrl_target_drone_v2.urdf")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf")
    parser.add_argument("--robot-name", default="navrl_target_drone_v3")
    parser.add_argument("--check", action="store_true",
                        help="Regenerate and report whether the file on disk is byte-identical")
    args = parser.parse_args()
    text = build_target_urdf(args.source, args.robot_name)
    if args.check:
        current = args.output.read_text() if args.output.exists() else None
        identical = current == text
        print("IDENTICAL" if identical else "DIFFERS", args.output)
        raise SystemExit(0 if identical else 1)
    args.output.write_text(text)
    print("wrote", args.output, f"({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
