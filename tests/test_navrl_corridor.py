"""CPU unit tests for the corridor (free-gap) extractor.

Run: PYTHONNOUSERSITE=1 python tests/test_navrl_corridor.py
No simulator, no GPU: synthetic range profiles with analytically known gap geometry.
"""

import importlib.util
import math
import os
import sys

import torch

# Load by file path: the aerial_gym package __init__ imports isaacgym (GPU/vendored), which this
# pure-geometry test must not depend on. navrl_corridor.py itself imports only torch + math.
_MOD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "aerial_gym", "task", "navrl_task", "navrl_corridor.py",
)
_spec = importlib.util.spec_from_file_location("navrl_corridor", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CORRIDOR_DIM = _mod.CORRIDOR_DIM
extract_corridor_tokens = _mod.extract_corridor_tokens

HBEAMS = 72
MAX_RANGE = 12.0
FOV = 240.0
HORIZON = 6.0
MIN_W = 0.55


def bearings():
    bin_rad = 2.0 * math.pi / HBEAMS
    return torch.linspace(math.pi, -math.pi + bin_rad, HBEAMS)


def profile(fill=MAX_RANGE):
    return torch.full((1, HBEAMS), float(fill))


def put(near, bearing_deg, r):
    """Set the bin nearest to bearing_deg to range r."""
    b = bearings()
    idx = (b - math.radians(bearing_deg)).abs().argmin().item()
    near[0, idx] = r
    return idx


fails = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        fails.append(name)


def run(near, k=6):
    return extract_corridor_tokens(
        near, bearings(), max_range=MAX_RANGE, num_corridors=k,
        fov_deg=FOV, horizon_m=HORIZON, min_width_m=MIN_W,
    )


# T1: empty scene -> exactly one corridor spanning the whole FOV, centered forward.
tok, aux = run(profile())
check("empty: 1 valid corridor", int(aux["valid"].sum()) == 1)
check("empty: centered ~0", abs(float(aux["center"][0, 0])) < math.radians(3.0),
      f"({math.degrees(float(aux['center'][0, 0])):.1f} deg)")
check("empty: ang width ~FOV", abs(float(aux["ang_width"][0, 0]) - math.radians(FOV)) < math.radians(6.0))
check("empty: shape", tuple(tok.shape) == (1, 6, CORRIDOR_DIM))
check("empty: invalid slots zero", float(tok[0, 1:].abs().sum()) == 0.0)

# T2: two bars at +/-30 deg, 4 m -> the gap between them is centered at 0 with a known chord.
near = profile()
put(near, 30.0, 4.0)
put(near, -30.0, 4.0)
tok, aux = run(near)
v = aux["valid"][0]
centers = aux["center"][0]
mid = None
for i in range(6):
    if bool(v[i]) and abs(float(centers[i])) < math.radians(8.0):
        mid = i
        break
check("two-bar: central gap exists", mid is not None)
if mid is not None:
    # chord between surface endpoints at (4 m, +/-(30-bin) deg): 2*4*sin(~27.5 deg) ~= 3.69 m
    expect = 2.0 * 4.0 * math.sin(math.radians(27.5))
    got = float(aux["width_m"][0, mid])
    check("two-bar: width ~ chord", abs(got - expect) < 0.5, f"(got {got:.2f}, expect {expect:.2f})")
    check("two-bar: clearances = 4 m",
          abs(float(aux["left_clear_m"][0, mid]) - 4.0) < 1e-4
          and abs(float(aux["right_clear_m"][0, mid]) - 4.0) < 1e-4)
    check("two-bar: depth = max range", abs(float(aux["depth_m"][0, mid]) - MAX_RANGE) < 1e-4)

# T3: solid wall across the whole FOV at 3 m -> no corridor at all.
near = torch.full((1, HBEAMS), 3.0)
tok, aux = run(near)
check("wall: zero corridors", int(aux["valid"].sum()) == 0)
check("wall: tokens all zero", float(tok.abs().sum()) == 0.0)

# T4: a 5.5-m return does NOT open a corridor through it (horizon 6 m blocks it),
#     a 6.5-m return DOES stay open.
near = profile()
b = bearings()
fov_mask = b.abs() <= math.radians(FOV / 2)
near[0, fov_mask] = 5.5
tok, aux = run(near)
check("sub-horizon surface blocks", int(aux["valid"].sum()) == 0)
near[0, fov_mask] = 6.5
tok, aux = run(near)
check("supra-horizon stays open", int(aux["valid"].sum()) == 1)
check("supra-horizon depth=6.5", abs(float(aux["depth_m"][0, 0]) - 6.5) < 1e-4)

# T5: narrow slit below min width is dropped. Two bars 2 bins apart at 3 m:
#     gap ang ~5 deg at bound range 3 m -> chord ~0.26 m < 0.55.
near = torch.full((1, HBEAMS), 3.0)
b = bearings()
idx0 = (b - math.radians(2.5)).abs().argmin().item()
near[0, idx0] = MAX_RANGE  # single open bin between blocked neighbours
tok, aux = run(near)
check("slit below min width dropped", int(aux["valid"].sum()) == 0)

# T6: slot ordering is forward-first: gaps at ~+60 and ~-15 -> first slot is the -15 one.
near = torch.full((1, HBEAMS), 3.0)
for d in range(-25, -6):   # open run covering ~[-25,-7] deg
    put(near, d, MAX_RANGE)
for d in range(50, 71):    # open run covering ~[50,70] deg
    put(near, d, MAX_RANGE)
tok, aux = run(near)
v = aux["valid"][0]
check("two gaps found", int(v.sum()) == 2)
c0 = abs(float(aux["center"][0, 0]))
c1 = abs(float(aux["center"][0, 1]))
check("forward-first ordering", c0 <= c1, f"(|{math.degrees(c0):.0f}| <= |{math.degrees(c1):.0f}|)")

# T7: batch independence + determinism.
near2 = torch.cat([profile(), torch.full((1, HBEAMS), 3.0)], dim=0)
tok_a, _ = run(near2)
tok_b, _ = run(near2)
check("deterministic", bool(torch.equal(tok_a, tok_b)))
check("batch rows independent", float(tok_a[1].abs().sum()) == 0.0 and float(tok_a[0].abs().sum()) > 0.0)

# T8: feature normalisation bounds.
near = profile()
put(near, 10.0, 2.0)
put(near, -40.0, 3.5)
tok, _ = run(near)
check("features in [-1,1]", bool((tok >= -1.0 - 1e-6).all() and (tok <= 1.0 + 1e-6).all()))

print()
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("navrl corridor extractor: ok")
