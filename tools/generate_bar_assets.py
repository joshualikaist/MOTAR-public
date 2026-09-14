"""Generate the pool of variable-size obstacle bars for navrl_bars_env.

Each bar is an axis-aligned box whose horizontal footprint (width, depth) is random in
[0.4, 0.8] m and whose height is fixed at 2.0 m. The box is centered at its own origin, so
placing it at z = 1 m makes it stand from the floor (0 m) up to 2 m. The environment picks
`num_assets` of these at random per env (bar_asset_params.file = None).

Run once (positions are randomized at run time, only the *shapes* live in these files):
    python tools/generate_bar_assets.py               # legacy 2.0 m pool -> .../bars
    BAR_HEIGHT=3.0 BAR_POOL=bars_h3 python tools/generate_bar_assets.py
        # v2 full-height pool: 3.0 m bars in a 3 m arena remove the fly-over option entirely
        # (NavRL's obstacles are mostly 4-6 m in a 4.5 m map for the same reason). A separate
        # folder keeps old checkpoints/evals bit-identical; select at runtime with
        # NAVRL_BAR_POOL=bars_h3 (see env_object_config.bar_asset_params).
"""
import os
import numpy as np

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "resources/models/environment_assets",
    os.environ.get("BAR_POOL", "bars").strip() or "bars",
)
N = 40           # pool size (env randomly picks 16 with replacement each reset)
HEIGHT = float(os.environ.get("BAR_HEIGHT", "2.0") or 2.0)  # bar height [m]
MIN_WD, MAX_WD = 0.4, 0.8  # random width/depth range [m]
SEED = 42

URDF = """<?xml version='1.0' encoding='UTF-8'?>
<robot name="bar_{i:03d}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual name="base_link_visual">
      <geometry><box size="{w:.4f} {d:.4f} {h:.4f}"/></geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </visual>
    <collision name="base_link_collision">
      <geometry><box size="{w:.4f} {d:.4f} {h:.4f}"/></geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
  </link>
</robot>
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".urdf"):
            os.remove(os.path.join(OUT, f))
    rng = np.random.default_rng(SEED)
    for i in range(N):
        w = float(rng.uniform(MIN_WD, MAX_WD))
        d = float(rng.uniform(MIN_WD, MAX_WD))
        with open(os.path.join(OUT, f"bar_{i:03d}.urdf"), "w") as fh:
            fh.write(URDF.format(i=i, w=w, d=d, h=HEIGHT))
    print(f"generated {N} bars in {OUT} (w,d in [{MIN_WD},{MAX_WD}] m, height {HEIGHT} m)")


if __name__ == "__main__":
    main()
