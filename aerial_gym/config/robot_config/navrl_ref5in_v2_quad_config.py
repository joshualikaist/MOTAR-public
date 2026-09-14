from aerial_gym.config.robot_config.navrl_ref5in_quad_config import NavRLRef5inQuadCfg


class NavRLRef5inV2QuadCfg(NavRLRef5inQuadCfg):
    """Fresh-only ref5in geometry correction; never compatible with v1 checkpoints."""

    class robot_asset(NavRLRef5inQuadCfg.robot_asset):
        file = "quad_navrl_ref5in_v2.urdf"
