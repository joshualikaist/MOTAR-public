"""Low-overhead PhysX configuration for the single-environment NavRL viewer.

The normal base configuration reserves contact buffers for thousands of parallel environments.
Those fixed allocations are wasteful for the interactive viewer and can exceed an 8 GB GPU while
training is also resident. These values change capacity only, not timestep or dynamics.
"""

from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig


class NavRLViewerSimConfig(BaseSimConfig):
    class sim(BaseSimConfig.sim):
        class physx(BaseSimConfig.sim.physx):
            # One viewer environment has roughly 150 static bars and one robot. 262k potential
            # contact pairs leaves a large margin without the 2**24 (~8000-env) reservation.
            max_gpu_contact_pairs = 2**18
            default_buffer_size_multiplier = 1
