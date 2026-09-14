"""Small-VRAM (~4 GB) variant of BaseSimConfig for secondary training machines.

Identical to ``base_sim`` except the PhysX **GPU buffer capacities** are shrunk from the
8000-env defaults. Those defaults (``max_gpu_contact_pairs = 2**24`` and a 10x buffer
multiplier) pre-allocate a fixed, num_envs-independent chunk of GPU memory that alone
exceeds a 4 GB card once the torch CUDA context is loaded — so training OOMs in PhysX
(``GymPhysX.cpp: out of memory``) even at a handful of envs.

These are memory-capacity ceilings, not dynamics parameters: physics behaviour is
unchanged. Sized for the few dozen envs a 4 GB GPU (e.g. GTX 1650) can run headless.
Select it via ``AERIAL_GYM_SIM_NAME=base_sim_4gb`` (navrl_task_config reads this) — the
``GPU4GB=1 ./train_navrl.sh`` wrapper sets it together with ppo_navrl_cnn_4gb.yaml, whose PPO
hyperparameters (incl. minibatch_size=4096) deliberately MATCH the 8 GB yaml so density runs
stay comparable; only the PhysX buffer capacities shrink. Validated at 512 envs for the
25/50-bar builds (~2.5 GB). The 8 GB main machine keeps ``base_sim``.
"""

from aerial_gym.config.sim_config.base_sim_config import BaseSimConfig


class BaseSim4GBConfig(BaseSimConfig):
    class sim(BaseSimConfig.sim):
        class physx(BaseSimConfig.sim.physx):
            # Caps the PhysX GPU aggregate/contact-pair capacity. The 2**24 default is sized
            # for 8000+ envs and by itself exhausts 4 GB of VRAM. 2**22 (~4.2M pairs) is 4x
            # smaller yet still covers a dense arena (e.g. 192 envs x 150 built bars needs
            # ~1.14M aggregate pairs); 2**20 overflowed there -> PhysX "increase
            # totalAggregatePairsCapacity" + a warp illegal-memory crash.
            max_gpu_contact_pairs = 2**22
            # 10x -> 2x PhysX default buffers keeps the GPU allocation within 4 GB.
            default_buffer_size_multiplier = 2
