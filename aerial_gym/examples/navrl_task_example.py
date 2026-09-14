"""Watch the NavRL Phase 1 environment (static obstacles + goal) with the viewer.

Opens an Isaac Gym window with a few drones in the obstacle field and drives them
with a constant "toward the goal" velocity command so you can see the scene, the
LiDAR-equipped quad, and the goal geometry. This does NOT use a trained policy --
to watch a trained policy, use the rl_games runner in --play mode (see README).

Run (with the aerialgym conda env active):
    python navrl_task_example.py
"""
import time

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
from aerial_gym.registry.task_registry import task_registry
import torch

if __name__ == "__main__":
    logger.print_example_message()
    env = task_registry.make_task("navrl_task", headless=False, num_envs=16)
    env.reset()

    # constant "go toward the goal" command: +x in the goal frame at full speed
    actions = torch.zeros(
        (env.task_config.num_envs, env.task_config.action_space_dim), device="cuda:0"
    )
    actions[:, 0] = 1.0

    logger.info(
        "\n\nWatching navrl_task with a constant toward-goal command (no trained policy).\n"
        "Close the viewer window or Ctrl-C to stop.\n"
    )
    start = time.time()
    with torch.no_grad():
        for i in range(100000):
            if i == 100:
                start = time.time()
            obs, reward, terminated, truncated, info = env.step(actions=actions)
    logger.info(f"steps/s: {(100000 - 100) / (time.time() - start):.1f}")
