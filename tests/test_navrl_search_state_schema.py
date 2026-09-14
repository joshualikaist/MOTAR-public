import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


PROBE = r'''
import importlib.util, json, pathlib, sys, types
import torch
import torch.nn as nn

root = pathlib.Path(sys.argv[1])
task = root / "aerial_gym/task/navrl_task"
for name in ("aerial_gym", "aerial_gym.task", "aerial_gym.task.navrl_task",
             "aerial_gym.rl_training", "aerial_gym.rl_training.rl_games"):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

load("aerial_gym.task.navrl_task.navrl_corridor", task / "navrl_corridor.py")
load("aerial_gym.task.navrl_task.navrl_search_state", task / "navrl_search_state.py")
perception = load("aerial_gym.task.navrl_task.navrl_perception", task / "navrl_perception.py")

rl_games = types.ModuleType("rl_games")
algos = types.ModuleType("rl_games.algos_torch")
builder = types.ModuleType("rl_games.algos_torch.network_builder")
class NetworkBuilder:
    class BaseNetwork(nn.Module):
        def __init__(self):
            super().__init__()
builder.NetworkBuilder = NetworkBuilder
sys.modules["rl_games"] = rl_games
sys.modules["rl_games.algos_torch"] = algos
sys.modules["rl_games.algos_torch.network_builder"] = builder
network = load("s1_network_probe", root / "aerial_gym/rl_training/rl_games/navrl_transformer_network.py")
model = network.NavRLTransformerBuilder.Network(
    {}, actions_num=4, input_shape=(perception.STRUCTURED_OBS_DIM,)
)
with torch.no_grad():
    mu, log_std, value, _ = model({"obs": torch.zeros(2, perception.STRUCTURED_OBS_DIM)})
print(json.dumps({"obs": perception.STRUCTURED_OBS_DIM, "tokens": network.NUM_TOKENS,
                  "search_dim": perception.SEARCH_DIM, "mu": list(mu.shape),
                  "value": list(value.shape)}))
'''


class SearchStateSchemaTest(unittest.TestCase):
    def probe(self, arm, geofence):
        env = dict(os.environ)
        env.update(
            NAVRL_LIDAR_HBEAMS="72",
            NAVRL_LIDAR_VBEAMS="4",
            NAVRL_MAX_OBSTACLES="8",
            NAVRL_CORRIDOR_TOKENS="0",
            NAVRL_SEARCH_STATE=arm,
            NAVRL_GEOFENCE_ACTOR="1" if geofence else "0",
        )
        return subprocess.run(
            [sys.executable, "-c", PROBE, str(ROOT)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_arm_dimensions_and_tokens(self):
        expected = {
            "off": {"obs": 898, "tokens": 17, "search_dim": 0, "mu": [2, 4], "value": [2, 1]},
            "geofence": {"obs": 906, "tokens": 18, "search_dim": 0, "mu": [2, 4], "value": [2, 1]},
            "coverage": {"obs": 934, "tokens": 19, "search_dim": 28, "mu": [2, 4], "value": [2, 1]},
            "belief": {"obs": 959, "tokens": 19, "search_dim": 53, "mu": [2, 4], "value": [2, 1]},
        }
        for arm, contract in expected.items():
            with self.subTest(arm=arm):
                completed = self.probe(arm, arm != "off")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(json.loads(completed.stdout), contract)

    def test_nonoff_without_geofence_fails_closed(self):
        completed = self.probe("coverage", False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires NAVRL_GEOFENCE_ACTOR=1", completed.stderr)

    def test_unknown_state_fails_closed(self):
        completed = self.probe("telepathy", True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("NAVRL_SEARCH_STATE must be", completed.stderr)


if __name__ == "__main__":
    unittest.main()
