"""Exact read-only network for legacy 305-D NavRL semantic-vision checkpoints.

New runs must use ``navrl_transformer``. This builder exists so the interactive application can
replay already-trained July 2026 baselines without pretending they are Transformer checkpoints.
Parameter names and shapes intentionally match the original checkpoint ABI.
"""

import torch
import torch.nn as nn
from rl_games.algos_torch.network_builder import NetworkBuilder


STATE_DIM = 17
VBEAMS = 4
HBEAMS = 36
OBS_DIM = STATE_DIM + 2 * VBEAMS * HBEAMS


class NavRLVisionLegacyBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return NavRLVisionLegacyBuilder.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            NetworkBuilder.BaseNetwork.__init__(self)
            if int(input_shape[0]) != OBS_DIM:
                raise ValueError("legacy NavRL vision expects obs dim %d" % OBS_DIM)

            self.scan_cnn = nn.Sequential(
                nn.Conv2d(2, 8, kernel_size=(5, 3), padding=(2, 1)),
                nn.ELU(),
                nn.Conv2d(8, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
                nn.ELU(),
                nn.Flatten(),
                nn.Linear(16 * (HBEAMS // 4) * (VBEAMS // 2), 128),
                nn.LayerNorm(128),
            )
            self.fuse = nn.Sequential(
                nn.Linear(128 + STATE_DIM, 256),
                nn.ELU(),
                nn.Linear(256, 256),
                nn.ELU(),
            )
            self.mu = nn.Linear(256, actions_num)
            self.sigma = nn.Parameter(torch.zeros(actions_num), requires_grad=True)
            self.value = nn.Linear(256, 1)

        def is_rnn(self):
            return False

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            state = obs[:, :STATE_DIM]
            n_scan = VBEAMS * HBEAMS
            ranges = obs[:, STATE_DIM : STATE_DIM + n_scan].view(-1, VBEAMS, HBEAMS)
            target = obs[:, STATE_DIM + n_scan :].view(-1, VBEAMS, HBEAMS)
            scan = torch.stack((ranges, target), dim=1).permute(0, 1, 3, 2).contiguous()
            latent = self.fuse(torch.cat((self.scan_cnn(scan), state), dim=1))
            mu = self.mu(latent)
            return mu, mu * 0.0 + self.sigma, self.value(latent), None
