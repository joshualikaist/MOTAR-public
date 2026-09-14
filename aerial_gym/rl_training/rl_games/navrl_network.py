"""NavRL-style CNN feature extractor as an rl_games custom network ("navrl_cnn").

Mirrors the feature extractor in reference/NavRL isaac-training/training/scripts/ppo.py:
the 36x4 LiDAR range scan passes through a 3-layer conv stack into a 128-d embedding
(LayerNorm), which is concatenated with the 12-d internal state (S_int: goal-frame
direction/distance/velocity + body-frame heading terms for learned yaw) and
fused by a [256, 256] ELU MLP. Unlike the flat-MLP baseline (ppo_navrl.yaml), the conv
stack preserves the scan's spatial structure — which directions hold nearby obstacles —
instead of flattening it away, which is NavRL's lever for collision avoidance.

Differences from NavRL kept deliberately (recorded in RESEARCH_PLAN/WORKLOG): Gaussian
policy with fixed log-sigma via rl_games' continuous_a2c_logstd model instead of a Beta
head, and a shared actor/critic trunk.

Observation layout (156) as produced by navrl_task:
    [ S_int (12) | LiDAR scan, row-major (vbeams=4 x hbeams=36 -> 144) ]

Select it from YAML with:
    params:
      network:
        name: navrl_cnn
"""

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder

S_INT_DIM = 12  # (b) 8 nav dims + goal_bearing_body(2) + vel_body_xy(2); obs = 12 + 36*4 = 156
LIDAR_VBEAMS = 4
LIDAR_HBEAMS = 36
CNN_EMBED_DIM = 128
FUSE_UNITS = (256, 256)


class NavRLCNNBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return NavRLCNNBuilder.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            NetworkBuilder.BaseNetwork.__init__(self)

            expected = S_INT_DIM + LIDAR_VBEAMS * LIDAR_HBEAMS
            if input_shape[0] != expected:
                raise ValueError(
                    f"navrl_cnn expects obs dim {expected} "
                    f"(S_int {S_INT_DIM} + {LIDAR_VBEAMS}x{LIDAR_HBEAMS} scan), got {input_shape[0]}"
                )

            # NavRL LiDAR feature extractor. Input (N, 1, 36, 4): width = 36 horizontal
            # beams, height = 4 vertical beams. Downsamples horizontally twice
            # (36 -> 18 -> 9) and vertically once (4 -> 4 -> 2): 16*9*2 = 288 -> 128.
            self.lidar_cnn = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=(5, 3), padding=(2, 1)),
                nn.ELU(),
                nn.Conv2d(4, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
                nn.ELU(),
                nn.Flatten(),
                nn.Linear(16 * (LIDAR_HBEAMS // 4) * (LIDAR_VBEAMS // 2), CNN_EMBED_DIM),
                nn.LayerNorm(CNN_EMBED_DIM),
            )

            fuse_layers = []
            in_dim = CNN_EMBED_DIM + S_INT_DIM
            for units in FUSE_UNITS:
                fuse_layers += [nn.Linear(in_dim, units), nn.ELU()]
                in_dim = units
            self.fuse = nn.Sequential(*fuse_layers)

            self.mu = nn.Linear(in_dim, actions_num)
            self.sigma = nn.Parameter(
                torch.zeros(actions_num, dtype=torch.float32), requires_grad=True
            )
            self.value = nn.Linear(in_dim, 1)

            # NavRL initializes its actor/critic output layers orthogonally with a small
            # gain; sigma starts at 0 (std = 1) like the flat-MLP baseline config.
            nn.init.orthogonal_(self.mu.weight, 0.01)
            nn.init.constant_(self.mu.bias, 0.0)

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states")
            s_int = obs[:, :S_INT_DIM]
            scan = obs[:, S_INT_DIM:].view(-1, LIDAR_VBEAMS, LIDAR_HBEAMS)
            scan = scan.permute(0, 2, 1).unsqueeze(1).contiguous()  # (N, 1, 36, 4)
            feature = self.fuse(torch.cat([self.lidar_cnn(scan), s_int], dim=1))
            mu = self.mu(feature)
            value = self.value(feature)
            return mu, mu * 0 + self.sigma, value, states
