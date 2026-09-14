"""Sensor-only NavRL vision policy network ("navrl_vision") for the Phase-3 pivot.

Observation layout (1265) as produced by navrl_task in NAVRL_VISION mode:
    [ ego+camera-detector state (17)
      | LiDAR range (144) | LiDAR target mask (144)
      | forward camera obstacle depth (40x24=960) ]
LiDAR and camera images have independent CNN encoders. Their embeddings are fused with the 17-d
state by a [256, 256] ELU MLP. Thus both sensors contribute to perception: LiDAR observes obstacle
range and target semantics; camera observes obstacle depth and the detector's target evidence.

Optional recurrence for the POMDP (target occluded / out of FOV / beyond range): add an `rnn`
block to the network params (units/layers/name: lstm) and a `seq_length` to the PPO config.
The dones-masked sequence handling mirrors rl_games' built-in A2CBuilder (non-separate branch),
because rl_games auto-wires RNN ONLY for its built-in network -- a custom BaseNetwork must
override is_rnn()/get_default_rnn_state() and consume rnn_states/seq_length/dones itself.

The VALUE head here is the actor-side value; in the asymmetric setup (central_value_config in
the YAML) rl_games uses the separate privileged critic on obs['states'] for advantages, and
this head is unused. It is kept so the same network also runs without a central value.

Select from YAML with:
    params:
      network:
        name: navrl_vision
        rnn:            # optional (POMDP memory)
          name: lstm
          units: 128
          layers: 1
"""

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.common.layers.recurrent import GRUWithDones, LSTMWithDones

STATE_DIM = 17  # ego (9) + detector (8); must match task_config.vision
LIDAR_VBEAMS = 4
LIDAR_HBEAMS = 36
LIDAR_CHANNELS = 2  # nearest range + target semantic mask
CAMERA_WIDTH = 40
CAMERA_HEIGHT = 24
CNN_EMBED_DIM = 128
FUSE_UNITS = (256, 256)


class NavRLVisionBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return NavRLVisionBuilder.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            self.num_seqs = kwargs.pop("num_seqs", 1)
            NetworkBuilder.BaseNetwork.__init__(self)

            expected = (
                STATE_DIM
                + LIDAR_CHANNELS * LIDAR_VBEAMS * LIDAR_HBEAMS
                + CAMERA_WIDTH * CAMERA_HEIGHT
            )
            if input_shape[0] != expected:
                raise ValueError(
                    f"navrl_vision expects obs dim {expected} (state {STATE_DIM} + "
                    f"{LIDAR_CHANNELS}x{LIDAR_VBEAMS}x{LIDAR_HBEAMS} LiDAR + "
                    f"1x{CAMERA_HEIGHT}x{CAMERA_WIDTH} camera), got {input_shape[0]}"
                )

            rnn_cfg = params.get("rnn", None)
            self.has_rnn = rnn_cfg is not None
            self.rnn_units = int(rnn_cfg["units"]) if self.has_rnn else 0
            self.rnn_layers = int(rnn_cfg.get("layers", 1)) if self.has_rnn else 0
            self.rnn_name = str(rnn_cfg.get("name", "lstm")) if self.has_rnn else ""

            # LiDAR encoder: input (N, 2, 36, 4), downsample 36->18->9, 4->4->2.
            self.lidar_cnn = nn.Sequential(
                nn.Conv2d(LIDAR_CHANNELS, 8, kernel_size=(5, 3), padding=(2, 1)),
                nn.ELU(),
                nn.Conv2d(8, 16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)),
                nn.ELU(),
                nn.Flatten(),
                nn.Linear(16 * (LIDAR_HBEAMS // 4) * (LIDAR_VBEAMS // 2), CNN_EMBED_DIM),
                nn.LayerNorm(CNN_EMBED_DIM),
            )

            # Forward depth encoder: input (N, 1, 40, 24), downsample to 10x6.
            self.camera_cnn = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ELU(),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Flatten(),
                nn.Linear(16 * (CAMERA_WIDTH // 4) * (CAMERA_HEIGHT // 4), CNN_EMBED_DIM),
                nn.LayerNorm(CNN_EMBED_DIM),
            )

            fuse_layers = []
            in_dim = 2 * CNN_EMBED_DIM + STATE_DIM
            for units in FUSE_UNITS:
                fuse_layers += [nn.Linear(in_dim, units), nn.ELU()]
                in_dim = units
            self.fuse = nn.Sequential(*fuse_layers)

            head_dim = in_dim
            if self.has_rnn:
                if self.rnn_name == "lstm":
                    self.rnn = LSTMWithDones(
                        input_size=in_dim, hidden_size=self.rnn_units, num_layers=self.rnn_layers
                    )
                elif self.rnn_name == "gru":
                    self.rnn = GRUWithDones(
                        input_size=in_dim, hidden_size=self.rnn_units, num_layers=self.rnn_layers
                    )
                else:
                    raise ValueError(f"navrl_vision rnn name must be lstm|gru, got {self.rnn_name}")
                self.rnn_ln = nn.LayerNorm(self.rnn_units)
                head_dim = self.rnn_units

            self.mu = nn.Linear(head_dim, actions_num)
            self.sigma = nn.Parameter(
                torch.zeros(actions_num, dtype=torch.float32), requires_grad=True
            )
            self.value = nn.Linear(head_dim, 1)
            nn.init.orthogonal_(self.mu.weight, 0.01)
            nn.init.constant_(self.mu.bias, 0.0)

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None
            shape = (self.rnn_layers, self.num_seqs, self.rnn_units)
            if self.rnn_name == "lstm":
                return (torch.zeros(shape), torch.zeros(shape))
            return (torch.zeros(shape),)

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)
            dones = obs_dict.get("dones", None)
            bptt_len = obs_dict.get("bptt_len", 0)

            state = obs[:, :STATE_DIM]
            n_lidar = LIDAR_VBEAMS * LIDAR_HBEAMS
            lidar_range = obs[:, STATE_DIM : STATE_DIM + n_lidar].view(
                -1, LIDAR_VBEAMS, LIDAR_HBEAMS
            )
            lidar_target = obs[:, STATE_DIM + n_lidar : STATE_DIM + 2 * n_lidar].view(
                -1, LIDAR_VBEAMS, LIDAR_HBEAMS
            )
            lidar_img = torch.stack([lidar_range, lidar_target], dim=1).permute(
                0, 1, 3, 2
            ).contiguous()
            camera_img = obs[:, STATE_DIM + 2 * n_lidar :].view(
                -1, 1, CAMERA_HEIGHT, CAMERA_WIDTH
            ).permute(0, 1, 3, 2).contiguous()

            out = self.fuse(
                torch.cat(
                    [self.lidar_cnn(lidar_img), self.camera_cnn(camera_img), state], dim=1
                )
            )

            if self.has_rnn:
                seq_length = obs_dict.get("seq_length", 1)
                batch = out.size(0)
                num_seqs = batch // seq_length
                out = out.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                if isinstance(states, list):
                    states = tuple(states)
                if self.rnn_name == "lstm" and states is not None and len(states) == 2:
                    states = (states[0], states[1])
                elif states is not None and len(states) == 1:
                    states = states[0]
                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1).contiguous().reshape(batch, -1)
                out = self.rnn_ln(out)
                if not isinstance(states, tuple):
                    states = (states,)

            mu = self.mu(out)
            value = self.value(out)
            return mu, mu * 0 + self.sigma, value, states
