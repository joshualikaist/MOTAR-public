"""NavRL++-Target structured temporal Transformer policy.

Base token layout (17): [CLS] + static geometry + 5 obstacle-history + 5 robot-history +
5 target-track-history. Opt-in corridor, mapped-geofence, and explicit search observations append
one token each (coverage and belief share the single search token).
Raw RGB-D, raw point clouds, simulator semantics, and GT target state are not part of the input.
"""

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder

from aerial_gym.task.navrl_task.navrl_corridor import CORRIDOR_DIM
from aerial_gym.task.navrl_task.navrl_perception import (
    CORRIDOR_TOKENS,
    GEOFENCE_ACTOR,
    GEOFENCE_DIM,
    HBEAMS,
    MAX_OBSTACLES,
    OBSTACLE_DIM,
    OBSTACLE_HISTORY,
    ROBOT_DIM,
    ROBOT_HISTORY,
    SEARCH_DIM,
    STATIC_DIM,
    STRUCTURED_OBS_DIM,
    VBEAMS,
    TARGET_DIM,
    TARGET_HISTORY,
)


EMBED_DIM = 64
# CLS + static-scan + one token per history step of obstacles / robot / target. Note this does NOT
# depend on MAX_OBSTACLES: raising the obstacle capacity widens each obstacle token's input
# (MAX_OBSTACLES * OBSTACLE_DIM) rather than adding tokens. When corridor tokens are enabled
# (NAVRL_CORRIDOR_TOKENS > 0), ONE additional token carries all corridor slots, appended LAST so
# position-embedding rows 0..16 keep their trained meaning across a corridor warm-start. Geofence
# is a new fresh-policy-only token and is appended after corridor when both are enabled.
NUM_TOKENS = (
    17
    + (1 if CORRIDOR_TOKENS > 0 else 0)
    + (1 if GEOFENCE_ACTOR else 0)
    + (1 if SEARCH_DIM > 0 else 0)
)


def _static_encoder_flat_dim(vbeams, hbeams, channels=16):
    """Flattened size of the static-scan CNN, whose two strided convs halve each axis.

    Derived rather than hard-coded so the scan resolution stays a single source of truth: a stale
    constant here would surface as an opaque shape error at network build time.
    """
    stride1 = (2, 1)  # (vertical, horizontal)
    stride2 = (2, 2)
    v = -(-vbeams // stride1[0])
    h = -(-hbeams // stride1[1])
    v = -(-v // stride2[0])
    h = -(-h // stride2[1])
    return channels * v * h


class NavRLTransformerBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return NavRLTransformerBuilder.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            NetworkBuilder.BaseNetwork.__init__(self)
            if int(input_shape[0]) != STRUCTURED_OBS_DIM:
                raise ValueError(
                    "navrl_transformer expects structured obs dim %d, got %d"
                    % (STRUCTURED_OBS_DIM, input_shape[0])
                )

            self.static_encoder = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, padding=1),
                nn.ELU(),
                nn.Conv2d(4, 16, kernel_size=3, stride=(2, 1), padding=1),
                nn.ELU(),
                nn.Conv2d(16, 16, kernel_size=3, stride=(2, 2), padding=1),
                nn.ELU(),
                nn.Flatten(),
                nn.Linear(_static_encoder_flat_dim(VBEAMS, HBEAMS), 128),
                nn.ELU(),
                nn.Linear(128, EMBED_DIM),
            )
            self.obstacle_project = nn.Sequential(
                nn.Linear(MAX_OBSTACLES * OBSTACLE_DIM, 128), nn.ELU(), nn.Linear(128, EMBED_DIM)
            )
            self.robot_project = nn.Sequential(
                nn.Linear(ROBOT_DIM, 128), nn.ELU(), nn.Linear(128, EMBED_DIM)
            )
            self.target_project = nn.Sequential(
                nn.Linear(TARGET_DIM, 128), nn.ELU(), nn.Linear(128, EMBED_DIM)
            )
            if CORRIDOR_TOKENS > 0:
                self.corridor_project = nn.Sequential(
                    nn.Linear(CORRIDOR_TOKENS * CORRIDOR_DIM, 128),
                    nn.ELU(),
                    nn.Linear(128, EMBED_DIM),
                )
            if GEOFENCE_ACTOR:
                self.geofence_project = nn.Sequential(
                    nn.Linear(GEOFENCE_DIM, 64), nn.ELU(), nn.Linear(64, EMBED_DIM)
                )
            if SEARCH_DIM > 0:
                self.search_project = nn.Sequential(
                    nn.Linear(SEARCH_DIM, 128), nn.ELU(), nn.Linear(128, EMBED_DIM)
                )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED_DIM))
            self.position_embedding = nn.Parameter(torch.zeros(1, NUM_TOKENS, EMBED_DIM))
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.position_embedding, std=0.02)

            layer = nn.TransformerEncoderLayer(
                d_model=EMBED_DIM,
                nhead=4,
                dim_feedforward=128,
                # 0.0, deliberately: rl_games collects rollouts in eval() (masks off) but runs the
                # 8 minibatch updates in train() (masks re-sampled every forward), so any nonzero
                # dropout injects mask noise into new_log_prob while old_log_prob stays clean --
                # corrupting the PPO ratio/KL and misleading the KL-adaptive LR schedule. PPO has
                # no fixed dataset to overfit; generalization comes from env randomization, and
                # sensor-level dropout (perception.detection_dropout_prob) is a separate knob.
                dropout=0.0,
                activation="relu",
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=4)
            self.final_norm = nn.LayerNorm(EMBED_DIM)
            self.actor = nn.Sequential(
                nn.Linear(EMBED_DIM, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU()
            )
            self.critic = nn.Sequential(
                nn.Linear(EMBED_DIM, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU()
            )
            self.mu = nn.Linear(256, actions_num)
            self.sigma = nn.Parameter(torch.zeros(actions_num), requires_grad=True)
            self.value = nn.Linear(256, 1)
            nn.init.orthogonal_(self.mu.weight, 0.01)
            nn.init.constant_(self.mu.bias, 0.0)

        def is_rnn(self):
            return False

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            batch = obs.shape[0]
            offset = 0
            static = obs[:, offset : offset + STATIC_DIM].view(batch, 1, VBEAMS, HBEAMS)
            offset += STATIC_DIM
            obstacle = obs[
                :, offset : offset + OBSTACLE_HISTORY * MAX_OBSTACLES * OBSTACLE_DIM
            ].view(batch, OBSTACLE_HISTORY, MAX_OBSTACLES * OBSTACLE_DIM)
            offset += OBSTACLE_HISTORY * MAX_OBSTACLES * OBSTACLE_DIM
            robot = obs[:, offset : offset + ROBOT_HISTORY * ROBOT_DIM].view(
                batch, ROBOT_HISTORY, ROBOT_DIM
            )
            offset += ROBOT_HISTORY * ROBOT_DIM
            target = obs[:, offset : offset + TARGET_HISTORY * TARGET_DIM].view(
                batch, TARGET_HISTORY, TARGET_DIM
            )
            offset += TARGET_HISTORY * TARGET_DIM

            token_list = [
                self.cls_token.expand(batch, -1, -1),
                self.static_encoder(static).unsqueeze(1),
                self.obstacle_project(obstacle),
                self.robot_project(robot),
                self.target_project(target),
            ]
            if CORRIDOR_TOKENS > 0:
                corridor = obs[:, offset : offset + CORRIDOR_TOKENS * CORRIDOR_DIM]
                token_list.append(self.corridor_project(corridor).unsqueeze(1))
                offset += CORRIDOR_TOKENS * CORRIDOR_DIM
            if GEOFENCE_ACTOR:
                geofence = obs[:, offset : offset + GEOFENCE_DIM]
                token_list.append(self.geofence_project(geofence).unsqueeze(1))
                offset += GEOFENCE_DIM
            if SEARCH_DIM > 0:
                search = obs[:, offset : offset + SEARCH_DIM]
                token_list.append(self.search_project(search).unsqueeze(1))
                offset += SEARCH_DIM
            if offset != STRUCTURED_OBS_DIM:
                raise RuntimeError(
                    "Transformer observation parse drift: %d != %d"
                    % (offset, STRUCTURED_OBS_DIM)
                )
            tokens = torch.cat(token_list, dim=1)
            if tokens.shape[1] != NUM_TOKENS:
                raise RuntimeError("Transformer token schema drift: %d" % tokens.shape[1])
            tokens = tokens + self.position_embedding
            encoded = self.transformer(tokens.transpose(0, 1)).transpose(0, 1)
            cls = self.final_norm(encoded[:, 0])
            mu = self.mu(self.actor(cls))
            value = self.value(self.critic(cls))
            # Clamp log-std. With fixed_sigma=True + entropy_coef>0 the log-std parameter has NO upper
            # bound and the entropy bonus inflates it without limit -- over ~5000 epochs sigma drifted
            # 1 -> ~13 (ppo/entropy 7 -> 16) while capture/value stayed healthy, then at sigma~13 the
            # PPO log-prob/gradient overflowed to NaN in a SINGLE step (5173: a_loss NaN while c_loss,
            # kl, explained_variance were all fine) -> weights NaN -> hover collapse. Both 0.005 and
            # 0.003 entropy_coef died this way (only the drift RATE differed). Bounding log-std to
            # [-5, 0.4] (sigma in [0.007, 1.49], around the healthy early level) removes the runaway.
            log_std = (mu * 0.0 + self.sigma).clamp(-5.0, 0.4)
            return mu, log_std, value, None
