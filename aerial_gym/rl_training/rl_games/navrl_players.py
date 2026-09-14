"""rl_games player compatibility for bounded NavRL action models."""

import json
import math
import os
from pathlib import Path

import torch

from rl_games.algos_torch.players import PpoPlayerContinuous, rescale_actions
from rl_games.common.tr_helpers import unsqueeze_obs

from aerial_gym.rl_training.rl_games.ppo_update_safety import (
    mirror_navrl_actions,
    mirror_navrl_structured_observation,
)
from aerial_gym.rl_training.rl_games.navrl_mode_probe import (
    ModeProbeRecorder,
    build_probe_observations,
    model_outputs,
)


class NavRLPpoPlayerContinuous(PpoPlayerContinuous):
    """Use a model-provided bounded deterministic action when one is available.

    rl_games 1.6.5's continuous player always selects ``res_dict['mus']`` for deterministic
    evaluation.  For a tanh-squashed policy that tensor is the pre-tanh Gaussian mean (also needed
    for PPO's latent KL), so returning it directly evaluates a different controller.  Legacy models
    do not provide ``deterministic_actions`` and retain the stock behavior.
    """

    def __init__(self, params):
        super().__init__(params)
        self._reflection_mode = os.environ.get(
            "NAVRL_EVAL_REFLECTION_MODE", "original"
        ).strip().lower()
        if self._reflection_mode not in ("original", "conjugate"):
            raise ValueError(
                "NAVRL_EVAL_REFLECTION_MODE must be original or conjugate, got %r"
                % self._reflection_mode
            )
        self._reflection_diag_path = os.environ.get(
            "NAVRL_REFLECTION_DIAG_JSON", ""
        ).strip()
        self._reflection_diag = {
            "samples": 0,
            "abs_error_sum": [0.0, 0.0, 0.0, 0.0],
            "squared_error_sum": [0.0, 0.0, 0.0, 0.0],
            "max_abs_error": [0.0, 0.0, 0.0, 0.0],
            "original_signed_y_sum": 0.0,
            "mirrored_input_signed_y_sum": 0.0,
            "lateral_sign_mismatch": 0,
            "lateral_sign_comparable": 0,
        }
        self._mode_probe_path = os.environ.get("NAVRL_MODE_PROBE_JSON", "").strip()
        self._mode_probe = None
        if self._mode_probe_path:
            try:
                max_velocity = float(os.environ.get("NAVRL_MAX_VELOCITY", "2.5"))
                offset_deg = float(os.environ.get("NAVRL_MODE_PROBE_OFFSET_DEG", "5.0"))
            except ValueError as exc:
                raise ValueError("invalid NavRL mode-probe numeric setting") from exc
            self._mode_probe = ModeProbeRecorder(
                self._mode_probe_path,
                max_velocity_mps=max_velocity,
                offset_deg=offset_deg,
            )

        # A recurrent hidden state also has a reflection transform. That transform is not part of
        # the current diagnostic, so refusing it is safer than silently auditing only half of a
        # recurrent policy. The deployed ep24000 transformer is feed-forward.
        if (
            self._reflection_mode == "conjugate"
            or self._reflection_diag_path
            or self._mode_probe is not None
        ) and bool(
            getattr(self, "is_rnn", False)
        ):
            raise RuntimeError(
                "NavRL reflection/mode-probe evaluation supports feed-forward policies only"
            )

    @staticmethod
    def _model_action(res_dict, deterministic):
        sampled = res_dict["actions"]
        mean = res_dict.get("deterministic_actions", res_dict["mus"])
        return mean if deterministic else sampled

    def _record_reflection_pair(self, original_action, mirrored_input_action):
        expected = mirror_navrl_actions(original_action)
        error = mirrored_input_action - expected
        abs_error = error.abs()
        diag = self._reflection_diag
        n = int(original_action.shape[0])
        diag["samples"] += n
        for axis in range(4):
            diag["abs_error_sum"][axis] += float(abs_error[:, axis].sum().item())
            diag["squared_error_sum"][axis] += float(
                error[:, axis].square().sum().item()
            )
            diag["max_abs_error"][axis] = max(
                diag["max_abs_error"][axis], float(abs_error[:, axis].max().item())
            )
        diag["original_signed_y_sum"] += float(original_action[:, 1].sum().item())
        diag["mirrored_input_signed_y_sum"] += float(
            mirrored_input_action[:, 1].sum().item()
        )
        # Near-zero values have no meaningful sign. Count a disagreement only when both sides
        # carry at least 0.05 normalized lateral authority.
        comparable = (original_action[:, 1].abs() >= 0.05) & (
            mirrored_input_action[:, 1].abs() >= 0.05
        )
        mismatch = comparable & (
            torch.sign(mirrored_input_action[:, 1])
            != -torch.sign(original_action[:, 1])
        )
        diag["lateral_sign_comparable"] += int(comparable.sum().item())
        diag["lateral_sign_mismatch"] += int(mismatch.sum().item())

    def _write_reflection_diag(self):
        if not self._reflection_diag_path:
            return
        diag = self._reflection_diag
        samples = max(1, int(diag["samples"]))
        comparable = max(1, int(diag["lateral_sign_comparable"]))
        payload = {
            "schema_version": 1,
            "reflection_mode": self._reflection_mode,
            "samples": int(diag["samples"]),
            "comparison": "pi(M o) versus M pi(o), deterministic bounded actions",
            "mean_abs_error": [value / samples for value in diag["abs_error_sum"]],
            "rmse": [
                math.sqrt(value / samples) for value in diag["squared_error_sum"]
            ],
            "max_abs_error": list(diag["max_abs_error"]),
            "original_signed_mean_y": diag["original_signed_y_sum"] / samples,
            "mirrored_input_signed_mean_y": (
                diag["mirrored_input_signed_y_sum"] / samples
            ),
            "lateral_sign_threshold": 0.05,
            "lateral_sign_comparable": int(diag["lateral_sign_comparable"]),
            "lateral_sign_mismatch_rate": (
                diag["lateral_sign_mismatch"] / comparable
            ),
        }
        out = Path(self._reflection_diag_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(out)

    def run(self):
        try:
            return super().run()
        finally:
            self._write_reflection_diag()
            if self._mode_probe is not None:
                self._mode_probe.write()

    def _record_mode_probe(self, original_obs):
        if self._mode_probe is None:
            return
        fixtures, contract = build_probe_observations(
            original_obs, offset_deg=self._mode_probe.offset_deg
        )
        outputs = {}
        fork_devices = [original_obs.device.index] if original_obs.is_cuda else []
        # Side forwards must not consume random numbers used by environment reset or stochastic
        # rollout action selection.  They are diagnostic only and never replace current_action.
        with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
            for arm, fixture in fixtures.items():
                result = self.model(
                    {
                        "is_train": False,
                        "prev_actions": None,
                        "obs": self._preproc_obs(fixture),
                        "rnn_states": self.states,
                    }
                )
                outputs[arm] = model_outputs(result)
        self._mode_probe.record(outputs, contract)

    def get_action(self, obs, is_deterministic=False):
        if (self._reflection_mode == "conjugate" or self._reflection_diag_path) and not is_deterministic:
            raise RuntimeError("NavRL reflection audit requires deterministic action selection")
        if not self.has_batch_dimension:
            obs = unsqueeze_obs(obs)
        original_obs = self._preproc_obs(obs)
        original_input = {
            "is_train": False,
            "prev_actions": None,
            "obs": original_obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            original_result = self.model(original_input)
        self._record_mode_probe(original_obs)

        need_mirror = self._reflection_mode == "conjugate" or bool(
            self._reflection_diag_path
        )
        mirrored_result = None
        if need_mirror:
            mirrored_obs = self._preproc_obs(mirror_navrl_structured_observation(obs))
            mirrored_input = {
                "is_train": False,
                "prev_actions": None,
                "obs": mirrored_obs,
                "rnn_states": self.states,
            }
            # Most rl-games continuous models sample an action even when the player later selects
            # their deterministic mean. Preserve RNG state around the diagnostic second forward;
            # otherwise merely measuring symmetry changes later simulator resets and invalidates
            # the common-seed outcome comparison.
            fork_devices = []
            if mirrored_obs.is_cuda:
                fork_devices = [mirrored_obs.device.index]
            with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
                mirrored_result = self.model(mirrored_input)

            original_deterministic = self._model_action(original_result, True)
            mirrored_deterministic = self._model_action(mirrored_result, True)
            self._record_reflection_pair(
                original_deterministic, mirrored_deterministic
            )

        if self._reflection_mode == "conjugate":
            selected_result = mirrored_result
            current_action = mirror_navrl_actions(
                self._model_action(selected_result, is_deterministic)
            )
        else:
            selected_result = original_result
            current_action = self._model_action(selected_result, is_deterministic)
        self.states = selected_result["rnn_states"]
        if not self.has_batch_dimension:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        return current_action
