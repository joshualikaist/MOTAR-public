# NavRL v2 recovery postmortem

- status: **stopped-interrupted**
- generated: `2026-08-01T19:04:36.742981+00:00`
- canonical epochs: 9501–24010
- active process: False

## Density summary (epoch-weighted diagnostics)

| bars | epochs | range | capture mean | capture tail1000 | crash tail1000 | slope/1000 ep |
|---:|---:|---:|---:|---:|---:|---:|
| 130 | 866 | 9501–10366 | 0.7411 | 0.7411 | 0.2145 | 0.0322 |
| 145 | 1622 | 10367–11988 | 0.7107 | 0.7123 | 0.2313 | 0.0117 |
| 160 | 2033 | 11989–14021 | 0.6948 | 0.7008 | 0.2680 | 0.0158 |
| 175 | 1815 | 14022–15836 | 0.6814 | 0.6888 | 0.2991 | 0.0170 |
| 190 | 3264 | 15837–19100 | 0.6799 | 0.6930 | 0.3000 | 0.0144 |
| 205 | 4910 | 19101–24010 | 0.6815 | 0.6902 | 0.2922 | 0.0079 |

## Exact density gate windows

| result | bars | next | capture | episodes | threshold |
|---|---:|---:|---:|---:|---:|
| promoted | 130 | 145 | 0.747 | 16384 | 0.700 |
| held | 145 | — | 0.699 | 16384 | 0.700 |
| promoted | 145 | 160 | 0.723 | 16385 | 0.700 |
| held | 160 | — | 0.681 | 16384 | 0.700 |
| held | 160 | — | 0.695 | 16384 | 0.700 |
| held | 160 | — | 0.691 | 16386 | 0.700 |
| promoted | 160 | 175 | 0.708 | 16385 | 0.700 |
| held | 175 | — | 0.679 | 16384 | 0.700 |
| held | 175 | — | 0.661 | 16385 | 0.700 |
| held | 175 | — | 0.680 | 16385 | 0.700 |
| promoted | 175 | 190 | 0.700 | 16384 | 0.700 |
| held | 190 | — | 0.654 | 16384 | 0.700 |
| held | 190 | — | 0.666 | 16384 | 0.700 |
| held | 190 | — | 0.680 | 16384 | 0.700 |
| held | 190 | — | 0.683 | 16384 | 0.700 |
| held | 190 | — | 0.686 | 16385 | 0.700 |
| held | 190 | — | 0.682 | 16385 | 0.700 |
| promoted | 190 | 205 | 0.704 | 16384 | 0.700 |
| held | 205 | — | 0.647 | 16384 | 0.700 |
| held | 205 | — | 0.659 | 16385 | 0.700 |
| held | 205 | — | 0.670 | 16384 | 0.700 |
| held | 205 | — | 0.685 | 16385 | 0.700 |
| held | 205 | — | 0.693 | 16384 | 0.700 |
| held | 205 | — | 0.688 | 16384 | 0.700 |
| held | 205 | — | 0.691 | 16384 | 0.700 |
| held | 205 | — | 0.690 | 16385 | 0.700 |
| held | 205 | — | 0.694 | 16385 | 0.700 |
| held | 205 | — | 0.685 | 16386 | 0.700 |

## Durable checkpoint residual window

- checkpoint epoch: 24000
- bars: 205
- unfinished gate evidence: 11171/16327 (capture 0.6842)

| axis | bin | successes | episodes | capture |
|---|---|---:|---:|---:|
| speed | q0 | 693 | 991 | 0.6993 |
| speed | q1 | 3523 | 5051 | 0.6975 |
| speed | q2 | 3639 | 5183 | 0.7021 |
| speed | q3 | 3316 | 5102 | 0.6499 |
| distance | q0 | 2571 | 3305 | 0.7779 |
| distance | q1 | 3232 | 4385 | 0.7371 |
| distance | q2 | 3008 | 4466 | 0.6735 |
| distance | q3 | 2360 | 4171 | 0.5658 |
| pattern | cv | 5509 | 8250 | 0.6678 |
| pattern | waypoint | 5662 | 8077 | 0.7010 |
| pattern | circle | 0 | 0 | — |

## Latest diagnostics

- action: `{"epoch": 24000, "bars": 205, "task_input_oob": [0.0, 0.0, 0.0, 0.0], "exec_edge98": [0.231, 0.0692, 0.0013, 0.0], "signed_y": 0.752, "positive_y": 0.939, "negative_y": 0.045, "delta_y": 0.175, "sign_flip_y": 0.038}`
- motion: `{"epoch": 24000, "bars": 205, "speed_m_s": 2.372, "command_m_s": 2.937, "low_speed": 0.0318, "commanded_stall": 0.0315}`
- crash: `{"epoch": 24000, "bars": 205, "bar_contact": 0.939, "mean_x_m": 19.5, "mean_steps": 109.0, "below": 0.058, "above": 0.0, "oob": 0.003, "n_crash": 603}`
- barprobe: `{"epoch": 24000, "bars": 205, "n": 566, "bars_range": 49.9, "bars_fov": 33.8, "occupied_bins": 52.5, "hit_fov": 0.905, "hit_token": 0.885, "hit_token_given_fov": 0.895, "tokens": 8.0, "associated": 3.9, "unique": 3.8, "duplicate": 0.1, "center_offset_m": 0.35, "cross_track_m": 0.25, "radial_gap_m": 0.18}`
- strata: `{"epoch": 24001, "bars": 205, "speed": {"q0": {"rate": 0.7, "episodes": 994}, "q1": {"rate": 0.698, "episodes": 5067}, "q2": {"rate": 0.703, "episodes": 5200}, "q3": {"rate": 0.651, "episodes": 5125}}, "distance": {"q0": {"rate": 0.779, "episodes": 3315}, "q1": {"rate": 0.737, "episodes": 4400}, "q2": {"rate": 0.675, "episodes": 4486}, "q3": {"rate": 0.566, "episodes": 4185}}, "pattern": {"cv": {"rate": 0.668, "episodes": 8275}, "waypoint": {"rate": 0.702, "episodes": 8111}, "circle": {"rate": null, "episodes": 0}}, "gate": "pass", "reason": "diagnostic-only"}`

## PPO/policy health

- `ppo/kl`: last 0.00240245, tail500 0.0023611, range [0.000190263, 0.00851531]
- `ppo/behavior_kl_audit_max`: last 0.00283127, tail500 0.00519892, range [0.000725544, 0.0132231]
- `ppo/behavior_kl_sample_max`: last 0.0453512, tail500 0.0662332, range [0.00927627, 1.00354]
- `ppo/entropy`: last -6.88854, tail500 -7.04289, range [-9.52635, -5.94106]
- `ppo/explained_variance`: last 0.776477, tail500 0.789055, range [0.674499, 0.926052]
- `ppo/learning_rate`: last 5e-06, tail500 5e-06, range [5e-06, 5e-06]
- `ppo/epoch_rollback`: last 0, tail500 0, range [0, 0]
- `ppo/epoch_rollback_total`: last 0, tail500 0, range [0, 0]
- `ppo/epoch_rollback_streak`: last 0, tail500 0, range [0, 0]
- `ppo/kl_skipped_minibatches`: last 0, tail500 0, range [0, 0]
- `policy_action/raw_oob_x`: last 0, tail500 0, range [0, 0]
- `policy_action/raw_oob_y`: last 0, tail500 0, range [0, 0]
- `policy_action/raw_oob_z`: last 0, tail500 0, range [0, 0]
- `policy_action/raw_oob_yaw`: last 0, tail500 0, range [0, 0]
- `policy_action/edge99_x`: last 0.117676, tail500 0.0935479, range [0.0285645, 0.472656]
- `policy_action/edge99_y`: last 0.0170898, tail500 0.0296216, range [0.00952148, 0.35791]
- `policy_action/signed_y`: last 0.74823, tail500 0.757954, range [0.374904, 0.812517]
- `policy_action/positive_y`: last 0.932617, tail500 0.93415, range [0.696777, 0.961182]
- `policy_action/negative_y`: last 0.0505371, tail500 0.0497656, range [0.0246582, 0.274414]

## Interpretation limits

- epoch_metrics rates are epoch-weighted because per-epoch termination counts are not recorded there
- promotion windows are exact episode-count statistics and take precedence over epoch means
- a final causal decision requires held-out evaluation and the v2 geometry audit
- signed-y is not a chirality verdict without target-bearing and mirrored-layout conditioning
