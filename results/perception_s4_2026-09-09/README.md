# S4 frozen P7c-v2 NPS test evaluation

Completed once under `docs/plans/perception_s4_rc_execution_2026-09-09.md`.
1,725 frames from 8 source videos; checkpoint SHA `9b1f52299eea83f89ed95e8bc7bcf26c0b51140d388887a83937486cc6113cca`.

| Metric | Result |
|---|---:|
| Hit (any-GT IoU >= 0.3) | 1021 / 1725 = 59.19% |
| False lock among selected | 210 / 1231 = 17.06% |
| No lock | 494 / 1725 = 28.64% |
| Utility | (1021 - 210) / 1725 = 0.470145 |
| Reacquisition | 68 completed, 2 right-censored |
| Reacquisition mean / maximum | 0.9864 / 19.3448 s |

Validation utility was 0.68554; test utility is lower by 0.21540. These are different videos,
so this is a generalization warning, not a paired effect estimate. No post-test retuning is done.
NPS test had prior P5 use; this is the first frozen P7c-v2 final evaluation, not a pristine
never-observed holdout. P8/P9 remain validation-calibrated and completed P10 results are unchanged.

`report.json`, `receipt.json` and `predictions.jsonl.gz` retain metrics, runtime and provenance.
The motion cache is external at `detector_runs/results/perception_s4_2026-09-09/` and its hash is
bound in the evaluation receipt. Metric range, true identity-switch and real-hardware claims
remain unsupported by these annotations.
