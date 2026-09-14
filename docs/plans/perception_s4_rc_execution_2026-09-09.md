# S4 final selector evaluation and R-C replication

Authorized by the user after P10 archival; registered before these executions.

S4 evaluates only frozen `candidate_motion_transformer_v2/best.pt`, SHA
`9b1f52299eea83f89ed95e8bc7bcf26c0b51140d388887a83937486cc6113cca`,
using existing NPS test manifest/candidates in `detector_runs/results/nps_detfly_joint_final`.
Generate the missing test motion cache using unchanged corrected-v2 feature mathematics.
Record all input hashes, execution runtime and `test_used=true`. Report existing fixed any-GT
IoU 0.3 metrics, utility, hit, false lock, no lock, reacquisition and source-video count.
No fitting or selection follows this final evaluation. NPS test was previously used in P5;
Det-Fly test also has prior P3-F results. This is the final P7c evaluation, not a claim of a
never-before-observed dataset. S4 occurs after P10, a sequencing deviation from the original
P9/P10 document; it does not modify P8/P9 calibration or completed PPO results.

R-C follows confirmation_phase_plan_2026-09-06.md section 3 exactly: four continuations from
frozen ref5in D1 ep1900 to ep2900, training seeds 233 and 239, governor off/riskcap.
Use `train_navrl_v2_ref5in_a8_readapt.sh`; no simultaneous GPU work or commits between runs.
Evaluate each terminal checkpoint with riskcap/stopcap: eight cells, contract ref5in,
evaluation seed 521, 70 bars, 2049 requested episodes, actual counts as denominators.
Generate `docs/specs/grid_r2_d4_trainseed_rep.json` after resolving terminal checkpoint paths.
Report preregistered per-seed crash contrasts and their interaction, Wald 95% CI, and the
original 2/2, 1/2, 0/2 replication rule. Non-significance does not establish equivalence;
the T1 +-2 pp point-estimate criterion is only the historical replication rule.

Monitor each process and log for progress, numerical failure, OOM, traceback, and terminal
markers. Abort the dependent stage on failure and preserve its output for diagnosis. Verify
checkpoint epoch/hash/seed and cell outcome accounting before declaring completion.
