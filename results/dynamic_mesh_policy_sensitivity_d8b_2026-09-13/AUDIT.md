# D8b archival closure — integrity review

This is a post-execution audit of existing records, not a new experiment or a replacement
preregistration. No GPU cell is rerun. D8b and any proposed perception-adaptation lineage remain
separate. The [original preregistration](../../docs/preregistration_dynamic_mesh_policy_sensitivity_d8b_2026-09-13.md)
has **eight**, not six, integrity gates. It requires **at least** 2,049 actual episodes per cell;
all nine stored cells happen to contain exactly 2,049 (18,441 in total).

## Review before finalization

| Original §4 gate | Finding | Evidence / qualification |
|---|---|---|
| 1. D8-A identity and technical gate | PASS | Receipt SHA `92ce1c6a9d5f419de7b873af327c006b92246be9d9ba0da8c656dc9fe659e7ee`; recorded `TECHNICAL_GO`. |
| 2. Ancestor chain and identical clean runtime | PASS | Preregistration `0aba9816d789c37e7d0ef8874e0a66d3d501299b` precedes runtime `1e0eed8b963ed3ccdae93cc2e9ea67d9f9ddbaab`. Nine receipts bind one clean manifest. All 370 snapshot files match both their hashes and the runtime Git blobs. |
| 3. Frozen artifact identity | PASS | All source/evaluated checkpoint hashes and nine local snapshot hashes equal `197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e`. No external detector snapshot. |
| 4. Episode accounting | PASS | Every request and actual count is 2,049; nonnegative integer outcome counts sum to actual; rates independently equal counts divided by actual. |
| 5. Held-fixed conditions | PASS | The full stored evaluation contract varies only by seed. Runtime condition differences are seed, nonce and the four treatment identity fields. Receipt differences are seed, nonce, timestamps, output paths and output hashes. Registered values also checked individually. |
| 6. Effective treatment and actor boundary | PASS — artifact plus source review | Three analytic cells attest no attached treatment; six mesh cells attest the pinned v3 asset, mode and exact constants. Logs retain actor/critic dimensions 898/906. Source review finds no newly added normal/face/material/raw-hit/occlusion field in actor assembly. This is not a separate dynamic information-flow proof. |
| 7. Python/ninja binding; no blanket force | PASS — recorded environment scope | Pinned launcher source binds PATH to the selected Python prefix, selects its adjacent ninja and rejects blanket force. All logs identify the selected Python; the shared, hash-bound environment file records Python 3.8.20 and ninja package 1.13.0. This does **not** establish an execution-time SHA of the ninja binary (see below). |
| 8. Complete registered grid | PASS | Exact 3×3 Cartesian product, nine distinct nonces, no missing/extra cell directory. Result/log/environment/manifest receipt hashes checked before finalization. |

The original runner's seven `summary.integrity` booleans are **not** an implementation of all
eight preregistration gates. `audit.py` supplies additional count, rate, log, Git-blob,
cross-cell configuration and duplicate checks. Gate 6's actor boundary and gate 7's launcher
behavior also rely on review of the pinned source, not on those summary booleans alone.

### Environment and interpretation qualifications

- Stored inactive settings are dropout `0.3`, RGB noise `0.015`, depth noise `0.02`.
  `perception_perturb=false` and effective dropout `0.0` are recorded throughout. The pinned
  task passes that perturbation flag as the perception noise `training` argument. These are
  inactive settings, not nonzero noise treatments and not literally zero-valued configuration.
- Logs contain `160.0`/`90.0` integer-parse warnings. They explicitly fall back to the registered
  defaults 160/90. This is a launcher formatting warning, not evidence for a different resolution.
- Finalization reads Python/ninja path/version/binary hash **at finalization time**. That generated
  `summary.provenance.launcher` must not be described as a nine-cell contemporaneous binary
  fingerprint. Historical evidence is the committed launcher, logs and shared package inventory.
  Full per-process TF32/cuDNN/device flags are not present in these cells; cross-machine bitwise
  reproduction is not established.
- Episode identities and trajectories are not paired across arms. The registered uncertainty
  unit is the three paired evaluation seeds, not 18,441 independent experimental replicates.
- This campaign contains no distractors. It cannot establish shortcut reduction, identity
  correctness, or a cause of any policy response. No downstream bottleneck diagnosis is added.

## Retention and reproduction

Retain raw cell JSON, original receipts, CSV summaries, original small logs, the shared source
manifest and original package inventory. `README.md` / `summary.json` are outputs of the original
finalizer; `audit.py` independently checks the primary contrast without importing that runner.
Original raw evidence is not edited.

The original logs contain logger trailing spaces and the original CSV uses CRLF. Narrow
`.gitattributes` rules preserve those bytes and recognize those formats during whitespace checks;
code, JSON and documentation retain the normal whitespace checks.

The finalizer's Markdown template emitted one trailing space. After finalization, only that
space was removed from the template and generated README to pass `git diff --check`.
No JSON value, analysis rule, GPU source or raw cell changed. The recorded runtime/source hashes
still refer to the original campaign revision, not this presentation-only follow-up.

Exclude the nine duplicate checkpoint copies and redundant source snapshots from Git under the
existing evidence policy. They remain on disk. Absolute per-cell source-manifest symlinks also
remain local; the portable audit uses the shared manifest through a relative path instead.
No cache, checkpoint or log is deleted.

After a full-history clone, the following is CPU-only, read-only and needs no simulator,
checkpoint or source-snapshot directory:

```bash
python3 -B results/dynamic_mesh_policy_sensitivity_d8b_2026-09-13/audit.py
```

The source copies are exactly recoverable from Git revision
`1e0eed8b963ed3ccdae93cc2e9ea67d9f9ddbaab`; the manifest pins each path/size/SHA.
For example, to recover source files in a **new temporary directory**, without executing them:

```bash
d8b_sources=$(mktemp -d)
git archive 1e0eed8b963ed3ccdae93cc2e9ea67d9f9ddbaab \
  aerial_gym resources/robots resources/models/environment_assets/objects tools/renderer_validation \
  | tar -x -C "$d8b_sources"
```

The checkpoint is **not** promised to be regenerable from a training command. Its original local
file and copies are retained; a recipient wanting to validate the actual weights needs that
hash-matching artifact separately. Portable archive verification checks the checkpoint identity
recorded in receipts, whereas `--local` additionally hashes the stored checkpoint copies.

Original full `verify` (not `evaluate`) also requires the local copies and aliases:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  /home/fair/miniconda3/envs/aerialgym/bin/python -B \
  tools/run_dynamic_mesh_policy_sensitivity_d8b.py verify
```

No D8c preregistration, detector/association training, P8/P9 replacement, or new policy evaluation
is authorized or performed by this archival closure. Proposed closed-loop perception integration
is not marked complete. Independent generic renderer validation remains a separate possible task.
