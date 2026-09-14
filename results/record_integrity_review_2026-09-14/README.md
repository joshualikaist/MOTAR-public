# Record-envelope review — 14 September 2026

Scope: read-only checks of a preserved JSON log and synthetic CPU serialization/validation cost.
No policy, simulator evaluation, acquisition/terminal analysis, training or recorder-hook change.
This is not the 12-cell diagnostic sweep. Existing results are never repaired or relabelled here.

Baseline: `74515779d27926b899836b84b280950d7231cd88`, HEAD and origin/main identical, clean tree.
Session start: 2026-09-14 01:17:58 UTC. Preliminary read/parse of the 361,392-byte log: about 1.6 ms.
Initial total-work estimate: 20–35 minutes, including implementation, checks and documentation;
not an estimate of policy-evaluation time.

The envelope contract was written after inspecting the existing log's schema/metadata. It is a
software audit contract, **not prospective experimental preregistration**. It requires complete
non-null identifiers and a literal SHA-256, but does not check domain-specific outcomes or rules.
Its expected count is the historical 128-record smoke artifact, not a new 2,049-episode cell.

## Result: byte integrity and metadata completeness are different

The preserved file is
[`on_episode_forensics.json`](../task_diagnostics_invariance_2026-09-14/on_episode_forensics.json).
No operational metrics or failure classifications were computed from it.

| Check | Result |
| --- | --- |
| SHA-256 versus baseline Git blob | MATCH, `1475b0c226e9dd546938687e575dcfe22d45af0c7603f6d6b5fde2e92278870a` |
| Recorded / expected rows | 128 / 128 |
| Duplicate `(env_index, episode_index)` | 0 |
| Non-finite numerical leaves | 0 |
| Required envelope keys absent | 0 |
| `seed` null | 128/128 records; also null in context |
| `density_bars` null | 128/128 records; also null in context |
| `checkpoint_sha256` invalid format | 128/128 records; also invalid in context |
| File and contract unchanged across audit | Yes |

The `checkpoint_sha256` value is a file path rather than a 64-hex SHA-256. The checker does **not**
follow that path, read a checkpoint, or infer/fill the missing seed/density. These three envelope
defects are not repaired in historical data. Other outcome/telemetry fields are outside the
declared six-field envelope validation, except that every numeric leaf is checked for finiteness.

[`envelope_audit.json`](envelope_audit.json) reports **`INVALID_FOR_DECLARED_CONTRACT`**:
258 null-required-value issues and 129 invalid-hash-format issues including document context.
This does not mean corrupted file bytes, does not retroactively label the original smoke as a
VOID cell, and does not overturn its separately recorded trajectory-digest comparison.
It means this file, by itself, lacks the complete reproducibility metadata required by this audit.

The check took **7.35 ms**, including read, strict parse, SHA, metadata validation and Git-blob
comparison. This is one wall-time observation, not a stable performance estimate.

## Synthetic CPU cost

[`synthetic_benchmark.json`](synthetic_benchmark.json): deterministic generic five-field rows,
not task telemetry and not policy execution. Three timing batches of five samples per mode in
one process; one warmup per mode/batch. Aggregate mean, median and P95 and all 225 raw timings
are retained. The total benchmark wall time was **21.999 s**. No GPU was used.

Means in milliseconds:

| Synthetic rows | Serialize | SHA only | Strict parse | Hash + parse + validate | Serialize + hash + parse + validate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.102 | 0.008 | 0.087 | 0.481 | 0.593 |
| 10,000 | 8.527 | 0.581 | 6.653 | 36.911 | 45.402 |
| 100,000 | 87.561 | 5.857 | 73.613 | 382.439 | 468.670 |

The synthetic payload sizes are 19,087 / 1,499,032 / 15,089,209 bytes; the real 128-row artifact
is much wider (361,392 bytes). Equal row counts therefore do not imply comparable workload.
End-to-end synthetic P95 is 0.602 / 45.914 / 472.571 ms respectively. Independent arithmetic
recalculation of all means/medians/P95 from 225 raw samples agreed within 1e-9 ms.

Timings end on function return, include allocations but exclude object destruction/GC and disk
writes/fsync. Separate tracemalloc audit peaks were 0.073 / 5.565 / 55.706 MiB, excluding the
pre-existing fixture/serialized input; RSS is a cumulative Linux process high-water mark.
Repeated timing batches are not independent process or seed experiments. **This does not estimate
live instrumentation overhead, a 12-cell evaluation duration or a renderer/policy throughput.**

## Regression scope and preserved harness failures

Only an explicit public/independent-renderer/metadata allowlist was executed, with
`CUDA_VISIBLE_DEVICES=''`, `PYTHONNOUSERSITE=1`, OMP/MKL threads 2. No `test_navrl*` suites,
simulator rollouts or evaluation launchers were selected. No existing test was removed, skipped,
weakened or edited. This is **not the full 1,784-test repository suite**.

The allowlist contains 358 tests, including 27 newly added generic tests. Per selection:

| Selection | Tests |
| --- | ---: |
| Record-envelope corruption/CLI tests | 23 |
| Synthetic benchmark / allowlist tests | 4 |
| Independent renderer tests | 201 |
| RC tests | 78 |
| Public tooling | 18 |
| CPU install evidence | 3 |
| Runtime fingerprint | 8 |
| Repository claim boundaries | 12 |
| Research overview | 11 |

First attempt: all 358 tests passed, skip 0; public schema check could not import `jsonschema`
from the historical aerialgym environment. This is a harness/environment failure, not a JSON
schema verdict. The first log, summary and runner snapshot are preserved in
[`regression_attempt1/`](regression_attempt1/).

The existing isolated documentation environment was then located and checked: jsonschema 3.2.0,
cffconvert 2.0.0, PyYAML 6.0.2, matching `requirements-public-validation.txt`; `pip check` passed.
**No packages were installed or changed.** Attempt 2 passed the schema check and all 358 tests,
but the newly added citation command incorrectly used `python -m cffconvert`; that package has
no `__main__`. Its failure and source snapshot remain in
[`regression_attempt2/`](regression_attempt2/). The harness was corrected to the installed CLI
entrypoint. Snapshots are historical source evidence, not standalone launchers.

Final attempt and timing are recorded in [`regression_attempt3/summary.json`](regression_attempt3/summary.json).
The unchanged first/second attempt records are not overwritten by the final run.

Final result: **358 tests passed, zero failures/errors/skips**, plus public-document schema,
citation schema and both Node site checks PASS. Final subprocess group wall time: **14.311 s**.
The two previous groups took 14.067 and 14.302 s. Repeating the same suite does not increase
the unique test count. The full repository suite and the 12-cell evaluation remain NOT_RUN.

## Reproduce without a policy

Run from the repository root with the recorded Python environment; outputs must be new files or
directories. The first command deliberately returns exit **2** for the preserved metadata defects.

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' /home/fair/miniconda3/envs/aerialgym/bin/python -B \
  tools/check_record_integrity.py \
  --input results/task_diagnostics_invariance_2026-09-14/on_episode_forensics.json \
  --contract docs/record_envelope_audit_2026-09-14.json \
  --git-ref 74515779d27926b899836b84b280950d7231cd88 --output /tmp/record-audit-new.json

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' /home/fair/miniconda3/envs/aerialgym/bin/python -B \
  tools/benchmark_record_integrity.py --output /tmp/record-benchmark-new.json

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' /home/fair/miniconda3/envs/aerialgym/bin/python -B \
  tools/run_record_validation_checks.py \
  --docs-python /tmp/motar-release-validation-6rTnrm/venv/bin/python --output /tmp/record-regression-new
```

The temporary documentation interpreter is a local path, not a portable dependency. On a new
machine use a separate environment satisfying `requirements-public-validation.txt` and pass its
interpreter explicitly. Do not install those packages into a historical research environment.

The audit tool is standard-library-only. It rejects duplicate JSON keys, nonstandard numerical
tokens, non-finite leaves, missing/null/wrong-type envelope fields, invalid SHA-256 strings,
duplicate identifiers, context mismatches and record-count mismatches. It does not silently
repair rows; only issue counts and up to 40 locations are output, never the field values.
Conditional telemetry rules, taxonomy quality, acquisition/capture statistics and task decisions
are not implemented or assessed.

## Handoff and provenance

The original renderer contract, task/instrumentation code, label thresholds and historical results
were not modified. New utilities and tests are local, uncommitted changes; no push or deployment
was performed. `receipt.json` pins the exact working-file sources and result files. The baseline
Git SHA alone does not contain this new implementation. `summary.json` records the wall interval
from the accepted-scope start to final evidence sealing, including implementation and documentation;
it is distinct from the individual measured script durations above.
