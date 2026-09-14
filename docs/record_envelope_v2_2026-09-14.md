# Record Envelope v2 — 14 September 2026

A provenance contract for **future** artifact producers. Nothing historical is repaired by it.

## Why the v1 historical artifact failed

The 128-row smoke artifact
[`on_episode_forensics.json`](../results/task_diagnostics_invariance_2026-09-14/on_episode_forensics.json)
is byte-identical to its committed blob and still fails its declared contract
([audit](../results/record_integrity_review_2026-09-14/README.md)):

| Check | Result |
| --- | --- |
| SHA-256 versus the Git blob | `MATCH` (`1475b0c2…78870a`) |
| Declared metadata contract | **`INVALID_FOR_DECLARED_CONTRACT`** |
| `seed`, `density_bars` | null in the context and in all 128 rows |
| `checkpoint_sha256` | a filesystem **path**, not a digest |

Byte integrity and metadata completeness are different properties, and that file has the first
without the second. Both verdicts stand. The file is not edited, not backfilled, and no value is
inferred for it from a neighbouring receipt or launcher — an inferred number is not a recorded one.

## What v2 requires

Schema: [`record_envelope_v2.json`](record_envelope_v2.json), producer:
[`tools/record_envelope_v2.py`](../tools/record_envelope_v2.py).

```text
schema_version   schema_sha256      run_id
source_git_commit  source_dirty     source_files_sha256
seed               density_bars
checkpoint_path    checkpoint_sha256
config_path        config_sha256
created_at_utc     record_count     records
```

* Every `*_sha256` is a real content digest matching `^[0-9a-f]{64}$`, computed by reading the
  file. A path, a filename or a symbolic name in such a field is a hard error.
* `seed`, `density_bars`, `checkpoint_sha256` and `source_git_commit` can never be null. A producer
  that cannot obtain them **fails closed** before any artifact exists, rather than writing a null
  and continuing.
* `run_id` is a UUIDv4. It identifies the run; it never carries meaning about the result.
* `source_dirty` is observed from `git status`, never assumed clean.
* Where a row repeats a context identifier, the two must agree, or the artifact is rejected with
  `CONTEXT_RECORD_MISMATCH`. v2 keeps the context as the single source and does not require rows to
  repeat it; the v1 consumers that read per-row identifiers are untouched, which is why v2 is a
  separate format rather than a migration.

## Path is not identity

`checkpoint_path` and `checkpoint_sha256` are different fields with different jobs, and so are
`config_path` and `config_sha256`. The path is machine-local information, recorded to help a human
find the file. The digest is the identity. Two copies of the same bytes at different paths have the
same identity; the same path with one byte changed does not, and the producer detects that.

## Preflight

`preflight(...)` runs **before** an artifact exists and refuses unless all of it holds: seed and
density are non-negative integers, checkpoint and config exist as regular files and are hashed from
their bytes, the repository resolves and its HEAD commit is a real commit object, the source files
are hashed, the schema file parses and matches the required field set, and Git state has not moved
during the capture. It returns an immutable capture; a caller mutating its own copy cannot alter it.

## Postflight

`postflight(...)` runs **after** the bytes are on disk: strict JSON parse, schema validation, the
externally supplied expected record count, unique record ids, finite numeric leaves, context
consistency, the checkpoint/config/source digests re-verified, and the artifact's own SHA-256
re-read afterwards so a file changed during checking cannot pass. `publish(...)` writes a receipt
only when postflight passes; on failure it keeps the raw artifact and writes a failure record
instead of a success receipt.

## Status boundary

```text
Record Envelope v2
producer_unit_validation   = PASS
live_generation_validation = NOT_APPLICABLE
```

The producer passes its unit tests against synthetic files in temporary directories. Live
generation was then **attempted** on the generic renderer dataset exporter and found **not
applicable**: five of the fifteen required fields — `density_bars`, `checkpoint_path`,
`checkpoint_sha256`, `config_path`, `config_sha256` — have no honest source in a task-independent
producer, and the producer refused each of them (`INVALID_DENSITY_BARS`, `MISSING_CHECKPOINT_PATH`,
`MISSING_CONFIG_PATH`) instead of filling a default. Evidence:
[live-generation feasibility](../results/record_envelope_v2_live_2026-09-14/README.md).

**This is a schema-design limitation, recorded rather than worked around.** Envelope v2 inherited
its task shape from the v1 audit of a simulator artifact. Making the task section optional would be
a v2.1, which is a deliberate design decision and is not taken as a side effect of a validation
run. Nothing here is `METADATA_FIXED`, and `live_generation_validation` becomes `PASS` only when a
producer that can supply every required field truthfully is validated end to end.

## Using the checker

```bash
# v1, unchanged: historical audit against a declared contract
python tools/check_record_integrity.py --input <artifact> --contract <contract.json> --output <report.json>

# v2: generic producer validation, structural by default
python tools/check_record_integrity.py --input <artifact> --envelope-v2 \
    --expected-records <n> [--verify-files] --output <report.json>
```

The two modes are mutually exclusive on the command line. A v1 artifact is never re-read under the
v2 schema, so no historical `INVALID_FOR_DECLARED_CONTRACT` verdict can be converted into a pass by
adopting v2. A regression test pins that.
