# Record Envelope v2 — live-generation feasibility, 14 September 2026

**Verdict: `LIVE_GENERIC_VALIDATION_NOT_APPLICABLE`.**

Question: Does Record Envelope v2 fit a real task-independent artifact producer in this repository?

## What was tried

The candidate was the generic renderer dataset exporter (`tools/export_renderer_dataset.py`), chosen because it is already audited, already task-independent, and touches no policy, task or controller.

It still works: the exporter ran and wrote a `generic_renderer_export_v1` receipt with 1 frame(s).
Its own receipt has no checkpoint field and no density field, which is the point.

## Why the envelope does not fit it

| Field | Availability | Reason |
| --- | --- | --- |
| `schema_version` | AVAILABLE | constant declared by the contract |
| `schema_sha256` | AVAILABLE | sha256 of the schema file |
| `run_id` | AVAILABLE | uuid4 minted per run |
| `source_git_commit` | AVAILABLE | git rev-parse HEAD in the repository |
| `source_dirty` | AVAILABLE | git status, observed |
| `source_files_sha256` | AVAILABLE | sha256 of the producer's own sources |
| `seed` | AVAILABLE | the exporter takes --material-seed and --light-seed; either is a real integer seed, though the contract has one slot for two seeds |
| `density_bars` | NOT_APPLICABLE | obstacle-bar density is a simulator-task concept; a static renderer export has no bars and no honest value |
| `checkpoint_path` | NOT_APPLICABLE | a generic exporter loads no policy checkpoint |
| `checkpoint_sha256` | NOT_APPLICABLE | no checkpoint exists to hash |
| `config_path` | NOT_APPLICABLE | the exporter is configured by command-line arguments and writes its configuration into its own receipt afterwards; there is no pre-existing config file to hash |
| `config_sha256` | NOT_APPLICABLE | no config file exists to hash |
| `created_at_utc` | AVAILABLE | clock at capture |
| `record_count` | AVAILABLE | number of exported frames |
| `records` | AVAILABLE | one row per exported frame |

Five of the fifteen required fields have no honest source in a task-independent producer. `density_bars` and the two checkpoint fields are simulator-task concepts; the two config fields assume a pre-existing configuration file, and this exporter is configured by command-line arguments.

## The producer invented nothing

Asked for an envelope without those values, the producer refused every time rather than filling a default:

| Attempt | Result | Code |
| --- | --- | --- |
| density_bars unknown | REFUSED | `INVALID_DENSITY_BARS` |
| checkpoint absent | REFUSED | `MISSING_CHECKPOINT_PATH` |
| config absent | REFUSED | `MISSING_CONFIG_PATH` |

That is the correct behaviour and it is why this check ends in `NOT_APPLICABLE` rather than in a passing envelope. Setting `density_bars = 0`, pointing `checkpoint_path` at a URDF, or hashing the exporter's own source as a "config" would each have produced a green result and a false record.

## What was not done

No value was fabricated for the unavailable fields, and the envelope was not modified to accommodate the producer. A v2.1 with an optional task section is a design change and needs its own decision, not a side effect of this check.

The historical 128-row artifact is untouched, and its two verdicts stand: byte integrity `MATCH`, metadata contract `INVALID_FOR_DECLARED_CONTRACT`.
