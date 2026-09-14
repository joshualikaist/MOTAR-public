# Reproducibility — where to start

A short entry point. It tells you what runs without a GPU, what needs one, what cannot be shipped at
all, and how to find out what produced any given result.

## Environment

| | |
| --- | --- |
| Verified host | Linux x86-64 (Ubuntu 20.04 family), Python 3.8 |
| GPU work | NVIDIA RTX 3070 (8 GB), CUDA 12.x driver, Isaac Gym Preview 4, Warp 1.0.0, Torch 2.4.1+cu121 |
| CPU-only work | No GPU, no Isaac Gym, no detector, no policy |

Every result receipt records its own runtime (interpreter, Torch, CUDA/cuDNN, device) — the receipt
is authoritative for that result, not this table.

## Install

* **CPU renderer profile** (recommended first step, no GPU):
  [`docs/renderer_cpu_quickstart_2026-09-12.md`](renderer_cpu_quickstart_2026-09-12.md) with
  [`requirements-renderer-cpu.txt`](../requirements-renderer-cpu.txt). It installs the independent
  graphics prototype only: no root package, no Isaac Gym, no detector, no policy or controller.
* **Public document/schema validation profile**:
  [`requirements-public-validation.txt`](../requirements-public-validation.txt).
* **Full simulator profile**: [`requirements.txt`](../requirements.txt) plus Isaac Gym Preview 4,
  which is not redistributable and must be obtained from NVIDIA.
* **Environment check**: `python tools/motar_doctor.py` reports what is present and what is missing
  rather than guessing.

## Renderer-only validation (no GPU)

```bash
python tools/validate_renderer_export.py --output <new directory>
python -m unittest discover -s tests -p 'test_renderer*.py'
python -m unittest discover -s tests -p 'test_rc_*.py'
```

The renderer characterization figures can be rebuilt from committed results without re-rendering:

```bash
python tools/render_rc_figures.py
python tools/render_renderer_contract_figures.py
```

## Main tests

```bash
python -m unittest discover -s tests            # whole repository
python tools/check_public_docs.py               # declared public surfaces
```

GPU-dependent tests skip themselves when no CUDA device is present; the suite reports the skips
rather than silently passing.

## Result manifest

```bash
python tools/build_result_manifest.py --print-issues
```

[`results/MANIFEST.json`](../results/MANIFEST.json) indexes every result directory with its status
string, receipt, summary, preregistration, source manifest and their hashes. It records identity and
provenance only — it never extracts or interprets a research quantity. Directories created before
the current convention are marked `legacy_exception` and their missing files are recorded as
history, not as defects.

## Evidence index

[`docs/RESEARCH_EVIDENCE_INDEX.md`](RESEARCH_EVIDENCE_INDEX.md) is the one-page map: seven axes,
each with its question, its strongest supported claim, its status and its known limitation.
Negative, inconclusive and blocked results are listed there, not hidden.

## Artifact provenance for new work

New artifact producers are governed by
[Record Envelope v2](record_envelope_v2_2026-09-14.md). It fails closed: a producer that cannot
obtain a seed, a density, a real content hash or a git commit refuses before writing anything.
Its current status is `producer_unit_validation = PASS`,
`live_generation_validation = NOT_APPLICABLE`.

## Known unavailable data

These are referenced by results but are **not** in the repository and mostly cannot be
redistributed. A result that depends on them is reproducible from its recorded form, not from raw
data.

| Data | Availability |
| --- | --- |
| ETH drone-tracking ds5 (video, extracted frames, review JPEG/ZIP) | Not redistributed in current tree; obtain separately under CC BY-NC-SA 4.0: [external acquisition and commands](external_data/ETH_DS5.md) |
| Det-Fly, NPS-Drones, detenv | Not redistributed; upstream terms `NEEDS_CONFIRMATION` |
| Isaac Gym Preview 4 | Not redistributable; obtain from NVIDIA |
| Training checkpoints (`*.pth`) | Large binaries; only those already tracked are present |

See [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md) for the full inventory and the items
still marked `NEEDS_CONFIRMATION`.

ETH consumers already accept `--dataset /path/to/eth-source` and `--video /path/to/cam0.mp4`.
Run `tools/check_eth_ds5_local_data.py` with those flags before local preparation. Keep newly
generated review material under ignored `artifacts/private/eth_ds5/`; the linked external-data
contract gives source-checked extraction and analysis commands. No dataset download or analysis
is performed by the release checks. The original ten-image review pack cannot be regenerated
exactly because its complete centre-coordinate input was already unavailable.

## Known hardware requirements

* The GPU evaluations in this repository were run on a single RTX 3070 (8 GB). A 128-environment
  density cell of 2,049 episodes takes roughly 10–12 minutes on that machine.
* The CPU renderer path needs no GPU and no Isaac Gym.
* Nothing here has been validated on real flight hardware.
