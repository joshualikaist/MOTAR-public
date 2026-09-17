# MOTAR

MOTAR: Moving Object Tracking And Rendezvous.
Reinforcement Learning for UAV Pursuit in Random Obstacle Fields.

![Research sources, analysis and recorded outputs](docs/assets/paper/system-overview-block-diagram.svg)

## Overview

MOTAR is a research repository for measured perception error, dense-obstacle navigation and
safety-filter diagnosis. The public quick start runs an **independent static graphics renderer**:
it does not load a detector, flight policy or simulator task.

The historical project studied UAV **interception**, with metrics named `capture` and `crash`.
Those experiments, meanings, paths and checkpoints are preserved, not renamed as evidence of a
different capability. “Rendezvous” describes the public research framing, not a validated contact
system or a change to historical success criteria.

## Research Questions

- How can measured perception uncertainty be documented without overstating generalization?
- Which assumptions limit existing safety-filter results?
- How can geometry, material and lighting be tested independently in synthetic images?
- What evidence is needed before claiming that an experiment is reproducible?

## Scope and Limitations

This is **simulation-only** research on moving-target rendezvous: tracking and approach in a
simulated dense-obstacle arena. It makes **no real-flight validation claim** and provides no
deployment instructions. Independent renderer tests do not certify simulator integration, target
identity or physical safety. Data and code have different licensing boundaries.

The bounded claims travel with the results and are not softened anywhere in this repository:

| Result | Status |
| --- | --- |
| P2 held-out, D1 adaptation | **FAIL**; P3 full budget **BLOCKED** |
| P10 readaptation | **INCONCLUSIVE** — a net adaptation benefit is **not established** |
| S4 held-out generalization | **warning stands** |
| E3-P attitude reliability | **ATTITUDE_NOT_RELIABLE**; decomposition BLOCKED |
| E3-S size-to-range proxy | usable, on **one** flight only |
| D8b frozen-policy sensitivity | **MATERIAL_LOSS** |
| Renderer Contract v1 versus D8b | **causality NOT_TESTED** |

Numbers belong with their receipts, not on a landing page. The one-page map of what each track
established, with its figures and its limits, is the
[research evidence index](docs/RESEARCH_EVIDENCE_INDEX.md). Historical, negative and withdrawn
findings remain visible in [Verification](VERIFICATION.md).

## External datasets

No raw dataset is distributed with this repository. ETH ds5 derived images are **excluded** from the
public release under CC BY-NC-SA 4.0; Det-Fly, NPS-Drones and detenv are referenced by tooling only.
Results that depend on them are reproducible from their recorded form, not from raw data. See
[reproducibility](docs/REPRODUCIBILITY.md) and [third-party notices](THIRD_PARTY_LICENSES.md).

## System Overview

The figure maps research inputs to analyses and recorded outputs; it is not a deployed closed loop.
[All nine diagrams and captions](docs/assets/paper/) and their
[SVG/PNG/PDF package](docs/assets/paper/motar-paper-block-diagrams.zip) remain available.
Detailed results and figures live in the [results overview](docs/results_overview_2026-09-12.md).

## Key Components

- Independent static renderer: procedural geometry, G-buffers and material/lighting variations.
- Evidence utilities: source hashes, runtime fingerprints and immutable result records.
- Historical research code and measurements: retained with experiment-specific contracts.
- Target-behavior axis: static, CV and obstacle-aware scripted implementations; reactive/self-play
  stages remain planned and are not reported as results.
- Public validation: environment inventory, documentation checks and CPU-only CI.

## Repository Structure

| Path | Purpose |
|---|---|
| `tools/renderer_validation/` | Independent static graphics components |
| `tools/` | Export, benchmark, evidence and maintenance utilities |
| `tests/` | Unit, packaging, evidence and documentation checks |
| `docs/` | Installation, contracts, public checklist and research records |
| `results/` | Historical measurements and provenance, not generic training data |
| `aerial_gym/`, `resources/` | Historical simulator code and assets |

## Requirements

The supported independent CPU profile is **Linux x86-64, Python 3.8, Git**, with a new virtual
environment. It does not need Isaac Gym or a GPU. Python 3.8 is a legacy compatibility constraint,
not a recommendation for unrelated new applications. See the
[CPU profile limitations](docs/renderer_cpu_quickstart_2026-09-12.md).

The historical simulator environment is separate and has proprietary/external prerequisites.
The CPU profile does not certify its installation.

## Installation

This repository is a **clean public release snapshot**. It does not include the original research
Git history. Historical receipts and provenance are preserved, but tests that verify a specific
research ancestor commit skip here with a stated reason. Source snapshot, exclusions and
external-data provenance are in [RELEASE_PROVENANCE.md](RELEASE_PROVENANCE.md). The CPU public
reproduction path is this README and [reproducibility](docs/REPRODUCIBILITY.md).

```bash
git clone https://github.com/joshualikaist/MOTAR-public.git
cd MOTAR-public
```

Follow [isolated CPU installation](docs/renderer_cpu_quickstart_2026-09-12.md).
Do not install the root requirements into that environment or replace a historical research environment.

## Quick Start

After activating the isolated CPU environment, run from the repository root:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' python -B tools/motar_doctor.py --profile renderer-cpu
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' python -B tools/run_renderer_validation.py \
  --device cpu --seed 0 --num-scenes 1 --width 160 --height 120 --frames 2 --output /tmp/motar-smoke-new
```

The output directory must not already exist. The smoke produces generic box images and debug
buffers, **not an experiment verdict**. See [export and benchmark tooling](docs/renderer_public_tools.md).

## Training and Evaluation

There is **no training step in the public quick start**. Historical commands and their authority
are preserved in [Operations](OPERATIONS.md); a recorded command is not a new execution approval.
This release work does not modify rewards, observation schemas or checkpoints.

## Reproducing Experiments

Use the exact environment, source revision, data split and hashes named by each result.
[CPU evidence reproduction](docs/public_evidence_reproduction_2026-09-12.md) and
[renderer installation evidence](results/renderer_cpu_install_2026-09-12/README.md) distinguish
stored-result verification from a fresh rendering run. Missing evidence stays missing.

## Documentation

Start with [reproducibility](docs/REPRODUCIBILITY.md) for what runs without a GPU, and the
[research evidence index](docs/RESEARCH_EVIDENCE_INDEX.md) for what the results do and do not say.


- [Documentation index](docs/README.md)
- [Results by track](docs/results_overview_2026-09-12.md)
- [Matched-baseline and published-system relation](docs/relation_to_published_systems_2026-09-16.md)
- [Target behavior ladder and current-motion audit](docs/target_behavior_ladder_2026-09-16.md)
- [Current verification and limitations](VERIFICATION.md)
- [Worklog](WORKLOG.md)
- [Public release checklist](docs/PUBLIC_RELEASE_CHECKLIST.md)
- [Public release audit](docs/public_release_audit_2026-09-12.md)
- [Master roadmap and blocked boundaries](docs/plans/moving_target_rendezvous_master_plan_2026-09-12.md)
- [Machine-readable status](docs/status_manifest.json) · [Research site](docs/status/)

## Citation

Use [CITATION.cff](CITATION.cff) for the software citation. No published MOTAR paper or author
identifier is invented. Cite the upstream software and datasets used by the relevant experiment
separately.

## License and Third-Party Materials

Original project code is offered under the repository's [BSD-3-Clause notice](LICENSE), subject
to retained upstream notices. This does **not** apply to every data file or dependency.
[Third-party inventory](THIRD_PARTY_LICENSES.md) identifies separate terms and unresolved items.
ETH ds5 source and derived review images/ZIPs are **not redistributed in the current tree**.
Obtain ETH ds5 separately from its original distributor under **CC BY-NC-SA 4.0**, not BSD:
[external acquisition and reproduction](docs/external_data/ETH_DS5.md).
Analysis code, numerical evidence and historical provenance remain available. The original
research repository's historical Git history contains earlier review assets. The public
MOTAR-public history does not include those objects; see
[release status](docs/public_release_status_2026-09-14.md).

## Acknowledgements

MOTAR derives from [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) and
acknowledges NavRL, rl_games, NVIDIA Warp, urdfpy, trimesh, and the cited dataset authors.
Acknowledgement does not imply endorsement. See the third-party inventory for use and redistribution details.
