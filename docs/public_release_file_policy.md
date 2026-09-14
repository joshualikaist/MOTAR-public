# Public release file policy

What a clean public snapshot of this repository contains, what it does not, and why. The research
repository keeps everything, including its full history; the release is a snapshot of one commit's
content with a short, written set of exclusions.

Built by [`tools/build_public_release_candidate.py`](../tools/build_public_release_candidate.py).

## Principle

```text
release = current approved public tree - explicitly excluded artifacts
```

A file leaves the release only under a rule written here. Nothing is dropped because it looked
unnecessary, and nothing is dropped to make a check pass.

## Included

| Category | Why |
| --- | --- |
| Source code (`aerial_gym/`, `tools/`, `tests/`) | The work itself and the checks that constrain it |
| Configuration needed for reproduction (`configs/`, `requirements*.txt`, `pyproject.toml`) | A reader cannot reproduce without them |
| `README.md`, `LICENSE`, `THIRD_PARTY_LICENSES.md`, `NOTICE.md`, `CITATION.cff` | Release landing and legal surface |
| `docs/REPRODUCIBILITY.md`, `docs/RESEARCH_EVIDENCE_INDEX.md`, renderer CPU quickstart | The reproduction path, self-contained |
| Public figures (`docs/assets/paper/…`) | Referenced by the paper page and the result pages |
| `results/` evidence: summaries, receipts, manifests, preregistrations, logs | The evidence the claims rest on, including negative and VOID results |
| `results/MANIFEST.json` | Index of every result by identity and provenance |

Negative, VOID, INCONCLUSIVE and superseded results stay in. A release that keeps only the
successful runs would misrepresent the record.

## Excluded

| Category | Rule | Reason |
| --- | --- | --- |
| ETH ds5 derived images and the review ZIP | 14 content hashes in [`public_release_denylist.json`](public_release_denylist.json) | CC BY-NC-SA 4.0 upstream; redistribution not established |
| Git history (`.git/`) | The snapshot is exported with `git archive`; no object database is copied | Historical blobs include the assets above |
| `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.DS_Store` | Path globs | Regenerated locally; not content |
| Symlinks already dangling at the source commit | Verified broken in the research tree too | A published tree should not ship pointers to nothing |

Raw datasets were never tracked, so nothing had to be removed for them: `datasets/` does not exist
in the repository.

## Hash, not filename

The exclusion check is by **content hash**, over every regular file and every member of every
archive in the candidate. A denylisted image renamed, or hidden inside a ZIP, is still a
redistribution and is still caught.

The converse matters as much: **a hash or a filename appearing as text inside a receipt is
provenance and stays.** A result that records which dataset file it consumed is evidence, and
deleting that reference would damage the record without removing a single byte of the dataset.

> Historical provenance references may name external dataset files. The corresponding dataset bytes
> are not distributed in this release.

## Symlinks

In-repository symlinks are rewritten from absolute to relative so they survive a clone. Only the
pointer changes; no regular file's bytes are touched. A link pointing outside the repository would
be removed rather than repointed, and the build reports either case.

## Absolute paths

Historical launchers, preregistrations and receipts contain the absolute interpreter and repository
paths of the machine that produced them. Those are provenance and are **not** rewritten. What must
be free of them is the public reproduction path — `README.md`, `docs/REPRODUCIBILITY.md`, the CPU
quickstart, the requirement files and the public tools — and the build report records that check.
