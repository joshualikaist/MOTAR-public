# Release provenance

This directory is a **snapshot**, not a clone: the content of one commit of the MOTAR research
repository, and none of its history.

| | |
| --- | --- |
| Source research repository | MOTAR |
| Source snapshot commit | `95053dd231fc6b9628d6166ef9ccf3f3d5c82815` |
| Source tree dirty at export | false |
| Release construction | fresh `git archive` export; no inherited Git history |
| Files / symlinks | 4863 / 293 |
| Bytes | 511,804,021 |
| Build verdict | **CLEAN** |

A commit here has a different SHA from the research commit above. That is expected: the two
repositories share content, not history. The line that ties them together is
`source_research_commit`, recorded here and in `RELEASE_BUILD_REPORT.json`.

## External dataset policy

ETH ds5 derived assets are **excluded**: 14 content hashes
(CC BY-NC-SA 4.0) are listed in `docs/public_release_denylist.json`, and the build
scanned every regular file and every member of every archive against them.

```text
denylist matches: 0
files scanned:    4863
archives opened:  4
```

No raw dataset of any kind is distributed here. Det-Fly, NPS-Drones and detenv appear only as
tooling and documentation; their data was never tracked.

## Historical evidence

Receipts, summaries and preregistrations keep their original text, including the names and hashes of
external dataset files they consumed.

> Historical provenance references may name external dataset files. The corresponding dataset bytes
> are not distributed in this release.

Historical launchers and receipts also contain the absolute paths of the machine that produced them.
Those are provenance and are not rewritten. The public reproduction path - `README.md`,
`docs/REPRODUCIBILITY.md`, the CPU quickstart, the requirement files and the public tools - contains
none.

## Transformations applied to the snapshot

| Change | Count | Why |
| --- | ---: | --- |
| Absolute in-repository symlinks made relative | 272 | An absolute link to the author's machine breaks on any clone |
| Symlinks dangling at the source commit removed | 3 | Already broken in the research tree; a published tree should not ship pointers to nothing |
| Path-rule exclusions (caches) | 0 | Regenerated locally; not content |

No regular file's bytes were modified. The rule behind each change is in
`docs/public_release_file_policy.md`.

## Result index

`results/RELEASE_MANIFEST.json` indexes 198 result entries for this
snapshot, against 201 in the research tree. Three result directories
exist there and in no commit - `.gitignore` calls them immutable local receipts with raw traces - so
a snapshot cannot contain them. The research manifest names them under `untracked_results`.

This snapshot has no research history, so it cannot date its own results; the legacy classification
is inherited from the research manifest, whose hash is recorded inside
`results/RELEASE_MANIFEST.json`.

## Checks that cannot run here

Some evidence tests verify a claim against a specific research commit, or read a result bundle that
`.gitignore` excludes, or need a training checkpoint that was never tracked. Those **skip** with a
stated reason instead of failing: the check is about something this repository deliberately does not
carry, which is different from the evidence being wrong. The full suite runs here and is green with
those skips.

## History rewrite

**None.** The research repository's history was not rewritten, filtered or force-pushed. This
snapshot exists so a public repository can carry the current content without carrying the history
behind it.
