# Public release audit — 2026-09-12

## Source and scope

After `git fetch origin`, HEAD and origin/main both equalled
`9732d12e1809c9da69f11e5b62e952b8a3e0fa17`; no divergence or fast-forward was necessary.
The prior follow-up changes were already committed/pushed and were reused, not duplicated.
All **locally available refs** were included, including historical branches. Remote-hidden refs,
reflogs and unreachable objects are not covered. No reset, force push or history rewrite occurred.

Tool: [audit_public_history.py](../tools/audit_public_history.py).
Machine-readable [corrected history inventory](../results/public_release_2026-09-12/history_scan_corrected_9732d12.json).
The first scan's current-index membership field was defective: it read the Git stage column
instead of the object-ID column. Content-scan counts were unaffected. That output is retained as
invalid for membership classification; a temporary-repository regression now checks both a current
and a removed blob. The corrected scan hashes the scanner source separately from the audited HEAD.

## Findings at the starting revision

| Check | Observation | Interpretation |
|---|---:|---|
| Reachable unique blobs | 7,521 | Across local refs, not just main's worktree |
| Uncompressed blob bytes traversed | 1,317,177,679 | Object contents streamed; no checkout of large binaries |
| Text blobs content-scanned | 7,322 | At most 2 MiB per text blob, no NUL bytes |
| Binary blobs not content-scanned | 115 | Inventory only; no secret-free claim |
| Oversized blobs not content-scanned | 84 | Inventory only; includes text/binary, not classified by this pass |
| Personal absolute-path matches | 15,517 | Historical occurrences, not unique people or current paths |
| Credential-pattern matches in scanned text | 0 | Finite private-key/provider-token/credential-assignment patterns only |

No `gitleaks` or `detect-secrets` executable was available. Neither was silently installed in a
research environment. The implemented scanner reports pattern names/counts, blob IDs and example
repository paths; it never exports matched credential strings. Documentation/fixture matches would
be candidates for review, not automatically confirmed secrets. Large removed files, archives,
images and weights are explicitly listed in the JSON inventory with current-index membership.
One example path per blob is retained; this is not a complete rename history.

**REVIEW_REQUIRED**, not “no secrets”. Binary archives are not recursively secret-scanned. No
exhaustive licence inference is made from extensions or folder names. Personal paths in frozen
receipts/code were not mechanically replaced because that would alter provenance; future public
instructions use relative paths or new task-specific virtual environments.

## Current distribution metadata

- Tracked JPEG count is **13**, not the earlier 34: ten review frames and three contact sheets.
- The ETH review ZIP's original 14 members are byte-preserved; two licence/attribution files
  are added. [Receipt](../results/eth_ds5_intake_2026-09-10/archive_notice_receipt.json).
- The full licence and creator/source/modification notice travel inside the current ZIP and
  beside the images. Earlier revisions still lack that packaging.
- Eighteen tracked `.pth` files are classified: eleven upstream examples and seven project files.
  Classification is not individual model/data licence clearance.
- No tracked `datasets/` files were found. This is **not** equivalent to no derived data anywhere.
- Root BSD copyright/disclaimer remains unchanged. No legal author name or ownership was invented.

Official licence sources and remaining unknowns are in [THIRD_PARTY_LICENSES](../THIRD_PARTY_LICENSES.md).
NavRL MIT and the separate NPS-Drones data BSD text were directly checked. Det-Fly's hosted
image/annotation licence remains **BLOCKED_BY_LICENSE**; the repository MIT text does not settle it.

## Validation tools and limits

`cffconvert==2.0.0` validates CFF 1.2.0 in a **separate new documentation venv**. An initial attempt
to combine it with jsonschema 4.23.0 failed dependency resolution: cffconvert requires jsonschema
below 4. The declared public validation profile uses 3.2.0 and passes `pip check`; this failure
was not a simulator defect and no research environment was modified.

The CPU workflow uses an independent venv and does not install Isaac Gym. Local commands and
the hosted workflow are different evidence; a workflow file is not a successful GitHub run.
No success badge is published before an observed hosted success. The documentation checker
checks declared public files and local path existence, not every historical link, external URL
or Markdown anchor. [Checklist](PUBLIC_RELEASE_CHECKLIST.md).

## Remaining blockers

- **BLOCKED_BY_EVIDENCE:** exhaustive binary/history secret scan, individual inherited
  asset/weight rights and historical archive notices.
- **BLOCKED_BY_LICENSE:** explicit Det-Fly downloaded-image/annotation terms; commercial or
  out-of-licence uses of ETH derivatives require separate review.
- **BLOCKED_BY_EVIDENCE:** hosted CI success until actually observed.
- **BLOCKED_BY_POLICY:** new target-acquisition/contact-enhancing code and research designs;
  no publication wording change removes that boundary.

These are not reasons to discard the completed independent CPU/doc checks, but they prevent
claiming an unqualified “public repository/release audit complete”. No dataset or user cache was deleted.
