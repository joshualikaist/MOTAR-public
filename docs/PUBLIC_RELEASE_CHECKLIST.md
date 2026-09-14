# Public release checklist

Statuses concern the stated check, not blanket permission to distribute every historical asset.
PASS means the named check was run; BLOCKED includes pending external evidence. N/A is not PASS.

| Check | Status | Evidence / remaining work |
|---|---|---|
| Landing README, stable scope | PASS | [README](../README.md), public documentation tests |
| Historical results preserved | PASS | [archived README](archive/readme_9732d12_2026-09-12.md), [overview](results_overview_2026-09-12.md) |
| Isolated CPU quickstart | PASS | [installation evidence](../results/renderer_cpu_install_2026-09-12/README.md) |
| Citation schema | PASS | cffconvert 2.0.0 validated [CITATION.cff](../CITATION.cff), schema 1.2.0 |
| Root licence/attribution retained | PASS | [LICENSE](../LICENSE), [NOTICE](../NOTICE.md) |
| Dataset/source licence inventory | BLOCKED | Det-Fly download terms and asset/weight-level review remain; [inventory](../THIRD_PARTY_LICENSES.md) |
| Current ETH image/review-pack non-redistribution | RESOLVED | JPEGs and ZIP excluded; [external contract](external_data/ETH_DS5.md). Historical [member-preservation receipt](../results/eth_ds5_intake_2026-09-10/archive_notice_receipt.json) retained |
| All historical ZIP licence metadata | BLOCKED | Old revisions lack notices; no history rewrite; [audit](public_release_audit_2026-09-12.md) |
| No secrets anywhere in history | BLOCKED | Finite text-pattern scan found no credential matches; binary/oversized contents not scanned |
| No personal absolute paths | BLOCKED | Historical text contains personal paths; frozen provenance not rewritten |
| Dataset inventory | PARTIAL_EVIDENCE | ETH source/derived images excluded from current tree; historical blobs and other dataset-derived provenance remain open. [Release status](public_release_status_2026-09-14.md); path checks alone are insufficient |
| Tracked weights classified | PASS | 18 inherited/project files inventoried, not all individually licence-cleared |
| Public local Markdown links | BLOCKED | Run final public checker after all document edits; anchor/external URL validation excluded |
| CPU CI hosted run | BLOCKED | Workflow implemented; no hosted successful run yet; no badge |
| Local renderer/exporter checks | BLOCKED | Pending final clean-source execution in this cycle |
| Shared status schema and site | BLOCKED | Pending final validation; [manifest](status_manifest.json) |
| New PPO or deployment | N/A | Not part of this release cycle |

Do not clear a BLOCKED row just because other tests pass. The [master roadmap](plans/moving_target_rendezvous_master_plan_2026-09-12.md)
separates policy boundaries from data, licence, dependency and evidence requirements.
