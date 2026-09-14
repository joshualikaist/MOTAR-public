# Gate 3 NI confirmatory replication (seeds 97/101, 205 bars, ep25000+riskcap)

| seed | arm | capture | crash | timeout | n |
|---:|---|---:|---:|---:|---:|
| 97 | analytic_bootstrap | 80.23% | 17.67% | 2.10% | 2049 |
| 97 | learned_v2 | 80.06% | 17.70% | 2.24% | 2051 |
| 101 | analytic_bootstrap | 79.40% | 17.37% | 3.22% | 2049 |
| 101 | learned_v2 | 79.55% | 17.91% | 2.54% | 2049 |

- pooled learned−analytic: **-0.015 pp**, 95% CI [-1.752, +1.723]
- preregistered margin −2.0 pp → **replication PASS**
- original campaign (seeds 83/89): −0.073 pp, CI [−1.790, +1.644] — reported separately, never pooled

Summary recomputed OUTSIDE the pinned launcher: the campaign's 4 cells were produced by the launcher whose SHA is pinned in campaign_contract.json, but its summariser still carried the original campaign's seed list (83/89) and crashed after the cells completed. The seed-list fix changed the launcher bytes, so the contract guard now correctly refuses to re-run it. Cell data is untouched; this file recomputes the preregistered endpoint from the archived cells.
