# Governor/adaptation A/B/C — schema-v2 re-baseline

seed 53; deterministic; exact 600 actions; shared source `cc71428b0445…`; ~2049 episodes/cell.

| bars | A ep24000/off | B ep24000/riskcap | C ep25000/riskcap | B−A | C−B |
|---:|---:|---:|---:|---:|---:|
| 130 | 83.70% | 87.07% | 89.75% | +3.37 pp | +2.68 pp |
| 160 | 79.17% | 85.26% | 86.34% | +6.09 pp | +1.08 pp |
| 190 | 75.07% | 81.16% | 81.75% | +6.09 pp | +0.59 pp |
| 205 | 70.67% | 78.40% | 80.28% | +7.73 pp | +1.88 pp |
| 220 (OOD) | 66.37% | 75.55% | 77.06% | +9.18 pp | +1.51 pp |

B−A is the governor sequential contribution; C−B is the adaptation sequential contribution.
Do not reinterpret either as an interaction. 220 bars is OOD.
