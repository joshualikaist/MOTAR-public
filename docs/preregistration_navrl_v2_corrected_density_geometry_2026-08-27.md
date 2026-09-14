# Preregistration — corrected-v2 density geometry contract

Date frozen: 2026-08-27.

Geometry **gates** were written into `VERIFICATION.md` before the audit JSON
existed. This document freezes the **density curriculum contract** implied by
those gates and the canonical receipt. It does not authorize GPU work.

## Binding measurement

Receipt:
`results/navrl_v2_density_geometry_audit_2026-08-27/density_geometry_canonical_6_28.json`
SHA-256 `6b1f1b36cf73409d0c09483c3e1767b7ff196aecf0b95769f0e07d8dffa268d5`.

| item | frozen value |
|---|---|
| placement | `footprint_clearance`, surface clearance 0.45 m, merge fallback 0 |
| arena | 40 × 40 × 3 m |
| goal band for this cap | canonical 6–28 m |
| binding inflation | body circumradius + 0.45 m tracking reserve |
| connectivity gate | ≥ 95% |
| no-route gate | ≤ 5% |
| generation-failure gate | 0 |
| 205 bars | **PASS** (99.167% / 0.833% / 0) |
| highest passing density | 250 bars |
| 300 bars | **FAIL** (94.661% / 5.339% / 0) |

## Density contract

1. Fresh training target remains **70 → 205 bars**, step 15, minimum dwell 1,000
   epochs/level. 250 passing does **not** raise the training cap.
2. Connected evaluation / OOD cells are 64, 70, 100, 130, 160, 190, 205, 220, 250.
3. 300 bars is an **asset ceiling and disconnected-stress** cell, not a connected
   OOD claim. Do not write “300-bar connected free space” from this lineage.
4. `body_only` connectivity is not a training gate.
5. YOPOv2 4 m / 5 m count-density cells (100 / 64 bars) remain reference labels
   only. They are not a second training cap.

## Explicit non-authority

This freeze does **not** authorize:

- PPO or any other policy training
- 500-epoch smoke
- Track A detection Stage 2
- Track B recovery/route GPU reruns, 1.5 m/s claims, or env-count changes
- raising 205 because 220/250 also passed
- treating 300 as a connected evaluation density

The still-blocked next GPU step, if authority is later lifted, is a separate
engineering/smoke preregistration. Hardware Track A remains
`docs/SIM2REAL_3DAY_EXECUTION_PLAN.md`.

## Claim boundary

Supported: under non-overlap `footprint_clearance` layouts and canonical 6–28 m
pairs, 205-bar free space is connected after body+tracking inflation.

Unsupported: corrected-v2 PPO performance; hard 22.5–28 m connectivity; hardware
validation; carrying historical overlap-permitting capture curves into this
lineage.

## 후속 메모 (2026-08-27, 같은 날) — spawn 캐비어트 해소

이 문서와 위 receipt가 인용하는 `start_in_inflated_obstacle_frac`(205막대 27.45%)은
스폰 로직이 막대 **중심**으로부터 flat 0.65 m만 확인하고 막대별 실제 반경(circumradius
0.3133–0.5465 m)을 몰랐던 결함 때문이었다. `navrl_task.py`의 스폰 수락 술어를 막대별
표면 여유 판정으로 고쳤다(요구 여유 = robot inflation 0.19980 + tracking reserve 0.45 =
0.649802 m, 이 문서의 binding inflation과 동일).

재측정(60×128, 205막대 단독): `start_in_inflated_obstacle_frac` **27.45% → 0.104%**,
connectivity 99.987%, no-route 0.013%. 잔여 0.104%는 실제 위반이 아니라 감사 도구의
0.05 m 격자 스냅 오차다(정확 좌표 기준 위반 0건, 최소 여유 0.652342 m > 요구 0.649802 m).
64회 rejection sampling 예산은 소진되지 않았다.

**위 receipt(SHA `6b1f1b36…`)는 이 캐비어트를 아직 안고 있던 스폰 로직으로 측정된 것이며
수정하지 않는다** — connectivity/no-route/generation-failure 판정 자체(205 PASS, 300 FAIL)는
스폰 버그와 무관하므로 유효하다. 이 메모는 그 판정을 재판정하지 않고, 캐비어트가 코드에서는
해소됐음을 기록한다. 상세: `WORKLOG.md` 2026-08-27 "스폰 clearance를 막대별로" 항목.

여전히 이 사전등록은 PPO, 500-epoch smoke, Track A/B GPU 어느 것도 승인하지 않는다.
