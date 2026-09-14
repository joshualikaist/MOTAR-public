# 2026-08-21 diagnostic synthesis and next gates

이 문서는 서로 다른 checkpoint/robot/밀도 계보를 섞지 않고, 2026-08-21에 완료한 진단 네 개의
판정 권한과 다음 순서를 고정한다. 결과 뒤에 기존 gate를 바꾸지 않는다.

## Evidence ledger

| 질문 | 계보와 조건 | 핵심 결과 | 허용되는 결론 |
|---|---|---|---|
| 알려진 경계 관측이 blind-search OOB를 줄이는가 | ref5in fresh seed197, train 70 bars; held-out seed367, 1 bar, away-CV, camera20m, 2,049 eps/arm | control→geofence: capture 39.04→85.75%, crash 22.21→7.91%, timeout 38.75→6.34%, never-acq OOB/all 21.28→7.32% | mapped geofence의 task-level 이득은 primary/guard gate 통과 |
| 이득을 geofence token 사용에 사전등록 방식으로 귀속할 수 있는가 | 같은 geofence checkpoint에서 token force-invalid | capture 39.78%, timeout 55.00%, never-acq OOB/all 4.88% | capture 붕괴는 강한 사후 증거지만 OOB가 timeout으로 바뀌어 preregistered masked-loss gate 실패; 공식 판정은 mechanism unresolved |
| symmetric corridor에서 single Gaussian action의 mode averaging이 보이는가 | ref5in D1 ep1900, 70-bar host cell, 1 synthetic physical fixture×6 arms | slot-order max 0.0078, reflection action error 1.8332, symmetric lateral action 약 -0.916, horizontal command 약 3.35m/s | policy chirality가 품질 gate를 먼저 실패시켜 mode averaging은 지지/기각 모두 불가 |
| 205-bar bar contact가 unsafe speed margin과 연관되는가 | legacy navrl_quad ep25000+riskcap, seed379, 205 bars, 4,097 eps | contact 전 1s actual-direction negative margin 67.58%, capture steps 9.25%, +58.33pp; 664 contacts | descriptive association 지지. 속도 인과·riskcap 재튜닝 권한 없음 |
| static topology가 기존 실패를 설명하는가 | legacy 205-bar old dump, 1,989 non-timeout records only, bar size assumed 0.60m | recorded episode path exists 100%, cul-de-sac proxy 0%; timeout 60개는 dump에서 누락 | recorded non-timeout subset의 static disconnection은 안 보임. timeout/인과 결론 불가 |

## Decision

1. **지금 speed limit 또는 riskcap 파라미터를 바꾸지 않는다.** Joint telemetry는 연관을 보였지만
   실제 진행방향 clearance와 속도가 함께 나빠지는 원인이 인지, 방향 선택, 동역학 중 무엇인지
   분리하지 못했다. 기존 사후 riskcap 탐색 금지도 유지한다.
2. **지금 multi-candidate action head를 구현하지 않는다.** 단일 synthetic fixture는 policy chirality에
   의해 교란됐고 symmetric stall 자체가 나타나지 않았다. 먼저 real-frame reflection audit가 필요하다.
3. **mapped geofence는 유일하게 task-level confirmatory replication으로 승격할 후보이다.** 다만 fresh
   training seed가 하나뿐이고 geofence noise/dropout=0이므로 최종 채택이나 camera/LiDAR-only 탐색
   주장은 아직 불가하다.
4. **topology 결과는 탐색 자료로만 유지한다.** 다음 episode dump에는 actual bar footprint와 timeout을
   반드시 포함하기 전까지 topology-based curriculum을 만들지 않는다.

## Next preregistered order

### N1 — Real-frame reflection audit (evaluation only, first)

- frozen ref5in policy의 실제 simulator observation 최소 4,096개를 저장한다.
- 각 observation과 정확한 reflected pair를 deterministic forward하고 conjugate action error,
  lateral sign agreement, context별 bias를 측정한다.
- fixture가 아니라 real-frame median/p95로 chirality가 재현될 때만 reflection intervention을 설계한다.
- 이 단계는 학습, reward, environment, riskcap을 바꾸지 않는다.

### N2 — Prospective geofence replication (only after N1 contract is frozen)

- fresh train seed 두 개를 추가하고 control/geofence를 각 900 epochs 동일 계약으로 학습한다.
- 결과 전에 held-out seed와 pooled rule을 고정한다.
- current result를 소급 재판정하지 않고, 새 mechanism primary는 acquisition-failure/all
  (`never acquired and [OOB or timeout]`)로 정의해 OOB→timeout 치환을 포착한다.
- non-OOB crash +2pp guard와 capture/crash/timeout 원값을 계속 보고한다.
- 이후 별도 robustness 단계에서 geofence pose noise/dropout을 한 축씩만 바꾼다.

### N3 — 205-bar cause separation (evaluation only)

- joint telemetry의 actual-direction negative margin을 고정 outcome label로 사용한다.
- 속도 상한이나 governor를 탐색하지 않고, 동일 frozen policy에서 위험 step이
  (a) late obstacle visibility, (b) large requested heading change, (c) low realized deceleration 중 어디에
  먼저 귀속되는지 시간 순서로 분해한다.
- N3가 없으면 speed-aware reward나 braking-aware head를 구현하지 않는다.

### N4 — Future dump contract

- episode dump에 timeout episodes, sampled bar XY half-extents, robot footprint, arena bounds,
  initial/terminal pose, outcome cause를 포함한다.
- exact geometry로 path existence/clearance를 다시 계산한 뒤에만 topology curriculum 여부를 판단한다.

## Branch evidence

- active-search evaluation/result: `codex/active-search-geofence-eval` at `77deabe`
- symmetric mode probe/result: `codex/mode-probe` at `db0c9a0`
- joint speed telemetry/result: `codex/joint-telemetry` at `15719b2`
- topology tooling/exploratory summary: `codex/topology-labels` at `3d0f9f2`

이 네 branch는 primary dirty physical-target WIP에 아직 merge하지 않는다. 통합 시에는 작은 commit
단위로 source overlap과 runtime import guard를 다시 검토한다.
