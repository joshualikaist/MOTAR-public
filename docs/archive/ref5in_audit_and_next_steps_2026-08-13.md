# ref5in 최종 감사와 다음 순서 (2026-08-13)

## 한 줄 결론

`navrl_ref5in_quad`는 내부 정합성이 검증된 **hardware-informed simulation candidate**이고 PPO 학습도
가능하지만, 실기 기준 기체는 아니다. P2는 timeout 상한을 넘겨 실패했고, 후속 진단은 남은 병목을
“장거리 공통 충돌 + 장거리 CV-specific non-capture”로 좁혔다.

## Claude 초안에서 바로잡은 주장

1. `arm_motor_N`은 모터가 아니라 팔 시각 geometry의 중점이다. legacy motor 좌표는 ±0.13 m다.
2. legacy가 어떤 실기에도 대응할 수 없다고 증명하지 않았다. 정확한 표현은 mixed-scale 개발 수치가
   한 BOM/CAD에 추적되지 않는다는 것이다.
3. 377/221 rad/s²는 최대 각가속도가 아니라 hover-thrust roll authority 계산이다.
4. ref5in 1.20 kg, 합성 관성, 9.60 N/motor, 40 ms, 0.12 m height는 실측값이 아니다.
5. 0.28 m XY는 220 mm/5-inch prop-tip AABB보다 약 0.91% 작고, 0.12 m height는 45°에서 projected
   support를 약 2.83 cm 늘린다. intervention은 질량만 바꾼 것이 아니다.

## 완료한 검증

| 단계 | 결과 | 해석 범위 |
|---|---|---|
| P0 repository contract | 26/26 PASS | URDF/config/allocation/inertia 내부 정합성 |
| P0 canonical simulator | 21/21 PASS | 같은 controller에서 open-arena envelope |
| P1c fresh 900 epoch | PASS | 학습 가능성·안전·source provenance |
| P2 seed313, 2,049 eps | STRICT FAIL | 68.28/26.16/5.56%; timeout 114 > 102 |
| D0 seed317, 8,194 eps | descriptive complete | 장거리·pattern별 failure composition |
| D1 warm-start + seed331, 8,194 eps | FAIL | global/q3 crash PASS, q3/CV timeout 15.98% > 12% |

P1c는 성능 주장이 아니고 P2는 한 training seed/한 evaluation seed/70 bars뿐이다. 실기 성능은 어느
단계에서도 측정하지 않았다.

## D0 숫자가 말하는 것

- 거리 6–11.5 m: capture/crash/timeout `78.05/21.89/0.06%`.
- 거리 22.5–28 m: `55.94/29.09/14.97%`.
- 최장 거리 CV: `48.84/29.00/22.16%`.
- 최장 거리 waypoint: `62.87/29.17/7.96%`.
- 속도 0.3–0.6 → 1.2–1.5 m/s: timeout `8.49→3.10%`.
- 전체 crash: contact 1,737(81.59%), OOB 392(18.41%), height 0.

따라서 표적 속도가 높아서 못 잡는다는 설명은 맞지 않는다. 최장 거리에서 crash는 CV/waypoint 모두
약 29%지만, timeout만 CV에서 크게 늘어난다. episode를 늘리면 timeout 일부가 capture가 아니라 crash로
바뀔 수 있으므로 horizon 연장은 해결책이 아니다.

## 코드 감사 중 잡은 운영 결함

- dashboard generator가 TensorBoard archive 이동 뒤 과거 run 32개를 삭제해 보일 수 있던 문제를
  고쳐 checked-in/legacy/local run을 합친다. 현재 77 runs를 보존한다.
- P2 verifier가 개발이 계속된 뒤 과거 source snapshot까지 current tree와 같아야 한다고 요구하던
  문제를 고쳤다. 새 run은 current-source fail-closed, 과거 proof는 immutable snapshot으로 검증한다.
- held-out strata가 checkpoint에 남은 density training counters를 재사용하던 구조를 분리했다.
- 모든 strata에 capture/crash/timeout과 crash cause를 저장하고 global 합계 불일치 시 export를 막는다.
- 첫 diagnostic speed bin `[0,1.5]` 오류를 검출해 해당 speed 해석을 VOID 처리하고, 실제
  `[0.3,1.5]` support로 deterministic parity replay를 완료했다.
- 빈 circle cell의 Wilson CI를 non-standard JSON `NaN`이 아니라 `null`로 저장한다.

## 남은 순서

1. **D1 완료·FAIL:** global `76.82/18.62/4.55%`, q3 `70.22/19.96/9.82%`, q3/CV
   `64.03/19.98/15.98%`다. D0 대비 q3/CV capture `+15.19pp [10.94,19.44]`, crash
   `-9.02pp [-12.74,-5.31]`, timeout `-6.17pp [-9.57,-2.77]`로 개선됐지만 사전등록 절대
   timeout gate를 넘었다. 추가 epoch와 threshold 완화는 하지 않는다.
2. **heading diagnostic 완료:** toward/tangent-left/tangent-right/away timeout은
   `3.22/16.06/18.24/26.54%`였다. crash는 `18.59–20.20%`로 평평했고 tangent 좌우 최대 차이는
   2.19pp였다. radial-heading 환경 채널은 확인됐지만 path length·visibility·wall reflection이
   결합돼 있다.
3. **near-open diagnostic 완료:** 1 bar away−toward timeout도 `+54.32pp`라 dense obstacle
   occlusion 필요성은 기각됐다. away의 hidden fraction은 95.09%, timeout은 저속 q0 77.16%에서
   고속 q3 30.95%로 감소해 FOV 이탈과 벽 반사가 함께 의심된다.
4. **outcome telemetry 완료:** seed 353에서 toward/away timeout `0.34/54.61%`가 재현됐다.
   away capture/timeout visibility `15.02/0.59%` 차이는 +14.43pp로 사전 20pp screen 미달이다.
   wall-reflection-any는 둘 다 99%를 넘어 반사 여부만으로 outcome을 설명하지 못하며, speed별 표는
   반사 전에 OOB 종료되는 episode 때문에 생존 선택편향이 있다.
5. **초기 관측 계약 — seed 359로 계측 완료:** hard-distance `[22.5,28]m`는 camera/LiDAR range
   `20/12m` 밖이라 모든 episode가 target token 0으로 시작한다. outcome별 never-acquired rate는
   away timeout `87.52%`, away capture `0.00%`, toward capture `0.00%`로, 사전 30 pp screen을
   `+87.52 pp`로 통과했다. 두 capture cohort가 모두 정확히 0%라는 것이 임계보다 강한 신호다 —
   이 조건에서 최초 취득이 capture의 필요조건처럼 동작한다. 다만 outcome은 궤적의 결과이므로
   연관이지 인과가 아니다. 독립 재계산에서 fused와 camera 최초취득이 6개 cohort 전부 일치해,
   LiDAR가 camera보다 먼저 취득을 준 적이 없음이 드러났다 — 이 발견은 사실상 camera range
   발견이다. 다음은 RESEARCH_PLAN 8.29의 seed 367 인과 대조(camera range 20m vs 28m, 그 한
   값만 변경)다.
6. **P3는 계속 차단:** heading 결과는 원인 분리용이라 어떤 셀 결과가 좋아도 P2나 D1을 PASS로
   소급 변경하지 않는다. 다음 PPO는 단일 메커니즘 intervention과 새 gate를 먼저 문서화해야 한다.
7. **hardware gate는 별도:** exact BOM/CAD/CG, thrust stand, inertia/actuator identification, power/thermal/
   endurance, camera/LiDAR FOV·latency, 실제 hover/step/collision-envelope 검증이 필요하다.

이 순서는 한 run에서 airframe, reward, horizon, representation, governor를 동시에 바꾸지 않는다.
