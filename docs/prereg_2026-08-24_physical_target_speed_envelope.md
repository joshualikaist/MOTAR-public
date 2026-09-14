# 사전등록 — physical-target 속도 포락선 진단

작성일: 2026-08-24. 결과를 본 뒤 속도 격자·gate·seed를 바꾸지 않는다.

## 목적

고정 속도 1.5 m/s에서 physical target의 205/300-bar feasibility gate가 실패했다. 이 진단은
동역학을 임의로 고치거나 PPO를 재학습하지 않고, **동일한 physical-target controller에서 밀도별로
어느 명령 속도까지 기존 safety contract를 만족하는지** 측정한다. 이는 fixed-speed interception
대신 density-conditioned speed envelope를 채택할지 판단하는 engineering 결과이며, 하드웨어 검증이
아니다.

## 고정 계약

| 항목 | 값 |
|---|---|
| seed | 509 |
| pattern | mixed (CV/waypoint) |
| densities | 70, 150, 205, 300 bars |
| speed grid | 0.6, 0.9, 1.2, 1.5 m/s |
| envs / measured steps | 32 / 280 (300 minus 20 warmup) |
| arena / placement | 40×40×3 m / navrl_band |
| target | `NAVRL_ROBOT=navrl_ref5in_quad`, `NAVRL_TARGET_DYNAMICS=physical` |

각 speed arm은 같은 seed로 별도 task를 재생성한다. 결과 JSON에는 16개 셀을 모두 보존한다.

## Gate

기존 `verify_navrl_physical_target.py`의 gate를 그대로 적용한다.

- tracking RMSE ≤ 0.35 m/s
- realized/command speed ratio ≥ 0.80
- contact step fraction ≤ 1%
- planner infeasible fraction ≤ 1%
- motor saturation fraction ≤ 15%
- max tilt ≤ 60°
- invalid state fraction = 0

각 밀도의 envelope 값은 **사전등록된 네 속도 중 가장 높은 passing speed**로만 기록한다. 모든
속도가 실패해도 속도·밀도별 실패 원인은 그대로 보고한다. gate를 만족하지 못하는 더 높은 속도를
사후에 추가하지 않는다.

## 해석 제한

이 결과는 target controller와 simulator geometry의 feasibility만 말한다. pursuer policy capture,
실제 기체 동역학, 실제 센서 성능, PPO 학습 가능성을 주장하지 않는다. envelope가 나오더라도 다음
fresh PPO는 별도 preregistration과 short smoke가 필요하다.

실행:

```bash
PYTHONNOUSERSITE=1 /home/fair/miniconda3/envs/aerialgym/bin/python \
  tools/verify_navrl_physical_target_speed_envelope.py
```
