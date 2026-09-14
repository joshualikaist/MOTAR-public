# 사전등록 — 거리(range) 충실도 진단 설계

작성일: 2026-08-24. 실제 sensor log가 없는 현재는 실행하지 않는다.

## 질문

현재 policy observation은 analytic exact relative range를 받는다. 검출 임계(fidelity Stage 1)와
독립적으로, **거리값의 오차·dropout·latency를 현실적인 측정 분포로 바꿨을 때 frozen policy가
어떻게 변하는가**를 측정한다.

## 실행 전 필수조건

1. 실기/bench에서 거리 ground truth와 sensor range를 같은 timestamp로 취득한다.
2. 거리 bin(5, 8, 12, 16, 20, 24, 28 m), 조명, 정지/횡이동 trial별 bias, p90 absolute error,
   valid fraction, dropout-burst p95, sensor→policy latency p95를 계산한다.
3. train/eval split은 frame이 아니라 독립 trial ID로 고정한다.
4. 위 profile과 source SHA가 없으면 range perturbation 수치를 임의로 정하지 않고 `NO-GO`다.

## 제안할 평가 arm (profile 확정 후에만 숫자를 채움)

- **A exact:** analytic range, 기존 frozen observation 계약.
- **B measured:** profile에서 고정한 bin-wise bias + bounded error distribution + measured dropout/
  age/valid mask + measured latency. 다른 sensor, airframe, reward, horizon, action, target dynamics는
  변경하지 않는다.

Primary는 held-out trial 단위 never-acquired와 capture/crash/timeout 원값이다. Secondary는 range
absolute error, acquisition step, action/margin telemetry다. 임계와 seed는 profile을 본 뒤 정하지
않고, profile 산출 전에 별도 preregistration amendment로 고정한다.

## 현재 판정

`BLOCKED_NO_REAL_RANGE_PROFILE`. 합성 Gaussian noise나 28 m exact range를 실측으로 부르지 않는다.
센서 로그가 생기면 `navrl_sim2real_ingest.py → navrl_sensor_profile.py → two_zone_replay.py` 뒤에
이 설계를 실행한다.
