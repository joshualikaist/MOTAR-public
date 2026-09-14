# `navrl_ref5in_quad` canonical simulator gate — 2026-08-13

판정: **PASS 21/21**, 단 **hardware validation이나 navigation 성능 주장이 아님**.

대상은 `navrl_ref5in_quad` hardware-informed simulation candidate와 legacy `navrl_quad`다.
원자료는 `flight_envelope.json`(schema 2, SHA-256
`35c315603af0afc41bd03adc7cbcaee35cd2d032fd01e86150d983e24bccf5a8`)이다.

## 이번 재측정이 필요한 이유

이전 JSON은 yaw 2.5 rad/s였고 forward/reversal에서 종료된 환경을 평균에서 제외해 생존자 편향이
있었다. 이번 측정은 다음 계약을 fail-closed로 고정했다.

- `base_sim`, 100 Hz physics / 10 Hz task control, seed 911, 16 environments
- 40×40×3 m, bars 0, governor off, vehicle-frame command, target motion 0
- yaw 3.0 rad/s, max tilt 45°, max horizontal command 2.5 m/s
- 모든 기동을 exact arena center, identity attitude, zero velocity에서 시작
- reset 직후 controller gains, motor time constant와 `k`를 각 config midpoint로 다시 고정
- current motor thrust를 runtime mass × gravity / 4로 초기화, 1 s settle 후 exact recenter
- state/actuator finite, 16/16 생존, 실제 runtime mass/inertia/body order/motor-link mask 검사
- controller는 두 기체 모두 동일한 inherited midpoint gains이며 ref5in용 retuning 없음

## 측정값

| 기동 | legacy | ref5in |
|---|---:|---:|
| hover steady altitude error | +0.0001 m | +0.0001 m |
| hover horizontal drift | ≈0 m/s | ≈0 m/s |
| forward steady / t90 | 2.490 m/s / 0.8 s | 2.490 m/s / 0.8 s |
| forward peak altitude error | 0.062 m | 0.011 m |
| reversal steady / zero-cross / t90 | −2.490 m/s / 0.5 s / 1.0 s | −2.490 m/s / 0.5 s / 1.0 s |
| reversal peak altitude error | 0.063 m | 0.027 m |
| yaw steady / t90 | 3.000 rad/s / 0.2 s | 2.999 rad/s / 0.2 s |
| 100 Hz fixed-gain pitch: 20° / peak | 0.15 s / 23.10° | 0.13 s / 25.01° |
| 100 Hz fixed-gain roll: 20° / peak | 0.15 s / 23.11° | 0.13 s / 25.01° |
| 100 Hz pitch/roll peak body rate | 5.363 rad/s | 3.844 rad/s |
| worst ref5in raw pre-clamp allocator limit fraction | — | **1.3%** |

Runtime actor 확인값:

| 항목 | legacy | ref5in |
|---|---:|---:|
| mass | 0.2500004 kg | 1.2000004 kg |
| ixx=iyy | 8.4501e-4 kg·m² | 4.1422e-3 kg·m² |
| izz | 1.6900e-3 kg·m² | 5.7692e-3 kg·m² |
| max thrust / motor | 2.0 N | 9.6 N |
| force bodies | motor_0..3 | motor_0..3 |

## 정확한 해석

이 결과는 **동일하고 ref5in에 맞춰 재튜닝하지 않은 Lee controller 아래에서 두 simulator model이
선택한 command gate를 통과했다**는 뜻이다. ref5in의 100 Hz peak body rate가 더 낮은데 20°에 더
빨리 도달한 것은 closed-loop transient와 overshoot가 함께 반영된 결과다. 이를 intrinsic 최대
각가속이나 ref5in이 더 민첩하다는 증거로 해석하지 않는다.

또한 ref5in collision box는 0.28×0.28×0.12 m이고 legacy는 0.28×0.28×0.08 m다. 45° tilt에서
한 축의 단순 projected support는 약 0.2546→0.2828 m로 **2.83 cm 증가**한다. 따라서 이 기체를
쓰는 navigation smoke는 dynamics-only ablation이 아니라 collision geometry까지 포함한
**whole-platform candidate** 평가다.

이 PASS가 보장하지 않는 것은 다음과 같다.

- obstacle avoidance, capture/crash/timeout, density curriculum 또는 PPO 학습 가능성
- ref5in에 맞춘 controller tuning과 plant-only identification
- exact motor/prop/ESC/battery의 thrust, RPM, current, time constant와 yaw torque
- exact BOM, CAD/CG/prop clearance/sensor FOV, 구조, power/thermal/endurance
- 실기 비행 또는 sim-to-real 성능

다음 허용 단계는 source receipt가 묶인 fresh 500-epoch learning-viability smoke뿐이다. 이 결과만으로
full training이나 hardware-reference 표현을 허용하지 않는다.

## 재현

```bash
cd /home/fair/workspaces/aerial_gym_ws/src/aerial_gym_simulator
export PYTHONNOUSERSITE=1

/home/fair/miniconda3/envs/aerialgym/bin/python tests/test_navrl_ref5in_platform.py
/home/fair/miniconda3/envs/aerialgym/bin/python tools/verify_navrl_ref_platform.py \
  --num-envs 16 --seed 911 \
  --output results/navrl_ref_platform_verification/flight_envelope.json
```

`verify_navrl_ref_platform.py`는 absolute conda Python으로 실행해도 같은 환경의 `ninja`를 찾도록
interpreter `bin/`을 PATH 앞에 고정한다.
