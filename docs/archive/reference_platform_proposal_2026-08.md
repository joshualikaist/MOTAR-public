# `navrl_ref5in_quad` — 하드웨어 정보 기반 시뮬레이션 후보

작성 2026-08-12 / 감사 개정 2026-08-13

관련: `docs/sim_vs_hardware_gap_2026-08.md`, `results/navrl_ref_platform_verification/`

모델: `resources/robots/quad/quad_navrl_ref5in.urdf`,
`aerial_gym/config/robot_config/navrl_ref5in_quad_config.py`

> **판정:** `navrl_ref5in_quad`는 5인치급 부품 자료를 참고해 만든 **hardware-informed
> simulation candidate**다. 아직 제작 가능성이 CAD로 확인되거나 비행으로 검증된
> reference platform이 아니다. 아래의 검사 결과는 저장소 내부 정합성과 동일 제어기 아래의
> 시뮬레이션 응답만 뜻한다.

---

## 1. 08-12 진단 정정과 남아 있는 문제

초판은 `arm_motor_N` 조인트의 팔 중점(±0.065 m)을 모터 위치로 잘못 읽었다. 실제 legacy
모터 링크와 allocation matrix는 x=y=±0.13 m이므로 모터간 대각은 약 0.368 m다.

| legacy 요소 | 저장소에서 확인한 값 | 해석 한계 |
|---|---:|---|
| 모터 위치 | x=y=±0.13 m | 5인치 220 mm 프레임과 맞지 않음 |
| 충돌 박스 | 0.28×0.28×0.08 m | 모터/프로펠러 조합을 추적할 BOM이 없음 |
| 총 질량 | 0.250 kg | Aerial Gym 계열 기본값에서 유래 |
| 최대 총추력 | 8 N | 정확한 모터·프롭·ESC·전압 조합이 없음 |
| 조립 ixx | 약 8.45e-4 kg·m² | 내부적으로 불가능한 값이라고 단정할 근거는 없음 |

따라서 legacy가 수치적으로 파탄난 모델이라는 주장은 철회한다. 정확한 결론은 **실제 BOM과
CAD에 추적되지 않는 서로 다른 스케일의 파라미터가 섞여 있다**는 것이다. 어떤 실기체도
절대 대응할 수 없다고 증명한 것은 아니다.

`ref5in`은 이 문제를 완전히 해결한 실기체가 아니라, 220 mm/5인치라는 하나의 스케일에 맞춰
시뮬레이션 가정을 다시 묶은 후보 모델이다.

---

## 2. 외부에서 확인된 사실과 설계 가정을 분리한다

### 2.1 제조사·1차 자료로 확인된 값

| 항목 | 확인값 | 적용 범위 |
|---|---:|---|
| NVIDIA Jetson Orin NX 모듈 | **28 g** | SOM 단품. 캐리어·방열·전원부 제외 |
| Livox Mid-360 | **265 g**, 65×65×60 mm, 평균 6.5 W | 제조사 사양 |
| Intel RealSense D435i | **72 g nominal** | 카메라 단품 |
| Holybro Pixhawk 6C Mini | 현행 Model A **42.4 g**; legacy A 39.2 g; Model B 46.8 g | 35 g 주장은 폐기 |
| T-Motor P2207 V3 1750KV 예시 | 6S 시험표에서 최대 약 **16.6 N** | 5인치급 추력 가능성의 예시일 뿐 후보의 확정 조합 아님 |
| Agilicious 플랫폼 | 모터당 연속 정추력 약 **9.5 N**, 식별된 모터 시상수 **39.1 ms** | 다른 기체의 1차 실측값 |

이 자료는 5인치급에서 9.6 N/motor와 40 ms가 **가능한 크기**임을 보여준다. 그러나
`ref5in`의 정확한 모터·프로펠러·ESC·배터리 조합을 검증하지는 않는다.

### 2.2 잘못된 472 g 합계 정정

기존 문서는 `Orin NX 100 + Mid-360 265 + D435i 72 + Pixhawk 35 = 472 g`이라고 썼다.
두 문제가 있다.

1. Orin NX 공식 모듈은 28 g이고, 100 g은 캐리어·방열을 포함한다고 가정한 **컴퓨팅
   서브시스템 예산**이었다. 어떤 캐리어와 냉각 부품인지 정하지 않았다.
2. Pixhawk 6C Mini 35 g은 현행 공식 사양과 맞지 않는다.

현재 명시된 단품만 더하면 `28 + 265 + 72 + 42.4 = 407.4 g`이다. 다만 이 값에는 Orin
캐리어, 히트싱크/팬, DC-DC, 케이블, 커넥터와 마운트가 빠져 있어 실제 payload 질량이 아니다.
컴퓨팅 서브시스템에 임시 100 g을 배정하면 `100 + 265 + 72 + 42.4 = 479.4 g`이지만,
이 역시 **설계 allowance**이지 제조사 BOM이나 실측값이 아니다.

`ref5in`의 1.200 kg은 이 미완성 payload를 포함할 수 있는지 탐색하기 위한 시뮬레이션
design point로만 사용한다. “1.2 kg 제작 기체가 확정됐다”는 뜻이 아니다.

---

## 3. 현재 모델 파라미터와 증거 수준

| 파라미터 | legacy | `ref5in` | 현재 증거 수준 |
|---|---:|---:|---|
| 총 질량 | 0.250 kg | **1.200 kg** | 합성 설계점; 실측 BOM 없음 |
| 최대 추력/모터 | 2.0 N | **9.60 N** | legacy T/W를 맞춘 값; 동급 자료로 가능성만 확인 |
| 모터 위치 x=y | 0.130 m | **0.0777817 m** | 220 mm 대각 기하의 산술값 |
| 충돌 박스 XY | 0.28 m | **0.28 m** | 계보의 숫자를 유지한 proxy |
| 충돌 박스 높이 | 0.08 m | **0.12 m** | CAD가 아닌 packaging allowance |
| 조립 ixx=iyy | 8.45e-4 | **4.142e-3 kg·m²** | uniform-box + point-mass 합성값 |
| 조립 izz | 1.69e-3 | **5.769e-3 kg·m²** | 동상; CAD/진자 측정 없음 |
| 추력계수 k | 1.376e-5 | **4.401e-5** | 9.6 N@467 rps 재정규화; 실측 보정 아님 |
| 모터 시상수 | 0.04 s | **0.04 s** | 동급 실측을 참고한 prior; 후보 조합은 미식별 |
| yaw torque ratio | legacy 값 | 설정값 유지/스케일 | 후보 하드웨어 실측 없음 |

### 3.1 프레임과 충돌 proxy

5인치 프로펠러 직경을 0.127 m로 가정한 220 mm True-X 프레임에서

```text
motor radius = 0.220 / 2 = 0.110 m
x = y = 0.110 / sqrt(2) = 0.0777817 m
axis-aligned tip span = 2 × (0.0777817 + 0.0635) = 0.2825634 m
```

이다. 0.28 m 박스는 이 산술 AABB보다 2.56 mm, 즉 0.91% 작다. 따라서 “실제 prop-tip
footprint와 정확히 같다”거나 “프로펠러 swept volume을 보수적으로 감싼다”고 쓰지 않는다.
이는 기존 계보와 같은 **숫자 리터럴을 사용한 단순 박스 proxy**다.

### 3.2 높이 0.12 m는 충돌을 바꾼다

기존 문서의 “높이는 바닥/천장만 바꾸고 막대 접촉은 XY가 같아 완전히 동일하다”는 설명은
틀렸다. 박스가 pitch/roll하면 높이 성분이 횡방향 support에 투영된다. 한 축으로 45° 기울 때
그 방향의 전체 투영 폭을 단순 계산하면 다음과 같다.

```text
width(theta) = 0.28 |cos(theta)| + h |sin(theta)|
h=0.08, theta=45° -> 0.2546 m
h=0.12, theta=45° -> 0.2828 m
difference             0.0283 m
```

즉 45°에서 약 2.83 cm 차이가 생긴다. 실제 충돌은 엔진의 회전된 박스 판정에 따르지만,
**0.12 m 높이는 막대·바닥·천장 모두에 영향을 줄 수 있는 3-D 기하 변경**이다. CAD로 실제
외형을 얻기 전에는 0.12 m도 단지 가정이다.

### 3.3 질량·관성·actuator

- URDF의 `base_link 1.048 kg + 4×0.038 kg` 분할은 모델링 편의를 위한 합성 분할이다.
- 관성은 0.15×0.15×0.12 m 균일 중앙 박스와 네 개의 점질량을 합친 값이다. 배터리,
  LiDAR, 컴퓨팅 모듈의 실제 위치가 없으므로 CAD inertia나 실측 inertia가 아니다.
- 9.60 N/motor는 `38.4/(1.2g)=3.262`로 legacy의 nominal T/W를 보존한다. 정확한 propulsion
  조합의 연속 추력, 전류, 전압 sag, 열 한계는 확인하지 않았다.
- 현재 `use_rps` 모터 경로는 `sqrt(T/k)`로 변환한 뒤 다시 `k*rps²`를 적용한다. RPM rate
  clamp가 걸리지 않는 범위에서는 k가 대수적으로 상쇄될 수 있으므로, k 랜덤화가 실제 힘
  응답 다양화를 만든다고 주장하면 안 된다. 이는 별도 actuator-model 검증 항목이다.
- 40 ms는 Agilicious의 39.1 ms와 크기가 비슷하지만, `ref5in` 조합의 thrust-stand 결과가 아니다.

---

## 4. “동일한 병진 성능” 검증의 정확한 의미

이론상 동일한 최대 틸트와 T/W를 넣으면 다음 nominal bound가 같다.

```text
horizontal acceleration at 45° = g tan(45°) = 9.81 m/s²
vertical acceleration at full thrust = (T/W - 1) g = 22.19 m/s²
```

이것은 연속시간 강체의 단순 상한 계산이지 비행 궤적의 동일성을 보장하지 않는다. 질량,
관성, box height, yaw torque, controller saturation 및 자세 transient가 달라졌다. 그러므로
숫자 리터럴이 같다는 이유로 전체 동역학이나 과제 난이도가 같고 회전 외에는 차이가 없다고
표현하지 않는다.

수정된 schema-2 GPU gate 결과는 다음과 같다(16 env, seed 911, 장애물 0, governor off,
yaw 3.0 rad/s). exact center spawn, deterministic `mg/4` 초기화와 controller/motor midpoint,
전 환경 생존 및 raw pre-clamp allocator 요청을 검사했다.

| 기동 | legacy | `ref5in` |
|---|---:|---:|
| hover 고도오차 | +0.0001 m | +0.0001 m |
| forward 정상상태 / t90 | 2.490 m/s / 0.8 s | 2.490 m/s / 0.8 s |
| reversal 0-cross / t90 | 0.5 s / 1.0 s | 0.5 s / 1.0 s |
| yaw 정상상태 / t90 | 3.000 rad/s / 0.2 s | 2.999 rad/s / 0.2 s |
| 100 Hz fixed-gain pitch 20° / peak | 0.15 s / 23.10° | 0.13 s / 25.01° |
| 100 Hz fixed-gain roll 20° / peak | 0.15 s / 23.11° | 0.13 s / 25.01° |
| 100 Hz peak body rate | 5.363 rad/s | 3.844 rad/s |

이 결과에는 **same-controller confound**가 있다. 두 plant에 같은 고수준/저수준 제어기,
명령 제한과 10 Hz 관측을 사용했으므로 폐루프 제어가 plant 차이를 가리거나 포화시킬 수 있다.
특히 계산상 hover-thrust roll authority 비는 약 0.586이고 100 Hz peak body rate도 ref5in이
낮지만, closed-loop 20° 도달은 overshoot를 포함해 ref5in이 더 빨랐다. 따라서 이 표는 “해당
고정 제어기 아래에서 선택한 명령 gate를 통과했다”는 시뮬레이션 관찰일 뿐, raw dynamics
동등성, intrinsic 최대 각가속 또는 실기 타당성 검증이 아니다.

---

## 5. 지금까지 검사가 보여 준 것과 보여 주지 못한 것

### 현재 재현된 것

- CPU 정합성 테스트는 2026-08-13 감사 재실행에서 **26/26 PASS**다. 기존 k-spread 검사는
  `use_rps` 경로에서 motor-strength randomization이라고 볼 수 없음을 반영해, ref5in의
  `min=max=4.401e-5` fixed coordinate calibration과 implied 28,023 RPM 계약으로 교체했다.
- canonical GPU gate는 **21/21 PASS**다. hover/step/reversal/yaw뿐 아니라 100 Hz fixed-gain
  roll/pitch, 전 환경 생존, finite state/actuator, altitude/slip와 raw allocator saturation을 검사했다.
- legacy 기본값은 유지되고 `NAVRL_ROBOT=navrl_ref5in_quad`로만 후보를 선택한다.

### 보장하지 않는 것

- exact BOM의 제작 가능성, 구조 강도, prop clearance 또는 center of gravity
- LiDAR 360° FOV와 카메라 FOV의 프레임·프로펠러·배선에 의한 가림
- 실중량, CAD/실측 관성, 모터·프롭·ESC·배터리의 actuator dynamics
- 전력 분배, 전압 sag, 전류/발열, 냉각, 비행시간과 안전 마진
- 공력, ground effect, 센서 진동, latency, 통신 및 estimator 오차
- held-out navigation 성능, 장기·multi-seed 재현성, 기존 정책의 전이 가능성 또는 실기 성능

P1a/P1b fresh smoke에서 on-policy 학습 회복은 관측됐다. P1a는 behavior-KL rollback과 27 m
종료로 FAIL했고, 낮춘 LR의 P1b는 KL/rollback/outcome을 통과했지만 마지막 27→28 m evidence
window 전에 budget이 끝나 FAIL했다. 이는 held-out 성능이나 장기 학습 성공 증명이 아니다.

정확한 표현은 **repository consistency 26/26 및 same-unretuned-controller simulator gate
21/21 PASS**다. 이 두 검사가 통과해도 그 결과는 하드웨어 검증이 아니라 저장소·시뮬레이터 계약
검증이다.

---

## 6. 다음 순서와 주장 게이트

시뮬레이션 연구용 스모크는 현재 후보로 진행할 수 있다. 다만 실기 기준이라는 주장은 아래
게이트를 순서대로 통과한 뒤에만 허용한다.

1. **Exact BOM 동결** — frame, motor, prop, ESC, battery, Pixhawk revision, Orin carrier,
   cooling, DC-DC, sensors, cables, mounts를 제품명·수량·질량과 함께 고정한다.
2. **CAD/배치 검증** — prop clearance, 전체 collision hull, 0.12 m 높이, CG, LiDAR 360° FOV,
   카메라 FOV 및 센서 진동 절연을 확인한다.
3. **질량·관성 식별** — 조립 실중량과 CG를 재고, CAD inertia 또는 bifilar-pendulum 측정을 한다.
4. **추력대 식별** — exact motor/prop/ESC/battery로 thrust, RPM, current, voltage sag, k,
   time constant 및 yaw torque를 thrust stand에서 측정한다.
5. **power/thermal/endurance** — compute와 센서를 포함해 peak/steady power, 냉각, 비행시간과
   fail-safe 여유를 측정한다.
6. **독립 동역학 검증** — 동일 제어기만 비교하지 말고 open-loop 또는 식별 입력으로 plant를
   비교하고, 필요하면 확인된 분포로 domain randomization을 다시 정의한다.
7. **그 후 학습** — 현재 P1c fresh 900-epoch engineering gate → held-out P2 → 조건부 full seed 211
   → 검출기 arm → envelope arm 순으로 진행한다. 후보 파라미터가 바뀌면 smoke부터 다시 한다.

하드웨어 게이트 전의 논문·사이트 명칭은 `hardware-informed simulation candidate`로 통일한다.
실제 조립과 비행 검증 전에는 `buildable`, `flight-proven`, `real reference platform`을 사용하지
않는다.

---

## 7. 근거 목록

### 제조사·1차 자료

- NVIDIA, [Jetson Download Center](https://developer.nvidia.com/embedded/downloads) — Jetson Orin
  NX Series Modules Data Sheet의 모듈 질량 0.028 kg. 완성 compute assembly 질량은 아님.
- Livox, [Mid-360 specifications](https://www.livoxtech.com/de/mid-360/specs) — 265 g,
  65×65×60 mm, 평균 6.5 W, 9–27 V, 360°×59° FOV.
- Intel, [RealSense D400 Series Datasheet](https://www.intelrealsense.com/wp-content/uploads/2019/10/Intel-RealSense-D400-Series-Datasheet-Oct-2019.pdf)
  — D435/D435i nominal 72 g.
- Holybro, [Pixhawk 6C Mini](https://holybro.com/collections/flight-controller-peripheral/products/pixhawk-6c-mini)
  — revision별 공식 질량.
- T-Motor, [P2207 V3 official product/test data](https://store.tmotor.com/product/p2207-v3-fpv-motor.html)
  — 특정 1750KV/6S 조합의 추력 예시. 후보 BOM으로 확정한 것은 아님.
- Foehn et al., [Agilicious: Open-source and open-hardware agile quadrotor](https://lbfd.github.io/papers/ScienceRobotics22_Foehn.pdf),
  *Science Robotics* 2022, DOI 10.1126/scirobotics.abl6259 — 750 g 플랫폼, 모터당 약 9.5 N
  연속 정추력과 39.1 ms 모터 시상수.

### 저장소 내부 근거

- `resources/robots/quad/quad_navrl_collide.urdf` — legacy 질량·관성·모터 좌표·충돌 기하
- `resources/robots/quad/quad_navrl_ref5in.urdf` — 후보의 합성 질량·관성·기하
- `aerial_gym/config/robot_config/navrl_ref5in_quad_config.py` — thrust, allocation, k, tau
- `aerial_gym/control/motor_model.py` — `use_rps` 변환과 force dynamics
- `tests/test_navrl_ref5in_platform.py` — 저장소 정합성 검사
- `tools/verify_navrl_ref_platform.py` 및 `results/navrl_ref_platform_verification/` — 동일 제어기
  시뮬레이션 envelope 결과
