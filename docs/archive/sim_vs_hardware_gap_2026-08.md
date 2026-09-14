# 시뮬 기체 vs 실기 하드웨어 — 격차 명시 (2026-08-12)

> **보존 문서:** 당시의 정량 격차 감사다. 2026-08-23 이후 72시간 실행 순서와 새 학습 계측 계약은
> [`../SIM2REAL_3DAY_EXECUTION_PLAN.md`](../SIM2REAL_3DAY_EXECUTION_PLAN.md)를 단일 기준으로 삼는다.

> 목적: "이 정책을 실제 드론에 올릴 수 있는가"를 논할 때, **시뮬에서 가정한 기체**와
> **그 인지 스택을 실제로 돌리는 데 필요한 하드웨어** 사이의 정량 격차를 남긴다.
> 시뮬 수치는 저장소 설정에서 직접 읽었다. 아래 하드웨어 질량은 **개별 제조사 부품 사양**이며,
> 완성 기체의 BOM이나 실측 payload 질량이 아니다. 캐리어·냉각·전원·배선·마운트가 확정되지
> 않았으므로 부품 질량을 더해 곧바로 제작 가능한 기체 질량으로 해석하면 안 된다.

---

## 1. 시뮬 기체와 실행 profile을 분리한 스펙

아래 질량·충돌·actuator 행은 legacy `navrl_quad` rigid body다. task 기본값과 corrected-v2 고정
launcher 값은 서로 다른 계층이므로 한 profile처럼 섞지 않는다.

| 항목 | 값 | 출처 |
|---|---|---|
| 총 질량 | **250 g** (base 225 g + 로터 4×6.25 g) | URDF `mass value` |
| 충돌 박스 | 0.28 × 0.28 × 0.08 m | URDF `box size` proxy |
| 모터 최대추력 | 2.0 N × 4 = **8.0 N** | `base_quad_config.max_thrust` |
| **최대 T/W** | **3.26** (호버 2.45 N) | 계산 |
| 최대 상승 가속 | 22.2 m/s² | 계산 |
| 최대 틸트 | 45° → 수평 가속 ≈ 9.8 m/s² | `NAVRL_MAX_TILT_DEG` |
| 모터 시상수 | 0.04 s | `base_quad_config` |
| 최대 속도 / yaw rate | 2.5 m/s(축별) / corrected-v2 **3.0 rad/s** | task default 속도 / 고정 launcher |
| 제어 주기 | 10 Hz (물리 100 Hz) | `rl_step_dt_s=0.1` |

**corrected-v2 실행 profile**: LiDAR 72×4 @ 12 m, obstacle token FOV 240°, 전방 RGB-D
160×90, 87° HFOV @ 20 m, yaw command 3.0 rad/s. 일반 task/config 기본값이나 구형 3-D viewer를
이 profile로 간주하면 안 된다. IMU는 **비활성**(`enable_imu = False`)이고 ego-state는
시뮬레이터가 직접 제공한다.

---

## 2. 후보 인지 스택의 개별 부품 질량

| 부품 | 제조사 사양 질량 | 포함되지 않은 것 |
|---|---:|---|
| Jetson Orin NX | **28 g** | SOM 모듈만 해당. 캐리어, 방열판/팬, DC-DC, 저장장치와 케이블 제외 |
| Livox Mid-360 | **265 g** | 마운트·배선 제외 |
| RealSense D435i | **72 g nominal** | 마운트·케이블 제외 |
| Pixhawk 6C Mini | 현행 Model A **42.4 g** / legacy A **39.2 g** / Model B **46.8 g** | revision을 고정해야 함 |

현재 Model A를 택해 이름이 정해진 단품만 더하면 **407.4 g**이지만, 이는 완성 payload가
아니다. Pixhawk revision에 따라 같은 부분합도 404.2~411.8 g으로 달라진다. 무엇보다 Orin NX
28 g은 완성 연산 장치가 아니며, 캐리어·방열·전원·저장장치·케이블·마운트 질량이 아직 없다.
기존의 `Orin 100 + Mid-360 265 + D435i 72 + Pixhawk 35 = 472 g`은 제조사 사양끼리의 확정
합계가 아니므로 폐기한다. 100 g은 완성 compute subsystem을 가정했던 설계 allowance였고,
Pixhawk 35 g은 현재 확인되는 공식 revision 질량과 맞지 않는다.

공식 근거:

- NVIDIA [Jetson Download Center](https://developer.nvidia.com/embedded/downloads) — Orin NX
  module datasheet의 0.028 kg은 SOM 단품 질량이다.
- Livox [Mid-360 specifications](https://www.livoxtech.com/de/mid-360/specs) — 265 g,
  65×65×60 mm, 평균 6.5 W.
- Intel [RealSense D400 Series Datasheet](https://www.intelrealsense.com/wp-content/uploads/2019/10/Intel-RealSense-D400-Series-Datasheet-Oct-2019.pdf)
  — D435/D435i nominal 72 g.
- Holybro [Pixhawk 6C Mini](https://holybro.com/collections/flight-controller-peripheral/products/pixhawk-6c-mini)
  — Model/revision별 질량.

---

## 3. 핵심 격차 — 세 가지

### ① 이름이 정해진 단품만으로도 시뮬 기체보다 무겁다 (가장 큰 격차)

시뮬 기체 총 질량은 **250 g**인데, 아직 불완전한 명시 단품 부분합도
**404.2~411.8 g**이다. 즉 캐리어·냉각·전원·배선·마운트를 넣기 전부터 시뮬 기체 전체보다
무겁다. 따라서 질량·관성 격차라는 결론은 472 g이라는 잘못된 합계에 의존하지 않는다.

완성 기체 질량은 exact BOM과 실측 전에는 알 수 없다. 별도로 만든 `navrl_ref5in_quad`의
**1.2 kg**은 가능한 질량대를 탐색하기 위한 **hardware-informed simulation candidate**이지,
제작 가능하거나 비행 검증된 기준 기체의 확정 AUW가 아니다. 이 1.2 kg 설계점에서만 계산하면:

- T/W 3.26을 유지하려면 총추력 **38.4 N**(약 40 N)이 필요하다.
- 관성이 커져 **각가속도가 떨어진다** → 시뮬에서 학습한 기민한 회피 기동의 실현성이 낮아진다.
- 다만 ref5in 관성은 CAD나 실측값이 아니라 합성 모델이고, exact motor/prop/ESC/battery도
  고정되지 않았으므로 이 계산은 실기 성능 보증이 아니다.

고밀도 진단에 사용한 **historical frozen legacy policy**는 250 g 시뮬레이션 동역학에서 학습됐지만,
이름이 정해진 하드웨어 단품만으로 그 총질량을 초과한다. 현재 진행 중인 ref5in P1 계보는 별도의
1.2 kg synthetic design point에서 fresh 학습하며, 아직 held-out 성능을 증명하지 않았다. 전자의
legacy sim-to-real 질량 격차와 후자의 미검증 후보 계보를 하나의 “현재 정책”으로 합치지 않는다.

**함의**: 실기 이행 시 (a) 기체를 키우고 정책을 그 동역학으로 재학습하거나, (b) 인지 스택을
경량화해야 한다. (b)의 구체 경로가 §5.

### ② LiDAR가 가장 큰 단일 질량 항목이다

Livox Mid-360은 **265 g**으로, 미완성 단품 부분합에서도 가장 큰 항목이며 legacy 시뮬 기체
전체 250 g보다 무겁다. 정확한 완성 payload 분모가 없으므로 “전체의 56%” 같은 비율은 더 이상
사용하지 않는다. 우리 관측 계약은 360° 72×4 스캔에 강하게 의존한다(장애물 토큰, corridor,
riskcap governor의 clearance 전부 이 스캔에서 나온다).

**함의**: LiDAR를 빼면 가장 큰 명시 질량 항목은 사라지지만, 완성 기체가 얼마나 가벼워지는지는
BOM을 닫아야 알 수 있다. 또한 관측 계약을 재설계해야 한다. 발전 방향 **D8(학습형 융합)** 과
직접 연결된다 — depth 카메라만으로 장애물 표현을 만들 수 있다면 경량화 여지가 생긴다.

### ③ IMU/상태추정을 시뮬이 공짜로 준다

시뮬은 `enable_imu = False`로 두고 **위치·속도·자세를 정확히** 제공한다. 실기에서는 VIO나
LiDAR-inertial odometry가 그 값을 만들어야 하고, 그 오차가 검증 3에서 잰 **pose 오차 축**이다.

검증 3(수정판, seed181 RNG 격리)의 결과: **yaw 오차 2°에서 −3.28 pp, 5°에서 −12.75 pp**
(위치 오차는 10 cm까지 CI가 0을 포함 — 미검출). 단, 이는 **단일 환경 시드 + 스텝별 iid
가우시안**에 한정된 결과이며, 실제 odometry의 **바이어스·드리프트·상관 오차는 미검증**이다.
따라서 이 수치를 "하드웨어 허용 스펙"으로 제시하면 안 된다(Codex 검수 지적).

---

## 4. 정책 네트워크 연산은 작지만, 실기 연산 폐쇄는 미검증

토큰 확장을 걱정할 필요가 없다는 것이 실측으로 확인됐다. 아키텍처를 그대로 재현해 x86 CPU에서
batch=1로 측정:

| 장애물 용량 | 추론 시간 | 파라미터 |
|---:|---:|---:|
| 8 (현재) | 0.399 ms | 344.5k |
| 16 | **0.396 ms** | 356.7k |
| 32 | 0.399 ms | 381.3k |

**측정 오차 안에서 동일하다.** 이유는 아키텍처에 있다 — `navrl_transformer_network.py:31-34`가
명시하듯 장애물 용량을 올리면 **토큰이 늘지 않고** `obstacle_project`의 입력 폭만 넓어진다
(8×12=96 → 16×12=192). Transformer는 계속 17토큰이므로 어텐션 비용(토큰² 스케일)이 불변이다.

제어 주기 10 Hz(100 ms) 대비 정책은 **0.4 ms = 예산의 0.4%**.

**현재 따로 측정한 policy와 detector 중에서는 detector 비용이 더 컸다**:

| 항목 | 160×90 (현재) | 640×480 (실기 카메라) |
|---|---:|---:|
| 정책 | 0.4 ms | 0.4 ms (해상도 무관) |
| 검출기 v7 (11.3k params) | 1.5 ms | **43 ms** |

검출기가 정책보다 이미 4배 비싸고, 이 CPU 측정에서는 640×480 입력이 약 100배 비싸다.
Orin NX + TensorRT가 지연을 줄일 가능성은 있지만, 정확한 carrier, 전력 mode, TensorRT build,
메모리 전송, 센서 I/O와 동시 부하에서 측정하지 않았다. compute assembly의 전력·냉각·질량도
닫히지 않았으므로 “Orin으로 해결된다”는 결론은 아직 낼 수 없다.

---

## 5. 정리 — 실기 이행의 제약 순서

1. **무게/기동성** (구조적, 가장 큼): 불완전한 명시 단품 부분합 404.2~411.8 g부터 이미
   250 g 시뮬 기체보다 무겁다. → exact BOM을 먼저 닫고 그 질량·관성으로 재학습하거나,
   센서 경량화(D8)를 검토한다. ref5in 1.2 kg은 그 전 단계의 시뮬레이션 후보일 뿐이다.
2. **상태추정 품질**: yaw 오차가 민감축(2°에서 유의). VIO/LIO 성능이 곧 정책 성능.
   바이어스·드리프트 축은 아직 안 쟀다.
3. **검출기 연산**: 현재 x86 CPU 측정에서는 640×480 비용이 43 ms다. 목표 하드웨어의
   동시 부하·전력·열 조건에서 GPU/TensorRT 또는 해상도·백본 설계를 검증해야 한다.
4. **정책 연산**: 격리된 x86 측정에서는 제약이 아니었다. 토큰 확장(D2)의 추가비용도 해당
   측정 오차 안에서는 검출되지 않았지만, 목표 compute assembly에서 재측정해야 한다.

**D4(velocity → CTBR)는 실기 인터페이스를 단순화할 가능성**이 있다. 다만 Pixhawk revision,
firmware, command path와 end-to-end 지연을 고정·측정하지 않았으므로 “지연이 더 짧다”는 것은
아직 가설이다. 무거운 후보 기체의 속도 추종 지연도 실측 또는 식별 모델로 확인해야 한다.

---

## 부록 — 미검증 항목 (정직하게 남김)

- 실제 기체 질량/관성에서의 정책 성능은 **한 번도 측정하지 않았다**. 위 논의는 부품 사양과
  시뮬레이션 설계점의 비교다.
- exact buildable BOM, CAD packaging, prop clearance, CG, 센서 FOV 가림은 닫히지 않았다.
- 배터리 지속시간, 전력 분배, voltage sag, compute·센서·ESC 열관리, 프레임 강성,
  프로펠러 효율은 미검증이다.
- `navrl_ref5in_quad` 1.2 kg과 그 관성·actuator 값은 hardware-informed simulation candidate의
  합성 파라미터다. 실측 BOM, CAD inertia 또는 thrust-stand 결과가 아니다.
- Livox Mid-360의 실제 점밀도가 우리 72×4 스캔 계약을 만족하는지 미확인.
- 실기 카메라 노출/롤링셔터가 검증 2의 motion blur 축과 같은지 미확인.
