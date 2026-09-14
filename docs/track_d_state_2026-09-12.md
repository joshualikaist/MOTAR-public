# Track D — 시뮬레이터 외형: 단계별 상태 (2026-09-12)

한 곳에서 관리한다. 각 줄은 **무엇이 측정됐고 무엇이 아직 아닌지**만 말한다.

| | 단계 | 상태 | 근거 |
|---|---|---|---|
| D1 | 자산 로더 (URDF) | `COMPLETE` | `results/renderer_urdf_smoke_2026-09-11/` |
| D2 | 독립 렌더러 | `COMPLETE` | `results/renderer_r3_seed173_*`, CPU 설치 검증 `results/renderer_cpu_install_2026-09-12/` |
| D3 | 배경 장면 엔지니어링 | `TECHNICAL_PASS` | `results/renderer_background_v1_clean_2026-09-11/`, `..._2026-09-12/` |
| D4 | R4/R4b 통계 기준 | `FAIL` (보존) | `docs/renderer_r3_r5_results_2026-09-11.md` |
| D5 | 시뮬레이터 visual probe | `PARTIAL_EVIDENCE` | `results/target_appearance_in_sim_2026-09-12/` |
| D6 | 동적 메시 타당성 | `INCONCLUSIVE` (정확성 통과, 커널 질의 비용 1.53배) | `results/dynamic_mesh_raycast_feasibility_2026-09-12/` |
| D7 | 통합 렌더 비용 | **`GO`** (전체 step +0.41 ms / +1.0 %, 출력 불변) | `results/dynamic_mesh_integrated_cost_2026-09-12/` |
| D8 | mesh-derived observation 기술 검증 | D8-A **`TECHNICAL_GO`** | `results/dynamic_mesh_detector_d8a_attempt2_2026-09-13/` |
| D8b | 동결 정책 민감도 | **`MATERIAL_LOSS`**, COMPLETE | `results/dynamic_mesh_policy_sensitivity_d8b_2026-09-13/` |
| D8c | perception-path audit 제안 | `NOT_STARTED`; 사전등록·구현·측정 없음 | D8b와 별도 계보이며 이번 근거 보존 작업에 포함하지 않음 |
| D9 | 색 지름길 재측정 | `NOT_RUN` | — |

## D5가 확정한 구조적 경계

`navrl_physical_target_params`는 `include_in_warp = False`다. 표적 메시는 정적 Warp 장면에
들어가지 않는다. 따라서 검출기가 URDF visual을 보지 않는 것은 **자산 버그가 아니라 렌더러 구조의
결과**다. 현재 증거와 일관된 서술은 이것뿐이다.

```
URDF visual 변경  →  Isaac 자산 외형 변경
그러나
표적 마스크  →  해석적 구/상자 프록시  →  URDF visual geometry는 저장된 마스크에 영향을 주지 않는다
```

구조는 의도적으로 보인다. `정적 장애물 메시` + `동적 표적 해석적 프리미티브`로 분리해 매 스텝
BVH 재적합을 피한다.

## D7 — 측정 완료 (2026-09-12)

주요 셀 128 env × 160×90에서 전체 env step이 42.34 → 42.75 ms(**+0.41 ms, +1.0 %**), throughput
손실 3.8 %, torch 예약 +0.0 MiB, NVML device used **+32 MiB**. 네 셀 모두 detector `target_mask`·
`target_depth`와 궤적 해시가 **shadow OFF/ON에서 동일**하며, 같은 실행에서 shadow 질의는 667픽셀을
맞혔다(불변성이 공허하지 않다는 증거). 사전등록 네 축을 전부 만족해 판정 **`GO`**.

D6의 1.53배와 모순되지 않는다. 분모가 다르다: D6은 프록시 커널 질의(0.460 ms), D7은 스텝 전체
(42.34 ms)다. 추가분 자체는 0.245 ms 대 0.411 ms로 같은 자릿수다. **D6 판정은 그대로 `INCONCLUSIVE`다.**

### 이전 판의 문구 (보존)

**기존 숫자에서 보간하지 않는다.** 3.09 ms, 9.14 ms, R5의 음영 비용은 서로 다른 fixture와 단계의
측정이다. D6이 잰 것은 **독립 프로토타입에서의** 질의 비용(결정 셀 1.53배, +0.245 ms)이며,
그것은 통합 비용이 아니다. 정확한 표현은 이것이다:

> 추가 pixel-wise mesh query가 필요하므로 비용 증가는 예상되지만, **통합 비용은 현재 미측정이다.**

## 동결 체크포인트에 대해 할 수 있는 것과 없는 것

이전 판에 "관측이 바뀌므로 동결된 체크포인트 계보와 비교 불가"라고 적었는데 **너무 강했다.**

**가능하다.** 동결 체크포인트를 그대로 써서 `기존 인지` 대 `새 인지 처리`를 비교하는
**민감도/섭동 평가**는 할 수 있다. 그때 답하는 질문은 "기존 정책이 관측 분포 변화에 얼마나
민감한가"이며, 그것은 별도 treatment로 보고한다.

**할 수 없다.** 새 인지 결과를 기존 nominal 계보와 같은 관측 계약인 것처럼 합치는 것.
그리고 "새 인지에 적응한 정책의 효과"를 주장하려면 **별도 training 계보와 사전등록**이 필요하다.

```
frozen evaluation  = 가능, 단 별도 treatment
retraining         = adaptation 주장을 하려면 별도 계보 필요
```

## GEOMETRY_CONTRACT_DECISION_PENDING

접촉은 URDF 상자(0.283)를, 검출기는 하드코딩된 반치수(0.28)를 쓴다. 실재하는 불일치이지만
**타당성 실험의 선행조건은 아니었고** D6 프로토타입은 기존 기하 의미를 보존했다.
**본 검출기 통합 전에만** 결정하면 된다. 선택지 예시는 아래이며, **아직 고르지 않는다.**

```
A. 센서 visual geometry = 0.283, 접촉 = 기존 0.28 유지
B. 통합된 단일 기하 계약
C. visual / sensor / collision 기하를 명시적으로 분리
```

## 본 통합 전에 필요한 선행조건

1. D7 — 실제 검출기 경로에서의 통합 비용 측정 (D6의 독립 수치로 대체하지 않는다)
2. `GEOMETRY_CONTRACT_DECISION_PENDING` 해소
3. D6이 `INCONCLUSIVE`이므로, 통합을 정당화하려면 비용 근거를 D7에서 다시 세울 것
4. 관측이 바뀌는 변경이므로 사전등록과, adaptation을 주장한다면 별도 training 계보

2026-09-13 결정: 기하를 하나로 합치지 않고 위 선택지 C를 채택했다. collision, historical
analytic sensor proxy, v3 visual treatment를 명시적으로 분리한다. D8-A의 질문·arms·기술 gate는
[`D8 사전등록`](preregistration_dynamic_mesh_detector_d8_2026-09-13.md)에 결과 전 고정했다.
이는 D8-B 동결 정책 평가나 adaptation 학습을 자동 승인하지 않는다.

## D8-A — 구현 완료, GPU 판정 전 (2026-09-13)

별도 target-local Warp mesh를 한 번 구축하고 카메라 광선을 표적 좌표계로 변환하는 opt-in
경로를 구현했다. 표적 mesh와 정적 환경 mesh는 서로 다른 질의이므로 표적 self-occlusion이나
움직이는 표적 때문에 정적 BVH를 다시 만드는 경로가 없다. `mesh_flat`과 `mesh_shaded`는 동일한
mask/depth를 쓰며, 후자는 v3 URDF material의 상대 휘도와 고정 Lambertian 항으로 기존 nominal
red의 세기만 조절한다. face/material/normal/occlusion buffer는 진단 전용이고 actor observation에
들어가지 않는다.

unset/off는 기존 analytic 경로이고 모듈 import·mesh·buffer 할당이 없다. 알 수 없는 flag,
detect-resolution decoupling, D7 shadow와의 동시 부착은 fail closed한다. 구현·판정기 CPU 계약과
기존 D7/perception 회귀는 통과했으나, 이 문단은 GPU 커널 컴파일·무결성·비용 판정이나 detector
정확성/shortcut 감소를 주장하지 않는다. GPU 실행은 구현 커밋을 사전등록의 후손으로 고정한 뒤
clean tree에서만 열린다.

## D8-A — GPU 기술 gate 결과 (2026-09-13)

attempt 1은 `ninja` PATH 누락으로 task import 전에 끝난 0-cell `VOID_EXECUTION`이며 보존했다.
같은 conda prefix의 ninja path/version/SHA를 강제로 기록하도록 launcher를 고친 뒤 별도 attempt 2를
실행했다. 21 visible + 1 occluded pose의 2회 hash, 세 arm의 fixed-action 500-step 3회 hash,
debug-buffer 유효성, 정적 장면 occlusion이 모두 통과했다.

128 env×160×90에서 analytic 43.656 ms 대비 mesh-flat은 +0.256 ms(+0.585%), mesh-shaded는
+0.517 ms(+1.184%)였다. torch reserved 증가는 각각 +44/+66 MiB, NVML은 +44/+64 MiB다.
사전등록 비용·무결성 조건을 모두 만족해 D8-A는 **`TECHNICAL_GO`**다. 21 pose의 mesh/analytic
면적비 중앙값 0.556은 observation treatment가 실제로 기하를 바꿨다는 확인일 뿐 성능 향상값이 아니다.
D8-B frozen-policy sensitivity와 D9 shortcut audit, 학습은 이 판정에 포함되지 않는다.

D8-B의 3 arm×3 seed×2,049 episode, primary capture contrast, seed-level paired t interval,
−3.0 pp margin과 무결성 gate는
[`D8-B addendum`](preregistration_dynamic_mesh_policy_sensitivity_d8b_2026-09-13.md)에 결과 전에
고정했다. 이 addendum은 evaluation-only이며 adaptation/PPO를 열지 않는다.

## D8b — 기존 9셀의 보존·종료 (2026-09-13)

3 seeds × 3 arms × 실제 2,049 episodes, 총 18,441회를 새 GPU 실행 없이 검증·확정했다.
원문 §4는 8개 gate다. gate별 대조 및 출처 한계는
[감사 기록](../results/dynamic_mesh_policy_sensitivity_d8b_2026-09-13/AUDIT.md)에 둔다.
사전등록 primary는 −48.967 pp, seed-level 95% t CI [−50.113, −47.821] pp로,
−3.0 pp margin에 대해 **`MATERIAL_LOSS`**다. 원래 runner의 verify와 별도 count 기반 계산도 통과했다.
D8-A의 `TECHNICAL_GO`를 덮어쓰지 않으며, 이 정책 반응으로 특정 모듈의 실패 원인을 주장하지 않는다.
후속 D8c·detector/association adaptation·P8/P9 교체·D9는 실행하지 않았다.
