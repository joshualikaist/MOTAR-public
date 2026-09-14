# D7 — 통합 경로에서의 dynamic-mesh 비용 (2026-09-12)

사전등록 `PREREGISTRATION.md`(커밋 06c7643, 실행 전). receipt `benchmark/benchmark.json`.
구현 `aerial_gym/task/navrl_task/navrl_dynamic_mesh_shadow.py`, 훅 `navrl_detector.py`,
계측 `tools/probe_dynamic_mesh_shadow.py`, 검사 `tests/test_dynamic_mesh_shadow.py`.

**비용과 간섭 없음만 측정한다.** 검출기 출력, 관측 계약, association, 보상, 정책 중 무엇도 바꾸지 않았다.

## 판정: **`GO`**

주요 셀 **128 env × 160×90**에서 사전등록한 네 축을 **전부** 만족한다.

| 축 | 실측 | GO 조건 | NO-GO 조건 |
|---|---|---|---|
| 상대 증가 | **+0.97 %** | ≤ 10 % | > 30 % |
| 절대 증가 | **+0.411 ms** | ≤ 5 ms | > 20 ms |
| throughput 손실 | **+3.82 %** | ≤ 10 % | — |
| torch 예약 증가 | **+0.0 MiB** | ≤ 256 MiB | > 1024 MiB |

출력 불변성이 **모든 셀에서 성립**한다(선행 조건). 임계는 실행 전에 고정했고 바꾸지 않았다.

## 측정값

워밍업 50, 측정 500, 셀마다 shadow OFF/ON을 같은 seed·환경 수·해상도·행동 시퀀스로 짝지어 실행.

| 셀 | 기준선 중앙값 | shadow 중앙값 | 차이 | 상대 | 기준선 P95 | shadow P95 | steps/s | 불변 |
|---|---|---|---|---|---|---|---|---|
| **128 × 160×90** | **42.34 ms** | **42.75 ms** | **+0.41** | **+1.0 %** | 49.59 | 57.09 | 22.33 → 21.48 | ✓ |
| 32 × 160×90 | 38.29 ms | 38.11 ms | −0.19 | −0.5 % | 77.11 | 77.50 | 23.14 → 23.22 | ✓ |
| 128 × 320×180 | 47.81 ms | 48.18 ms | +0.37 | +0.8 % | 60.11 | 83.11 | 19.21 → 19.01 | ✓ |
| 128 × 480×270 | 56.71 ms | 57.75 ms | +1.04 | +1.8 % | 91.52 | 76.93 | 16.38 → 16.03 | ✓ |

32-env 셀의 −0.19 ms와 480×270의 P95 역전(91.52 → 76.93)은 **잡음 폭 안의 값**이다. 차이가
중앙값 대비 0.5 % 수준이고 P95가 양방향으로 흔들린다. 개선으로 읽지 않는다.

기준선 42.3 ms는 `WORKLOG.md:13876`이 기록한 평가 셀 50–80 ms와 같은 자릿수다. 임계를 그 예산에서
가져왔으므로 실측 기준선이 그 범위 근처라는 점은 판정의 전제가 성립함을 뜻한다.

## 메모리

| | 기준선 | shadow | 차이 |
|---|---|---|---|
| torch 최대 예약 | 동일 | 동일 | **+0.0 MiB** |
| NVML device used | 5,870 MiB | 5,902 MiB | **+32 MiB** |
| 프로세스 RSS | — | — | −15.4 MiB (잡음) |

torch가 0을 보고하는 것은 Warp가 자기 버퍼를 소유해 torch allocator를 거치지 않기 때문이다.
**그래서 NVML device-level 수치를 따로 기록했다.** 실제 증가는 그쪽의 +32 MiB이며, 삼각형 1,548개
메시와 `[128, 90, 160]` 두 버퍼에 해당한다. `UNAVAILABLE`을 0으로 쓰지 않았다.

## 출력 불변성 — 말이 아니라 해시로

shadow OFF/ON에서 다음 SHA-256이 **네 셀 모두 일치**한다.

```
detector target_mask
detector target_depth
obs_dict robot_position
task target_position
```

동시에 shadow 질의는 **실제로 일을 했다**: 주요 셀에서 667 픽셀이 표적 메시에 맞았다.
맞은 픽셀이 0이면 불변성은 공허하므로, 테스트가 `shadow_hit_pixels > 0`을 함께 요구한다.

## 간섭 불가를 구조로 고정

플래그 `NAVRL_DYNAMIC_MESH_SHADOW`, 기본값 `0`. 꺼져 있으면 **모듈 자체가 import되지 않는다**.
켜는 데는 환경변수와 명시적 `attach` 호출이 **둘 다** 필요하고, 모르는 플래그 값은 "off"로 읽지
않고 예외를 던진다. 테스트가 소스 수준에서 고정한다: 검출기는 shadow 버퍼 이름을 **언급조차 하지
않으며**, shadow 객체에 대해 호출하는 메서드는 `run` 하나뿐이고, shadow 경로에는 난수 사용이 없다.

## 계측 자체의 비용

주요 셀에서 타이밍 호출을 넣은 팔과 뺀 팔을 60스텝으로 비교했다. 타이밍 호출이 지배적 항이
아님을 확인했고 값은 receipt의 `instrumentation_overhead`에 있다.

## D6과 왜 숫자가 다른가

같은 작업의 서로 다른 분모다.

| | D6 | D7 |
|---|---|---|
| 재는 것 | 커널 질의만 | **전체 env step** |
| 기준선 | 해석적 프록시 질의 0.460 ms | env step 42.34 ms |
| 추가분 | +0.245 ms | **+0.411 ms** |
| 비율 | **1.53배** | **1.010배** |

추가분은 같은 자릿수다(0.245 대 0.411 ms, D7은 실제 검출기 해상도·메시·자세를 쓰므로 더 크다).
**비율이 다른 이유는 분모다.** D6의 분모는 프록시 커널 하나였고 D7의 분모는 물리·LiDAR·인지를
포함한 스텝 전체다. D6이 `INCONCLUSIVE`인 것과 D7이 `GO`인 것은 모순이 아니라 **다른 질문에 대한
다른 답**이다. D6의 판정은 그대로 둔다.

## 기하 계약 — 통일하지 않았다

`GEOMETRY_CONTRACT_DECISION_PENDING` 유지. receipt에 셋을 **명시만** 한다.

```
shadow 픽스처      navrl_target_drone_v3.urdf 의 visual 메시 (target-local)
기존 충돌 기하     navrl_target_drone_v2.urdf collision box 0.283 × 0.283 × 0.12
기존 센서 프록시   navrl_detector target_half_extents wp.vec3(0.14, 0.14, 0.06)
```

## `GO`가 뜻하지 않는 것

detector 마스크를 교체해도 된다는 뜻이 **아니다**. 다음 단계는 **D8 통합 사전등록**이며, 거기서
처음으로 `해석적 프록시` 대 `메시 유래 시각 관측`을 새 인지 treatment로 정의한다.

동결 정책에 대한 입장은 그대로다.

```
frozen policy sensitivity evaluation  = 가능 (별도 treatment로 보고)
adaptation 주장                        = 별도 training 계보와 사전등록 필요
```

주장할 수 있는 것은 딱 이것이다.

> 생산 렌더 경로에서 dynamic local-mesh 광선 질의를 추가 실행해도 검출기 출력은 바이트 단위로
> 같고, 주요 셀에서 전체 env step이 +0.41 ms(+1.0 %) 늘며 device 메모리가 +32 MiB 는다.

인지 정확도, shortcut 감소, association, 정책·태스크 성능에 대해서는 아무것도 말하지 않는다.
