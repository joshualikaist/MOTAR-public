# Perception 우선 실행 계획 (2026-09-09)

perception 파이프라인 완성을 우선으로 재작성한 계획이다. 정본 단계표는
[`perception_final_implementation_plan_2026-09-07.md`](perception_final_implementation_plan_2026-09-07.md),
직전 조사는 [`perception_streaming_findings_2026-09-09.md`](perception_streaming_findings_2026-09-09.md).

---

## 0. 먼저 정정 — 앞서 제안한 "다중 스케일 학습"은 이미 끝나 있다

09-09 초안에서 2순위로 제안했던 "검출기 다중 스케일 학습·앵커 재계산"은 **중복 실험이다.**
`runs/nps_detfly_joint/yolov5s_ms_b8_e30_s0`의 `opt.yaml`이 `multi_scale: true`이고, NPS 단독
checkpoint에서 warm-start해 joint 데이터로 30 epoch 학습돼 있다(P3-F, seed 0). 효과도 이미 측정됐다.

| Det-Fly, IoU 0.5 | NPS 단독 (P3 zero-shot, 13,271장) | **joint + multi-scale (P3-F, 6,913장)** |
|---|---:|---:|
| native AP | 0.0010 | **0.7166** |
| native 재현율 | 0.087 | **0.839** |
| scale-matched AP | 0.1126 | 0.4137 |

**두 수치를 같은 실험의 전후로 읽으면 안 된다.** P3는 Det-Fly를 한 장도 보지 않은 모델의 zero-shot
전수 평가이고, P3-F는 Det-Fly `010`을 학습에 쓴 모델의 **validation** 성적이다. P3-F의 0.7166은
"크기 문제를 고치면 이만큼 된다"의 증거이지 일반화 성능이 아니다.

또한 스트리밍 파이프라인이 이미 이 joint detector(`aab12f39…`)를 쓴다. 즉 P3의 처방은 **이미 적용된
상태**이며, 남은 것은 그 위에서 오차를 재고 주입하는 일이다.

**용어 교정**(사용자 지적 반영): 크기를 보정한 뒤 남는 격차는 "진짜 도메인 격차"가 아니라
**"크기 보정 후 잔여 도메인 격차"**로 쓴다. 배경·압축·시점·광학 특성은 그대로 남아 있다.

---

## 1. 지금 무엇이 남았나

| 단계 | 상태 | 남은 일 |
|---|---|---|
| P1–P3, P3-F, P4 | 완료 | — |
| P5 KF / P6 GRU | 완료(미채택) | — |
| P7 / P7b / P7c v2 | 완료(P7c v2 선택) | — |
| P7e crop verifier | 미채택 | — |
| streaming | 구현 완료, RGB parity PASS, **S1 overlap COMPLETE** | P8 오차 모델 |
| P7d learned ReID | 경계 문서만 | track-ID 데이터 없음 → 보류 |
| **P8 오차 모델** | **측정 완료**, 단일-GT 지원 범위 제한 | P9 적합도·주입 |
| P9 주입 | 미착수 | P8 뒤 |
| P10 PPO | 미착수 | 별도 승인 |

**test는 P5를 빼고 한 번도 열리지 않았다.** joint 데이터셋의 test는 봉인돼 있고 학습 yaml에
포함되지 않는다(`training_yaml_contains_test: false`). 이 봉인은 **마지막에 한 번만** 연다.

---

## 2. 실행 순서

### S1 — 광류 겹침 — COMPLETE (2026-09-09)

[전체 결과·hash](../../results/perception_streaming_overlap_s1_2026-09-09/README.md). validation
2,296프레임의 candidate/rank/motion bytes/하위 지표가 직렬·겹침 각 3회 모두 정확히 일치했다.
decode 포함 mean은 67.91 → 44.49 ms, 3회 평균은 14.72 → 22.48 FPS다.
따라서 이 validation 입력과 기록된 runtime에 한해 출력 비트 동일을 확정한다.

`estimate_backward_flow_and_gmc`를 둘로 쪼갠다.

- `compute_flow(previous_gray, gray)` — candidates 비의존, 36.6 ms, CPU
- `estimate_gmc(flow, candidates, …)` — 0.4 ms, candidates 필요

`PerceptionPipeline.process`에서 flow를 worker에 올리고 GPU 검출기와 겹친다. gray 계산(0.05 ms)만
선행하면 된다.

**"비트 동일" 표현 규칙(사용자 지적 반영).** 프로토타입 8회 측정은 flow 배열만 확인했다.
최종 문장으로 쓰려면 **validation 2,296프레임 전체에서 다음이 모두 일치**해야 한다.

1. `rank_mismatches == 0`, `candidate_frame_mismatches == 0`
2. **motion feature 배열 (5×12) 전 프레임 비트 일치** — 겹침이 바꿀 수 있는 유일한 중간 산출물이다
3. selected/hit/no-lock/false-lock/center error/loss/reacquisition 전부 일치

이 셋을 통과하기 전에는 "출력 보존 예상"이라고만 쓴다.

**benchmark 규율(사용자 지적 반영)**: 같은 환경(`detector_runs/venv`), **GPU 유휴 상태**,
동일 입력으로 **3회 반복**하고 mean/P50/P95를 전부 보고한다. Track A 학습이나 다른 GPU 작업과
겹치면 측정이 오염되므로 **동시 실행 금지**.

**완료 조건(PASS)**: 전후 latency mean/P50/P95 + 위 세 일치 + FPS. 사전 기준선은
detector 26.37 / motion 33.40 / selector 2.35 / decode 5.15 ms, decode 포함 67.28 ms, 14.86 FPS다.

**실패 시 처리**: motion feature가 한 프레임이라도 어긋나면 겹침을 되돌리고 원인을 적는다.
속도를 이유로 tolerance를 열지 않는다.

### S2 — P8 오차 모델 측정 — COMPLETE (2026-09-09)

[산출물·한계](../../results/perception_p8_2026-09-09/README.md): 2,296프레임 전체 baseline 재현,
단일-GT 1,310프레임의 크기별 분포·3-state/2-state 전이·중도절단 burst·S1 지연 측정 완료.
다중-GT 986프레임은 크기 조건부 집계에서 제외했다. <8 px는 0개, >=64 px는 2개이므로
미지원이며, 전이확률은 50–103 ms 관측 간격 기준이다. P9 적용 전에 이 경계를 처리해야 한다.
아래 초안의 “NPS는 영상당 UAV 1기”는 실제 복수 GT annotation과 맞지 않는다.
ID-switch 보류 이유는 안정적인 target identity가 없기 때문이다.

S1로 latency가 안정된 뒤에 한다. 순서가 중요하다 — latency 분포가 오차 모델의 한 성분이기 때문이다.

측정 대상은 **조건부 분포**이지 평균이 아니다.

| 성분 | 자료 | 형태 |
|---|---|---|
| center offset | matched 프레임 픽셀 오차 | 분위수 직접 샘플링(heavy tail: mean 1.942 / median 1.034) |
| miss burst | no-lock 174 / loss 106 / reacq mean 0.470 s max 19.93 s | 2-state Markov 전이확률 |
| false-lock burst | selected 중 12.91 % | 지속 길이 분포 |
| latency | p50 67.18 / p95 70.32 ms | 관측 지연 프레임 수 |

**표적 크기 조건부로 나눈다.** P3에서 크기가 지배 변수임이 확인됐으므로, 오차를 크기 구간별로 재지
않으면 시뮬 주입 시 잘못된 분포를 쓰게 된다.

**만들지 않는 것**: degree bearing(intrinsics 없음), metric range(거리 GT 없음),
실제 ID-switch(NPS는 영상당 UAV 1기). 이 셋은 P7d 경계 문서의 조건이 충족될 때까지 보류한다.

**완료 조건**: 크기 구간별 조건부 분포 + 시간 상관 파라미터 + receipt(runtime fingerprint 포함).

### S3 — P9 주입기 (2~3일, GPU 소량)

S2의 측정 분포를 `navrl_perception.py` 검출 단계에 주입한다. **renderer로 detector를 재학습하지
않는다**(시뮬에 쿼드로터 메쉬가 없다는 P3 이전의 결론 그대로).

**완료 조건**: 주입 분포와 실사 분포의 적합도, seed 고정 재현성. 주입 전후로 정책을 돌리지 않는다 —
그건 P10이다.

### S4 — 봉인 test 개봉 (마지막, 한 번만)

지금까지 선택·비교는 전부 validation에서 했다. joint test는 봉인돼 있다. **파이프라인이 확정된 뒤
한 번만** 열어 최종 수치를 낸다. 열기 전에 무엇을 보고할지 사전등록한다.

이 순서를 어기면 test가 선택에 오염된다. S1~S3 중에는 열지 않는다.

---

## 3. Track A와의 GPU 충돌 (사용자 지적 반영)

Track A의 R-C(학습 4회 3.25 h + 평가 40분)와 **S1의 latency 측정은 같은 GPU를 쓴다.**
동시에 돌리면 측정이 오염된다. 두 가지 중 하나로 처리한다.

- S1의 benchmark 구간(수 분)만 Track A를 멈추고 측정한다. 구현·검증은 겹쳐도 된다.
- 또는 Track A를 S1 완료 후에 착수한다.

논문 마감이 실제 제약이 되면 순서를 뒤집어야 한다(사용자 판단). 현재는 perception 우선이므로
S1 → S2 → S3 순서를 유지하되, **R-C는 S1의 benchmark가 끝난 직후 착수**하는 것이 GPU를 가장 덜 놀린다.

---

## 4. 규율

- **test 미개봉** — S4까지. `test_used: false`를 매 receipt에서 확인한다.
- **결과를 본 뒤 threshold·bin·tolerance를 바꾸지 않는다.**
- **RGB 실행 환경은 `detector_runs/venv`** (규칙 0.1). receipt에 runtime fingerprint가 들어간다.
- 기존 cache·checkpoint·frozen artifact는 덮어쓰지 않는다. 새 출력은 미존재 경로에 쓴다.
- 각 단계 완료 시 WORKLOG에 측정 숫자와 함께 기록한다.

## 5. 주장하지 않는 것

30 FPS, 실기 탑재, identity tracking, ROS/PPO 연결, 그리고 **P3-F의 0.7166을 일반화 성능으로**
읽는 것. 그것은 Det-Fly `010`을 학습에 쓴 모델의 validation 수치다.
