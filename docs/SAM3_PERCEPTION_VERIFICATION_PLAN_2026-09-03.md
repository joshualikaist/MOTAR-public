# SAM 3 instance-preserving perception: architecture and verification plan

> ## ⛔ 상태: SUPERSEDED (2026-09-04)
>
> SAM 3를 in-sim detector 자리에 놓는 이 경로는 **실행하지 않는다**. 이유는 SAM의 성능이 아니라
> **입력에 형상 정보가 없다는 것**이다: detector가 보는 표적은 반지름 0.15 m 해석적 구에 상수색을
> 칠한 것이고, 디코이 3종 중 하나는 같은 반지름의 구이며, 배경은 40×24 depth를 업샘플한 회색
> 명암(텍스처·조명 없음)이다. 어떤 segmentation 모델도 존재하지 않는 형상을 분할할 수 없다.
>
> 세 갈래 측정이 "구속 조건은 측정 품질이 아니라 물체 동일성"으로 수렴했고(자료구조 +1.3 pp,
> 거리 분산 −0.34 pp, 오표적 lock −55.5 pp), 그 동일성 인식기는 **실제 공대공 데이터**로 학습한다.
> 전체 사유는 `docs/plans/perception_shape_temporal_redesign_2026-09-03.md` 상단 고지와
> `README.md`의 *External data* 절에 있다.
>
> **살아남은 것**: 후보 보존(instance-preserving) 구조 계약과 association packet 경계는 실데이터
> detector에도 그대로 필요하다. 폐기된 것은 **SAM을 그 자리에 놓는 것**이다.

작성: 2026-09-03. 이 문서는 **실행 전 계획안**이었다. 현재 채택된 detector, 898-D actor,
checkpoint, PPO 결과 또는 연구 판정을 변경하지 않는다. 각 GPU/Isaac/실기 단계는 해당 단계의
입력·모델·threshold·seed·판정 규칙을 별도 사전등록한 뒤에만 실행한다.

## 1. 결론과 목표

SAM 3를 `AppearanceTargetSegmenter`와 일대일 교체하지 않는다. 목표는 다음 두 실패를 분리해
해결하는 것이다.

1. **인식 실패:** 색이 같은 표적과 distractor를 의미·형상·exemplar로 구분하지 못한다.
2. **자료구조 실패:** 여러 양성 영역을 합쳐 어느 물체도 아닌 하나의 centroid/range를 만든다.

후보 구조는 `timestamped sensor packet → asynchronous discovery/propagation → K instances →
3-D association → multi-hypothesis tracks → TARGET/AMBIGUOUS/REJECT → target history`다. 충돌 안전은
semantic detector와 분리한 raw range/occupancy 경로가 담당한다.

현재 완료된 것은 `navrl_instance_adapter.py`의 **오프라인 CPU 경계 하나**뿐이다. 실제 SAM worker,
IPC/transport, intrinsics·extrinsics 기반 3-D lifting, timestamp/pose compensation, track bank,
actor 연결 및 detector-independent safety path는 구현되지 않았다.

## 2. 현 구현 감사: 맞는 부분과 아직 맞지 않는 부분

### 이미 지켜지는 계약

- connected component를 instance별 mask로 보존한다.
- 후보별 `bbox`, `score`, `uv`, `depth_median`을 따로 저장한다.
- K>1 mask의 union centroid 계산을 명시적으로 거부한다.
- 점수가 비슷하면 `AMBIGUOUS`, 후보가 없으면 `REJECT`하며 lock을 강제하지 않는다.
- live `navrl_perception.py`는 adapter를 import하지 않아 기존 제어 경로가 바뀌지 않는다.
- SAM은 같은 Isaac 프로세스에 로드하지 않는다는 경계를 둔다.

### production 연결 전에 반드시 보완할 항목

| 영역 | 현재 상태 | 필요한 완료 조건 |
|---|---|---|
| SAM backend | `sam` 선택 시 fail-closed | 실제 별도 worker와 versioned request/response schema |
| 전송 | `npz` key 문자열만 선언 | atomic write 또는 IPC, checksum, timeout, stale-response 폐기 |
| 시간 | frame ID/timestamp 없음 | capture timestamp와 pose timestamp, monotonicity, max-age gate |
| 기하 | pixel `uv`와 depth만 있음 | intrinsics/extrinsics로 camera→vehicle→world 3-D 및 covariance 산출 |
| depth | 유효범위·NaN 계약 불완전 | finite, min/max range, valid-pixel ratio, robust median/MAD |
| association | score와 optional 2-D prediction만 사용 | semantic·exemplar·3-D innovation·motion·age의 단위가 분리된 gate |
| ambiguity margin | predicted ordering과 raw-score margin이 혼재 | 동일한 final association score에서 margin을 계산하도록 수정·테스트 |
| tracking | track bank 없음 | ID 생성/유지/종료, occlusion coast, re-identification, covariance |
| safety | 그림에만 미래 경로 존재 | semantic output 변화가 safety output에 영향을 주지 않는 코드·테스트 |
| resolution/rate | 미동결 | 아래 Stage 2/4 측정 후 선택; 640×360, 1–2 Hz를 선결론으로 쓰지 않음 |

이 표가 모두 닫히기 전에는 “SAM 통합 완료”라고 부르지 않는다.

### 2026-09-03 기준선 감사 결과

- offline instance adapter 단위 테스트: 9/9 PASS
- 기존 perception 회귀: 31 PASS / 1 skip
- detect-resolution 회귀: 34/34 PASS
- 정적 사이트 계약, 내부 링크와 SVG 필수 문구: PASS
- Chrome headless에서 CURRENT overview와 후보 구조도 렌더 확인: 겹침·잘림 없음
- offline stub CLI와 `git diff --check`: PASS

이 결과는 Stage 0의 software/documentation 기준선만 확인한다. 실제 SAM 정확도, 실시간성,
Isaac 연결, closed-loop 성능 또는 실기 안전성을 검증한 결과가 아니다.

## 3. 검증의 기본 원칙

1. **세 단계 결과를 섞지 않는다:** offline perception, shadow closed-loop, active closed-loop.
2. **frame split을 금지한다:** 같은 episode의 인접 frame이 train/validation/test에 갈라지지 않도록
   scene/asset/episode seed 단위로 분리한다.
3. **threshold는 test 전에 동결한다:** pilot/validation으로 prompt, score, association margin을 정하고
   held-out set을 연 뒤에는 바꾸지 않는다.
4. **trajectory confound를 피한다:** detector 비교의 1차 인지 지표는 같은 저장 frame을 함께 평가한다.
   active closed-loop outcome은 서로 다른 frame distribution임을 별도로 보고한다.
5. **GT 방화벽을 유지한다:** simulator semantic ID와 target pose는 label/evaluation/critic에만 쓰고
   candidate selector나 actor 입력에는 넣지 않는다.
6. **fail closed:** worker timeout, stale timestamp, invalid depth, ambiguous identity는 새 lock을 만들지 않는다.
7. **safety 독립성:** detector가 죽거나 틀려도 raw occupancy와 emergency path는 계속 갱신되어야 한다.

## 4. 단계별 실행 계획과 게이트

### Stage 0 — 저장소·그림·회귀 계약

목적: 문서가 현재 구현과 후보 구조를 섞지 않고, 기존 실행 bytes가 변하지 않았는지 확인한다.

- `CURRENT SOFTWARE PATH` 그림은 160×90 single-detector/single-KF 경로를 나타낸다.
- 후보 그림은 `IMPLEMENTED`, `PLANNED`, `NOT IN CONTROL LOOP`를 시각적으로 구분한다.
- adapter 기본값 OFF, live perception import 없음, 기존 detector state dict loading 동일.
- 사이트의 모든 이미지·링크·alt text와 headless render를 검사한다.

Gate: site contract PASS, SVG title/desc·render PASS, existing perception tests PASS,
`git diff --check` PASS. 하나라도 실패하면 이후 단계 중지.

### Stage 1 — instance boundary 단위 계약

목적: 모델 종류와 무관하게 K개 mask가 합쳐지지 않는지 검증한다.

- 0/1/2/K blob, touching/diagonal components, hole, tiny component, border-touching mask.
- empty, NaN/Inf/0/negative/far-plane depth, depth edge가 mask를 가르는 경우.
- SAM 형식의 `K×H×W` mask, score, box를 round-trip하고 instance ID/order를 보존한다.
- `predicted_uv` 사용 시 정렬과 ambiguity margin이 **같은 association score**를 쓰는지 검사한다.

Gate: 합집합 호출 0회, fixture expected instances 100%, invalid-depth lock 0회,
serialization round-trip exact. 이 단계는 CPU만 사용한다.

### Stage 2 — 센서·데이터 해상도 게이트

목적: SAM 성능보다 먼저 표적 정보가 센서에 존재하는지 확인한다.

후보 해상도 `160×90 / 320×180 / 640×360 / 실카메라 native`를 다음 축에서 비교한다.

- range: `2–5 / 5–10 / 10–15 / 15–20 / 20–28 m`
- bars: `70 / 115 / 145 / 205` (205는 perception-only이며 policy mastery 주장이 아님)
- distractors: `N=0 / 1 / 3 / 5`
- similarity: 다른 색·다른 형상 / 같은 색·다른 형상 / 거의 동일 외형
- occlusion: `0 / 25 / 50 / 75% / temporary full`
- blur, illumination, texture, calibration error를 각각 단일축으로 추가

기록: visible target pixels, projected diameter, valid depth pixels, signal/noise ratio,
render/copy memory와 latency. 표적이 정보론적으로 보이지 않는 셀은 detector 실패와 구분해
`SENSOR_UNOBSERVABLE`로 보고한다.

Gate: 주 운용 거리에서 최소 가시 픽셀/valid-depth 기준을 먼저 동결한다. 기준을 만족하는 최저
해상도만 Stage 3 후보가 된다. 160×90을 관성적으로 본선으로 두지 않는다.

### Stage 3 — 실제 SAM 3.1 worker와 transport

목적: Isaac과 분리된 환경에서 재현 가능한 inference artifact를 만든다.

요청에는 schema version, frame ID, capture timestamp, RGB hash, prompt text, positive/negative
exemplar ID를 넣는다. 응답에는 model/checkpoint SHA-256, masks, boxes, scores, local instance IDs,
worker start/end timestamp를 넣는다. 늦게 도착한 과거 frame 응답은 폐기한다.

비교 arm:

1. text only
2. positive exemplar only
3. text + positive exemplar
4. text + positive/negative exemplars

Gate: request/response hash 일치 100%, corrupt/partial response lock 0회, timeout이 control thread를
block한 횟수 0회. 모델 라이선스와 정확한 checkpoint hash를 receipt에 포함한다.

### Stage 4 — 같은 프레임 open-loop 인지 평가

1차 지표:

- candidate recall/precision 및 target-instance recall
- FTLR = `(DISTRACTOR_LOCK + GHOST_LOCK) / detector-visible frames`
- `AMBIGUOUS`, false reject, ghost measurement 비율
- bearing MAE/95th, range MAE/95th, 3-D position error
- ID switch, fragmentation, full-occlusion 뒤 reacquisition time

현재 seed-479 결과와 수치를 직접 뺄셈 비교하지 않고, **동일 저장 프레임에 current/stub/SAM을
동시에 돌린 paired 결과**를 만든다. test set을 열기 전에 최소 채택 기준을 동결한다. 초기 권장
gate는 visible recall ≥95%, N=5 FTLR ≤10%, 10 m range median absolute error ≤0.30 m,
1 s miss 뒤 1 s 내 재획득 ≥80%다. pilot 결과를 본 뒤 이 수치를 바꾸려면 새 버전·새 held-out set이
필요하다.

### Stage 5 — 실시간성·SWaP 평가

두 deadline을 분리한다.

- fast tracker/association: 기존 10 Hz sensor-policy tick을 block하지 않아야 한다.
- SAM discovery: 비동기 deadline을 별도로 측정하며 stale 응답은 사용하지 않는다.

측정: cold/warm p50/p95/p99 latency, throughput, peak VRAM/RAM, host↔device copy, dropped frames,
GPU contention, power/thermal throttling. RTX 3070 수치는 개발기 결과로만 쓰고 탑재 후보에서 다시
측정한다.

Gate: fast path p99 <100 ms, control-thread blocking 0, OOM 0, stale response lock 0. SAM discovery
deadline은 Stage 4 reacquisition 결과와 함께 사전등록하며 임의로 `1–2 Hz`로 고정하지 않는다.

### Stage 6 — shadow-mode 폐루프

SAM/track bank를 Isaac frame에 실행하지만 결과는 actor, obstacle map, governor, reset, reward에
쓰지 않는다. baseline action/outcome과 shadow-off action/outcome은 같은 seed에서 exact하게 같아야
한다. 동시에 end-to-end timestamp, queue age, track lifecycle을 기록한다.

Gate: action bytes와 termination cause exact parity, RNG state drift 0, control stall 0. 실패하면
active 연결 금지.

### Stage 7 — active closed-loop 단일변수 A/B

별도 사전등록과 권한 이후에만 candidate target-history 입력을 활성화한다. 첫 비교에서는 policy,
reward, spawn, target motion, safety layer를 고정하고 perception source만 바꾼다. 기존 정책과 detector
통계 coupling 가능성이 있으므로 frozen-policy 평가와 fresh-policy 학습을 같은 주장으로 합치지 않는다.

지표: capture/crash/timeout, never-acquired, FTLR, ambiguity duty cycle, target-lost duration,
bar contact, intervention, perception latency strata. `N=0/1/3/5`와 density별로 따로 보고한다.

Gate 예시: N=5 FTLR absolute gate 통과, capture difference의 95% CI lower bound ≥−2 pp,
crash difference의 upper bound ≤+2 pp, timeout 증가 ≤2 pp. 정확한 episode 수·multiplicity correction은
실행 전에 power analysis로 동결한다.

### Stage 8 — 경량화·distillation

SAM이 정확하지만 deadline/SWaP를 못 맞출 때만 연다. SAM mask와 hard-negative decision을 teacher로
사용해 경량 instance detector/tracker를 학습한다. teacher와 student는 동일 held-out test로 선택하지
않고 validation 뒤 test를 한 번만 연다. SAM보다 빠르다는 이유만으로 채택하지 않고 Stage 4–7의
인지·폐루프 gate를 동일하게 적용한다.

### Stage 9 — 실제 센서와 비행

순서: bench replay → 손으로 움직이는 센서 rig → prop-off assembled vehicle → tether/net 저속 hover →
open-area target following → sparse obstacles → density 상승. 각 단계에서 AUW/CG, camera/LiDAR
extrinsics, clock synchronization, rolling shutter, exposure, thermal/power 로그를 남긴다.

실기에서 simulator prompt/threshold를 그대로 쓰지 않는다. real validation으로 한 번 동결한 뒤
held-out route를 평가한다. perception gate를 통과해도 detector 출력을 collision safety에 사용하지 않는다.

## 5. 필수 ablation

| 질문 | 한 번에 바꿀 축 |
|---|---|
| SAM 의미가 필요한가 | CC-only vs SAM masks, 동일 association |
| exemplar가 필요한가 | text vs positive vs positive+negative exemplar |
| depth가 돕는가 | semantic-only vs +candidate depth gate |
| motion이 돕는가 | +3-D prediction gate |
| 다중 트랙이 필요한가 | single forced lock vs track bank+AMBIGUOUS |
| video memory가 돕는가 | periodic image discovery vs propagation |
| 해상도가 병목인가 | 동일 모델, 해상도만 변경 |
| 안전 분리가 되었나 | semantic mask/ID를 교란하고 safety output parity 확인 |

여러 축을 한 arm에서 동시에 바꾸지 않는다. full 후보는 마지막 조합 확인용이지 각 기여의 인과
증거가 아니다.

## 6. 산출물과 provenance

각 단계는 다음을 함께 보존한다.

- preregistration과 frozen config
- source commit 및 dirty-tree manifest
- model/checkpoint/prompt/exemplar SHA-256
- Python/PyTorch/CUDA/driver/GPU와 worker environment lock
- input episode/frame IDs, dataset split manifest와 label provenance
- raw per-frame candidates/tracks/decisions/timestamps
- summary, confidence interval, failure gallery와 machine-verifiable receipt
- README/사이트에는 `IMPLEMENTED`, `MEASURED`, `ADOPTED`를 분리한 문구와 그림

## 7. 중지/진행 결정

```text
Stage 1 구조 실패        → adapter 수정, SAM 실행 금지
Stage 2 비가시            → 카메라/FOV/해상도 수정, 모델 튜닝 금지
Stage 4 의미 실패         → prompt/exemplar/domain data 검토, closed-loop 금지
Stage 5 deadline 실패     → async 주기 조정 또는 distillation, active 연결 금지
Stage 6 parity 실패       → queue/RNG/timestamp 수정, A/B 금지
Stage 7 안전·outcome 실패 → candidate 미채택; 실패 결과 보존
Stage 7 통과              → 별도 fresh-policy/실기 사전등록 가능
```

현재 다음 실행점은 **Stage 1의 누락 edge case와 association-score 일관성 보완**, 이어서 **Stage 2의
해상도별 저장 프레임 데이터 계약 작성**이다. 실제 SAM 설치나 PPO보다 이 두 단계가 먼저다.
