# Perception 재설계 계획 — "빨간 것 찾기"에서 "UAV 형상 인식 + 시간적 연관"으로

> ## ⛔ 상태: SUPERSEDED (2026-09-04) — S2~S6는 실행하지 않는다
>
> 이 계획의 **문제 정의(§0의 1·7)는 측정으로 확증됐고, 실행 경로(S2~S6)는 폐기**됐다.
> 사후 판단이 아니라 이 계획이 스스로 넣은 게이트(S1)와 후속 사전등록 하나가 낸 결과다.
>
> **확증된 것** — "빨간 픽셀 찾기"가 아니라 "물체 동일성"이 문제라는 재정의:
>
> | 무엇을 바꿨나 | capture 효과 | 판정 | 출처 |
> |---|---:|---|---|
> | 자료구조 (CC 다중 후보 + χ²(3) 게이팅) | 없음 (FTLR **+1.3 pp**) | `RECOGNITION_DOMINANT` | `prereg_2026-09-03_s1_structure_fix_shadow.md` |
> | 거리 측정 분산 (물리적으로 옳은 2차 모델, 20 m에서 2.3배) | 없음 (**−0.34 pp**, CI [−3.19,+2.51]) | `VARIANCE_INSENSITIVE` | `prereg_2026-09-04_depth_noise_model_order.md` |
> | 어느 물체를 lock 하는가 (동색 디코이 5개) | **−55.5 pp** | `COLOR_SHORTCUT_CONFIRMED` | `prereg_2026-09-01_distractor_envelope.md` |
>
> **폐기된 것** — 그 인식기를 **우리 시뮬 이미지로 학습**한다는 실행 경로. 사유 둘:
>
> 1. **표적에 형상 정보가 물리적으로 없다.** detector가 보는 표적은 반지름 0.15 m **해석적 구**에
>    상수색 `[0.88, 0.08, 0.045]`를 칠한 것이고(`navrl_detector.py:321,551`), 디코이 3종 중 하나는
>    **같은 반지름의 구**다(`env_object_config.py:938`). 배경은 40×24 depth(최대 10 m)를 업샘플한
>    회색 명암이며 텍스처·조명·그림자·재질이 전혀 없다(`navrl_detector.py:558,876`). 저장소 전체에
>    쿼드로터 메쉬가 존재하지 않는다 — 우리 기체 자신의 visual도 `sphere radius="0.05"`다.
>    → **S2의 "quad-mesh 표적 visual 재사용"은 실행 불가능한 선택지였다**(메쉬가 없다).
> 2. **시뮬 이미지로 학습한 detector는 전이되지 않는다.** 일반 시뮬(Unreal+Colosseum)로 학습한
>    tiny-YOLOv4는 실제 저조도에서 mAP **37.2 %**, 실사진 재구성 기반은 96.4 %다
>    (Ning et al., *Unmanned Systems* 12(4), 2024). 우리 렌더러는 그 "일반 시뮬"보다 열악하다.
>
> **대체 경로**: 인지는 **실제 공대공 데이터**로 학습하고(NPS-Drones BSD-3 / Det-Fly MIT), 시뮬에는
> 그 detector의 **측정된 오차 모델**을 주입해 정책을 학습시킨다. 이는 Swift(Nature 2023)가 실비행
> 50초 잔차로 GP 관측모델을 적합해 정책만 미세조정한 것과 같은 구조이며, NavRL(RA-L 2025)을 포함해
> 이 계열 논문들이 건너뛴 단계다. 근거·데이터셋 라이선스·차단 항목은 `README.md`의 *External data*
> 절과 `docs/status/index.html`의 `05 · EXTERNAL DATA` 섹션에 있다.
>
> **살아남은 것**: §0의 문제 재정의(1), KF와 신경망의 역할 분리(7), episode-level split 원칙(5),
> hard negative 설계(4)는 실데이터 경로에서 그대로 유효하다. 폐기된 것은 **데이터를 어디서
> 얻는가**이지 **무엇을 배워야 하는가**가 아니다.
>
> WORKLOG 2026-09-03(S1 판정) · 2026-09-04(Phase 1 판정) 참조.

작성 2026-09-03. **실행 전 계획안이다.** 채택된 detector, 898-D 계약, checkpoint, 기존 판정을
바꾸지 않으며, GPU 단계는 각각 별도 사전등록 후에만 연다.
사용자 제안(40개 절, CNN=공간 / GRU·Transformer=시간 역할 분리)을 기초로 하되, MOTAR의 실측
제약(RTX 3070 8 GB, 128~256 병렬 카메라, 기존 하네스)에 맞춰 재구성하고 단계 하나를 **추가**한다.

관련 문서: `docs/SAM3_PERCEPTION_VERIFICATION_PLAN_2026-09-03.md`(후보 보존 구조·adapter 경계),
`docs/prereg_2026-09-01_distractor_envelope.md`(FTLR 측정 하네스),
`results/navrl_detector_distractor_envelope_seed479/`(현 결함의 정량화).

---

## 0. 채택하는 판단과 수정하는 판단

**그대로 채택** (사용자 제안):

1. 문제 재정의: "이 픽셀이 빨간가" → "이 물체가 UAV인가" + "그 UAV가 내가 쫓던 그놈인가".
   공간(외형)과 시간(연관)은 **서로 다른 문제**이고 모듈을 나눈다.
2. CNN→RNN→LSTM→Transformer 직렬 스택 금지. 공간은 CNN 하나, 시간은 **GRU 또는 Transformer 중
   하나**. 첫 시간 모델은 GRU(T=8), Transformer(T=16)는 비교군.
3. full-image Transformer 금지 — CNN이 압축한 candidate token만 (K=5·T=16이면 80 token).
4. 색-형상 독립 학습(target 색 랜덤화) + hard negative(동색 비UAV / 이색 UAV / 동색 UAV).
5. episode-level split + unseen-mesh test. frame 단위 split 금지.
6. Frozen perception → PPO. joint training은 맨 마지막에만 고려.
7. KF 유지 — 신경망은 "누가 표적인가", KF는 "표적이 어디 있나".

**수정/추가** (이 문서의 기여):

- **S1 단계를 새로 넣는다**: 신경망 교체 전에 connected-component 후보 + KF gating 연관만으로
  기존 detector를 다시 평가한다. 이유는 §2.
- **P0"해상도 올리기"를 두 개의 게이트로 재정의한다**: (a) 렌더 비용 실측, (b) **협시야(foveated)
  대안 평가**. 640×360 전면 인상은 8 GB에서 closed-loop이 성립하지 않을 가능성이 높다. §3.
- SAM 트랙과 **연관 계층을 공유**한다: 후보 → 연관 → KF 인터페이스를 SAM adapter의 packet 계약에
  맞춘다. 두 세션이 연관 스택을 두 벌 만들지 않는다. §5.
- 30 Hz vision 멀티레이트(§36 제안)는 **하드웨어 단계로 연기** — sim에서는 카메라가 정책 주기로
  렌더되므로 렌더 비용 3배에 대응하는 sim 편익이 없다.
- 정책 관측 확장(§35 제안: confidence·불확실성·t_seen)은 898-D 계약 파괴 = **새 계보 결정**이므로
  S6에서만 다룬다. age·covariance는 이미 토큰에 있다.

---

## 1. 현재 상태 (반복하지 않기 위한 고정점)

- 탐지: 1×1 conv 색 분류기 → **양성 픽셀 전부를 단일 중심점으로 붕괴**(CC 없음) → LiDAR 연관
  (±15°, ±0.55 m) → KF. `navrl_perception.py`.
- 실측 결함: default·v7 모두 `COLOR_SHORTCUT_CONFIRMED`. v7 FTLR **90.27 %** @N=5.
  평균 양성 픽셀 147–181, 표적 자체는 2–5 px — 중심점은 어느 물체 위에도 없다.
- 정보량: 160×90 @87° = **84.3 px/rad** → 0.30 m 표적이 20 m에서 1.27 px, 28 m에서 0.90 px.
- 이미 있는 것: `detect_width/height` 분리 훅(`detect_decoupled`), 벡터화 KF tracker,
  obs-dump 훅(오프라인 데이터셋 덤프), distractor-envelope FTLR 하네스, detector 항법 A/B
  비열등 게이트(learned-v2 vs analytic, −2 pp margin), SAM 오프라인 CPU adapter(CC·K후보·
  AMBIGUOUS/REJECT 보존).

---

## 2. 왜 S1(구조 수정 단독 평가)이 먼저인가 — 귀속 논증

FTLR 90.27 %에는 **두 원인이 합쳐져 있다**(SAM 계획 §1의 구분과 동일):

- (a) **인식 실패** — 특징이 색뿐이라 동색 물체를 구분 못 함.
- (b) **자료구조 실패** — 후보가 하나뿐이라 여러 양성 영역이 한 중심점으로 붕괴.

CNN+FPN로 직행하면 FTLR이 떨어져도 (a)를 고친 덕인지 (b)를 고친 덕인지 **영원히 알 수 없다**.
사용자 제안 §31의 목표("무엇이 기여했는지 정확히 설명")를 지키려면 (b)만 먼저 고치고 측정해야 한다.

S1의 내용: 기존 detector(v7·default 그대로, 재학습 없음) 위에 CC 라벨링 → 후보별
[u,v,w,h,count,depth] → KF 예측 게이트(Mahalanobis) 연관 → 게이트 밖 후보 기각.
정적 동색 sphere는 위치가 다르므로 track 연속성만으로 대부분 기각될 것으로 **예측**한다.
잔여 FTLR(교차·근접 케이스)이 곧 형상·시간 모델의 **정직한 동기**가 된다.

예측을 사전등록한다: S1 후 FTLR@N=5가 90.27 %에서 크게 떨어지면(예: <30 %) 결함의 주인은
자료구조였고, 남으면(>60 %) 인식이었다. 어느 쪽이든 논문 서사가 선다.

비용: 신경망 0, 해상도 변경 0, 오프라인 재생 + 기존 하네스 재실행. **가장 싼 최대 정보 실험.**

---

## 3. S0 — 정보량·비용 게이트 (신경망 이전, GPU 1~2시간)

### S0-a. 렌더 비용 곡선 실측

160×90 / 320×180 / 640×360 × num_envs {32, 64, 128}에서 VRAM과 step 시간 측정. 3070 기준
closed-loop 학습이 성립하는 (해상도, env 수) 경계를 확정한다. 이 숫자 없이는 "해상도를 올린다"가
계획이 아니다. (오프라인 데이터셋 생성은 env 8~16으로 충분하므로 어느 해상도든 가능.)

### S0-b. 협시야(foveated) 대안 — 같은 픽셀 예산으로 각해상도 5.4배

핵심 산수: 각해상도 = (W/2)/tan(HFOV/2).

| 스트림 | px/rad | 0.30 m 표적 @20 m | @28 m |
|---|---:|---:|---:|
| 160×90 @87° (현재) | 84.3 | 1.27 px | 0.90 px |
| 320×180 @87° | 168.6 | 2.5 px | 1.8 px |
| 640×360 @87° | 337 | 5.1 px | 3.6 px |
| **160×90 @20° (협시야)** | **453** | **6.8 px** | **4.9 px** |
| 320×180 @20° | 907 | 13.6 px | 9.7 px |

**같은 160×90 픽셀 예산의 협시야 카메라가 640×360 전면 인상보다 각해상도가 높다.**
Johnson 기준(CNN 신뢰 8–10 px)에 20 m에서 근접하고, 320×180 협시야면 확실히 넘는다.
비용은 env당 두 번째 소형 렌더(+1×160×90)뿐. 대가는 조준 문제 — 광각 SEARCH → 협시야 TRACK의
상태기계(사용자 제안 §22–23)와 정확히 맞물린다.

S0-b 판단 규칙: S0-a에서 640×360×128env가 성립하면 전면 인상도 후보로 유지, 아니면
**협시야 이중 스트림을 기본 경로로 채택**한다.

주의: 각해상도를 올려도 detector max range 20 m clip이 다시 binding한다
(detection Stage 1 `RANGE_INCONCLUSIVE` 스레드와 접점 — 해상도 결정과 함께 재론).

---

## 4. 단계 사다리 (각 단계 = 별도 사전등록, 이전 단계 동결 후 진행)

| 단계 | 내용 | 신경망 | GPU | 판정 기준(사전등록 시 고정) |
|---|---|---|---|---|
| **S0** | 렌더 비용 곡선 + 협시야 결정 | – | 1~2 h | 성립 경계표; foveated 채택 여부 |
| **S1** | CC 후보 + KF gating, 기존 detector 재평가 | 없음 | ~2 h (재생·평가) | FTLR@N=5 귀속: <30 % 구조 / >60 % 인식 |
| **S2** | 데이터셋 v1: 색 랜덤화, hard negative 3종, episode split, unseen-mesh 홀드아웃 | – | 수 h (덤프) | 셀당 프레임 수·클래스 균형 계약 |
| **S3** | 단일 프레임 형상 detector: 소형 CNN+FPN(≤1 M param), top-K=5, [u,v,w,h,c,e64] | 신규 | 수 h (오프라인) | 색-셔플 test에서 recall/FP; SAM 후보를 상한 baseline으로 |
| **S4** | 시간 연관 (detector 동결): A) KF-gating만(S1 이관) B) +GRU T=8 C) +Transformer T=16, d=128/4h/2L; T∈{4,8,16,32} 스윕 포함 | 신규(소형) | 소 | association 정확도·FTLR·지연; C가 B를 유의하게 못 이기면 B 채택 |
| **S5** | hard-distractor 벤치마크: ①동색 비UAV ②이색 UAV ③동색 UAV ④교차 ⑤가림 | – | 수 h | 조건×모델 FTLR 표 = 논문 메인 테이블 |
| **S6** | closed-loop: frozen 최선 perception → PPO A/B | – | 수 일 | 기존 비열등 게이트(−2 pp) 재사용 + FTLR 게이트; 토큰 확장(=새 계보) 여부 결정 |

원칙: **S3 이전에 S1 결과를 본다.** S1이 FTLR을 이미 크게 잡으면 S3~S5는 "더 어려운 조건"
(UAV형 디코이·교차)을 향한 연구가 되고, 못 잡으면 형상 인식이 주 서사가 된다.

GRU vs LSTM: GRU만. LSTM은 동절 비교군으로서 가치가 낮다(같은 계열, 파라미터만 많음).
Transformer의 존재 이유는 **가림 후 재획득**뿐이다 — 우리 timeout cohort의
visible↔hidden 전이가 에피소드당 7.7회로 가림이 실재하므로 비교군 유지가 정당하다.

---

## 5. SAM 트랙과의 역할 분담 (세션 간 충돌 방지)

- **공유 계층**: 후보 → 3-D 연관 → track bank 인터페이스는 SAM adapter의 packet 계약
  (`bbox, score, uv, depth_median`, K후보, AMBIGUOUS/REJECT)을 그대로 쓴다. S1의 CC 후보도
  같은 packet으로 내보낸다. **연관 스택은 한 벌만 존재한다.**
- **SAM의 위치**: zero-shot 후보 생성기의 **오프라인 상한 baseline**(S3에서 소형 CNN+FPN과
  같은 데이터셋·같은 지표로 비교). 실시간 배포 후보가 아니다(별도 프로세스·지연 계약은 SAM
  계획 문서의 소관).
- 이 계획이 만드는 것: S1 CC+gating, S2 데이터셋, S3 소형 detector, S4 시간 모델.
  SAM 계획이 만드는 것: worker/IPC/timestamp/3-D lifting. 겹치지 않는다.

---

## 6. 자원·리스크

- **최대 비용 항목은 신경망이 아니라 S2 데이터셋/자산이다**: UAV mesh 다양화(현재 표적 mesh
  1종), 텍스처·조명 변주, negative 배치. mesh 소싱이 실질 작업. 단, `resources/robots/**`·
  `aerial_gym/config/robot_config/**`는 provenance-frozen이므로 **표적/디코이 자산은 반드시
  그 밖에서** 추가한다(D9 distractor 자산 방식과 동일 — 선례 있음).
- 8 GB 한계: 오프라인 학습은 배치 스트림이라 여유. closed-loop은 S0 결과에 종속.
  1650 Ti(4 GB)는 GPU4GB 경로로 평가만 — 기기 간 수치 혼용 금지 유지.
- 색 랜덤화는 기존 painting 경로 재사용(디코이 페인팅 선례). 저비용.
- 기존 판정 보호: P2 STRICT FAIL·D1 FAIL·P3 BLOCKED 불변. 이 계획은 Track A 서사와 독립.

## 7. 연구 질문 (사용자 제안 §39 채택, S1 귀속 질문 추가)

- **RQ0 (신설)**: 현 FTLR 90.27 %의 지배 원인은 자료구조(단일 후보)인가 인식(색 특징)인가?
- RQ1: 색 단서 없이 형상 detector가 UAV를 식별할 수 있는가? (해상도별)
- RQ2: 강건한 표적 연관에 시간 정보가 얼마나 필요한가? (KF vs GRU vs Transformer, T 스윕)
- RQ3: 외형이 모호할 때 운동 이력이 false target lock을 줄이는가?
- RQ4: 해상도·이력 길이가 정확도·지연과 어떻게 교환되는가?
- RQ5: 연관 개선이 closed-loop 요격 성공률로 이어지는가?

## 8. 즉시 실행 가능한 첫 두 수 (승인 시)

1. **S0-a 렌더 비용 스크립트** — 사전등록 불요(성능 주장 아님, 엔지니어링 측정). 반나절.
2. **S1 사전등록 초안** — RQ0 판정 규칙(30/60 % 경계), 재생 소스(seed 479 셀), 게이트 파라미터
   (Mahalanobis 임계)와 함께. 승인 후 실행 ~1일.
