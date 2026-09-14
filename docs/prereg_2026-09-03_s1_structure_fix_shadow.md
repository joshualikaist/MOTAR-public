# 사전등록 초안 — S1: 자료구조 수정(CC 후보 + KF gating)의 shadow-mode 귀속 측정

작성 2026-09-03. **초안 — 사용자 승인 전 실행 금지.** 상위 계획
`docs/plans/perception_shape_temporal_redesign_2026-09-03.md` §2·§4의 S1 단계.

## 0. 질문 (RQ0)

FTLR 90.27 %(v7, N=5, seed 479/1650 Ti)의 지배 원인은 무엇인가?

- (a) **인식 실패** — 특징이 색뿐이라 동색 물체를 구분 못 함
- (b) **자료구조 실패** — 후보가 하나라 여러 양성 영역이 단일 중심점으로 붕괴

(b)만 고친 counterfactual을 같은 프레임에서 측정해 귀속한다.

## 1. 설계 — shadow-mode (프레임 덤프 아님)

표준 v7 distractor-envelope 평가를 그대로 실행한다. 정책은 기존 단일-중심점 경로를 그대로
소비한다(**궤적 불변**). 매 스텝, 같은 score map에 대해 병렬로:

1. `score ≥ pixel_threshold` 마스크에 **connected-component 라벨링** → 후보별
   `[u, v, w, h, count, depth_median]`
2. shadow KF(기존 벡터화 tracker와 동일 클래스 `BatchedConstantVelocityTracker`, 별도
   인스턴스)의 예측 게이트로 후보 연관; 게이트 안 다수면 최소 거리. 게이트는 3-D world
   위치의 **대각근사 Mahalanobis** `Σᵢ eᵢ²/(Pᵢᵢ+Rᵢᵢ) ≤ χ²(3 dof, 0.99) = 11.345`
   (초안 v1의 "χ²(2 dof) 9.21"은 게이트 차원 오기재 — **실행 전 정정**, 2026-09-03)
3. 연관 실패 시 track age 증가, `NAVRL_TRACKER_MEMORY_S`(5.0 s) 초과 시 track 해제
4. **track 부재 시 초기화**: 최대 `count` 후보로 시작 (색 정보뿐이므로 여기서는 디코이
   선택 가능 — 의도된 한계, §4 예측 2)
5. shadow가 이번 프레임에 잡은 위치를 기존 GT 분류기(반경 0.5 m,
   TARGET/DISTRACTOR_LOCK/GHOST_LOCK)로 분류해 **기록만** 한다

actor 관측·행동·보상·종료 경로에는 어떤 쓰기도 하지 않는다.

## 2. 고정 조건

| 항목 | 값 |
|---|---|
| detector | **v7** (sha `85c7974b…`, threshold 0.700 — envelope 사전등록과 동일) |
| 셀 | N ∈ {0, 1, 3, 5} distractors, 4 cells |
| 환경 | 70 bars 등 distractor-envelope 사전등록(§3) 조건 그대로 |
| seed | **481** (미사용; 479는 1650 Ti 산이므로 기기 혼용 금지) |
| 기기 | RTX 3070 |
| episodes/cell | 2,049 |
| shadow 파라미터 | gate χ²(3)=11.345 · top-K=5 · init=argmax count · memory 5.0 s — **본 문서로 동결, 사후 튜닝 금지** |
| 실행 방식 | envelope 런처의 closed env 빌더를 임포트 재사용, 오버라이드는 seed·result dir·`NAVRL_S1_SHADOW`·shared source bundle 경로 **네 필드뿐** (`tools/run_navrl_s1_shadow.py`; 번들 오버라이드는 seed479 번들의 1650 Ti 절대 root가 이 기계에서 검증 불가하기 때문 — 신선한 3070 번들을 스윕이 생성) |
| 주의 | num_envs는 3070 기본(128)을 따르며 seed479(1650 Ti, 64)와 다름 — seed479 수치와의 비교는 방향성 참고로만 |

## 3. 판정 규칙 (결과 이전 동결)

주지표: **shadow FTLR@N=5** (같은 run의 online FTLR@N=5 = 동일 프레임 baseline).

- `STRUCTURE_DOMINANT`: shadow FTLR@N=5 **< 30 %** — 결함의 주인은 단일-후보 붕괴였다.
  S3(형상 detector)의 동기는 "잔여 오류"로 재정의된다.
- `RECOGNITION_DOMINANT`: shadow FTLR@N=5 **> 60 %** — 구조를 고쳐도 색 특징으로는 부족.
  S3가 주 서사가 된다.
- `MIXED`: 30–60 % — 두 축 모두 논문 서사에 필요.

경계 근거: 30 %는 online 90.27 %의 1/3 미만(구조 수정만으로 오류의 2/3 이상 제거 = 지배),
60 %는 2/3 초과 잔존(= 구조 기여가 소수). 중간은 정직하게 MIXED로 둔다.

부지표(게이트 아님, 기제 확인): shadow lock을 **초기화 프레임**(track 부재에서 첫 연관)과
**추적 프레임**으로 분해 기록.

## 4. 예측 (기제 확인용)

1. 정적 디코이는 위치가 고정이므로 **추적 프레임**의 shadow DISTRACTOR_LOCK은 크게 감소한다.
2. **초기화 프레임**의 오선택은 남는다 — track 부재 시 색 count 말고는 단서가 없다.
   이 잔여가 S3~S4(형상·시간 모델)의 정량적 동기다.
3. N=0 셀의 shadow GHOST_LOCK ≈ online GHOST_LOCK (구조 수정이 무해함의 sanity).

## 5. 기계 무결성 게이트 (fail-closed)

- **M1 (영향 없음 증명)**: 50-episode 스모크 2회(shadow off/on, 동일 seed)의
  `NAVRL_OBS_DUMP` npz에서 **덤프 자기참조 필드(`run_obs_dump_path`, `run_pid`)를 제외한
  전 배열의 canonical SHA-256이 동일**해야 한다. 불일치 시 본 측정 실행 금지.
  런처가 M1 PASS 기록 없이는 evaluate를 거부한다(fail-closed).
  (정정 2026-09-03, 측정 전: 초안의 "파일 SHA-256"은 덤프 파일명·PID가 파일 안에 기록되어
  구조적으로 불일치 — 첫 스모크에서 obs 9,728×898 전 프레임 비트 동일을 확인한 뒤
  비교 대상을 의미 배열로 명시했다. 판정 경계·측정 조건 무변경.)
- **M2**: online FTLR 경로의 코드는 무변경(diff로 확인 가능해야 함).
- shadow 실패(예외)는 셀 VOID — 조용한 skip 금지.

## 6. 산출물

`results/navrl_s1_shadow_seed481/{n0,n1,n3,n5}/` + `summary.{md,json}`
(verdict 키: `verdict_rq0`, 단수 `verdict` 없음). WORKLOG 항목.

## 7. 명시적 범위 밖

색 특징 변경(재학습) 없음 · 해상도 변경 없음 · 정책 경로 변경 없음 · shadow 파라미터 스윕 없음
(χ²·초기화 규칙의 대안 비교는 S4-A에서 사전등록 후).
