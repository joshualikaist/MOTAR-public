# 사전등록 — 오프라인 instance adapter CPU 게이트

작성 2026-09-03, **어떤 SAM 추론·Isaac 롤아웃·PPO도 하기 전**. 결과를 본 뒤 임계·판정 규칙을
바꾸지 않는다.

## 1. 질문

현행 `_detect_rgbd`는 양성 픽셀 전부를 하나의 무게중심으로 축약한다
(`navrl_perception.py`). 동색 blob이 두 개면 중심점은 둘 사이 허공을 가리킨다. 이것이
distractor envelope의 `COLOR_SHORTCUT_CONFIRMED` / v7 FTLR 90.27%의 구조적 원인이다.

이 단계에서 묻는 것은 SAM 3 성능이 아니다. **SAM 가중치가 없어도**, K개 instance를 합치지
않고 후보별 depth/score를 보존하는 계약이 CPU에서 깨지지 않는가?

## 2. 계산이 예측하는 것 (측정 전)

같은 색·같은 픽셀 수의 blob 두 개가 서로 다른 depth에 있으면:

- 현행 union: `u`는 두 blob 사이, `range`는 두 depth의 픽셀 가중 평균 → 어느 물체도 아닌
  ghost centroid.
- stub adapter (색 임계 + connected components): instance 2개를 유지하고, 각 `uv`와
  `depth_median`은 자기 blob만 본다.
- 1등과 2등 score 차가 `ambiguous_margin`보다 작으면 `AMBIGUOUS`이고 lock id는 없다.

예측이 빗나가면(예: CC가 blob을 합치거나 ghost가 한 blob 안에 떨어짐) 그 자체가 발견이며,
임계를 사후 조정하지 않는다.

## 3. 실험 계약

| 항목 | 값 |
|---|---|
| 성격 | **CPU unittest + 오프라인 툴.** 검출기 수정·재학습·PPO·Isaac·SAM 설치 권한 없음 |
| 입력 | synthetic 2-blob RGB-D fixture (저장 프레임 불필요) |
| 현행 경로 | `AppearanceTargetSegmenter` / `_detect_rgbd` **호출하지 않고** 동일 합 공식만 재현 |
| 후보 경로 | `navrl_instance_adapter.py` stub backend |
| 제어루프 | `NAVRL_INSTANCE_ADAPTER` 기본 0. perception module은 어댑터를 import하지 않음 |
| SAM 백엔드 | 인터페이스만 정의. 같은 프로세스에 가중치 로드 시 fail-closed |

## 4. 1차 지표 (판정)

CPU 게이트 세 개. 숫자 성능 게이트가 아니다.

| Gate | PASS 조건 |
|---|---|
| G1 분리 | 두 빨간 blob fixture에서 stub가 instance **2개**를 반환 |
| G2 ghost | 동일 fixture의 union centroid가 두 instance `uv` 사이(양쪽 blob bbox 밖)에 있음 |
| G3 거부 | score 차가 `ambiguous_margin` 미만이면 `decision=AMBIGUOUS` 이고 `selected_id is None` |

보조(판정 아님): 각 instance의 `depth_median`이 서로 다름, 빈 입력이 `REJECT`.

## 5. 명시적으로 묻지 않는 것

SAM 3 FTLR, 실시간 Hz, capture/crash/timeout, sim-to-real, 160×90에서 foundation model이
2px 표적을 살리는지. 기존 seed-479 distractor 궤적과 detector를 비교하지 않는다
(`prereg_2026-09-01_distractor_envelope.md` §3-c).

## 6. 통과 후 열리는 것 / 안 열리는 것

PASS는 “합집합 붕괴를 피하는 출력 계약이 CPU에 존재한다”는 뜻이다. 다음을 열지 않는다:

- Track A GPU, hardware/real-log 없는 평가
- Isaac 제어루프에 SAM 또는 stub 연결
- 256-env PPO, S1 search-state와 축 혼합
- SAM 3.1 설치 (별도 conda·별도 프로세스·별도 사전등록)

## 7. 실행

```bash
export PYTHONNOUSERSITE=1
python -m unittest tests.test_navrl_instance_adapter
python tools/run_navrl_instance_adapter_offline.py --backend stub
```
