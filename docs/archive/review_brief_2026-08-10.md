# 검수 요청 브리프 — 2026-08-05 ~ 08-10 작업 (R3 latency/dropout + 대시보드 + v2 재측정)

> **검수 완료 / 역사 입력 문서.** 최종 판정과 후속 코드 감사 결과는
> `docs/codex_review_2026-08-10.md`를 따른다. 특히 speed×density interaction, H4 69% 분해,
> 601-action horizon과 누락된 rl_games `time_outs` bootstrap은 이 문서의 초기 상태를 supersede한다.

수신: 검수 세션(Codex) / 작성: Claude 세션
대상 커밋 범위: `715dc76..HEAD` (브랜치 `research/navrl-env`, **push 안 함**)

이 문서는 "무엇을 했는가"가 아니라 **"어디가 틀렸을 가능성이 높은가"**를 적은 것이다.
자화자찬은 생략하고 반증 포인트만 나열한다. 전체 서사는 `docs/midterm_summary_2026-08.md`,
날짜별 원본은 `WORKLOG.md` 2026-08-05 이후 항목에 있다.

---

## 0. 한 줄 요약

R3의 "detection latency가 1순위 인지 병목(−42.7pp)"이라는 판정을 **기각**했다. 원인은 지연
자체가 아니라 **지연된 기체-프레임 측정을 현재 pose로 월드 변환하던 모델링 결함**이었고,
수정 후 진짜 비용은 −2.5pp다. 이어서 dropout(−12.7pp) 채널을 5회 A/B로 분해했고, 대시보드가
v1 데이터로 현재 태스크를 설명하던 문제를 고친 뒤 v2 헤드라인 그림을 재측정했다.

**PPO 재학습은 전 구간에서 0회.** 모두 frozen ep25000+riskcap 위 inference-only 평가다.

---

## 1. 최우선 검수 대상 — 여기가 틀리면 나머지가 무너진다

### (A) P3 ego-motion 수정의 정당성 — `navrl_perception.py::observe()`

**주장**: 지연된 detection은 t−τ의 기체 프레임에서 관측된 값인데 코드가 **t의 pose로** 월드
변환했다. 따라서 드론 자신의 운동이 매 KF 보정에 주입된다(병진 0.233 m, yaw 0.408 m @
실측 평균 2.33 m/s / 0.81 rad/s). 취득 시점 pose로 변환하면 latency 0.1 s 손실의 94%가 회수된다
(37.82% → 78.04%).

**검수 포인트**
1. "취득 시점 pose로 변환"이 실기 표준 관행이라는 전제가 맞는가? 아니면 내가 시뮬레이터에
   유리한 가정을 넣은 것인가? — 이게 틀리면 R3 supersession 전체가 무효다.
2. pose 링 버퍼가 measurement와 **동일한 read index**를 쓰는지
   (`_apply_detection_latency` 내부, 인덱스 산술을 두 번 쓰지 않으려 했음).
3. 버퍼 미충전 첫 스텝의 init pose가 `visible=False`로 보호되는지.
4. `NAVRL_LATENCY_EGO_MOTION_FIX` **기본 ON 승격**이 타당한가? τ=0에서 산술적 no-op이라
   기존 비-latency 수치는 불변이라고 주장했다 — 검증 요망.
5. 테스트가 주장을 실제로 검증하는가: `tests/test_navrl_latency_compensate.py::LatencyEgoMotionFix`
   (월드 고정 표적에서 잔차가 **정확히 0**이 되는지).

### (B) v2 밀도 곡선의 confound — 내가 이미 한 번 틀렸던 곳

`results/navrl_v2_density_curve_riskcap` vs `results/navrl_v2_ep24000_heldout` 비교에서
Δ +4.54 → +9.27pp를 처음에 "riskcap 이득"이라고 썼다가 **자체 정정**했다. 두 arm은
체크포인트(ep24000 vs ep25000)와 governor(off vs riskcap)가 **동시에** 다르다.

**검수 포인트**
1. 정정된 서술(WORKLOG 08-10, `docs/midterm_summary_2026-08.md`)이 충분히 보수적인가?
2. 205막대 분리값(governor 단독 +8.17pp / 적응 단독 +3.74pp, seed45)을 다른 밀도로
   외삽하는 문장이 남아 있지 않은가?
3. 분리하려면 ep24000+riskcap을 같은 5밀도에서 재야 한다고 적었다 — 이 설계가 맞는가?

### (C) 속도×밀도 상호작용 주장 — 논문 서술을 바꾸는 근거

**주장**: 0.3→1.5 m/s 비용이 130막대 −0.88pp에서 220막대 −5.99pp로 커진다. 따라서
"속도는 난이도 축이 아니다"(v1 결론)는 저밀도에서만 참이다.

**검수 포인트**
1. **통계적 유의성.** 셀당 ~2050 ep이면 arm 간 SE ≈ 1.3pp다. −0.88pp는 유의하지 않고
   −5.99pp만 유의하다. "단조 증가"라고 말할 근거가 되는가, 아니면 **양 끝점 차이**로만
   말해야 하는가? 나는 표로 전 구간을 제시했는데 과대주장일 수 있다.
2. 220막대는 **OOD**(학습 최대 205)다. 상호작용이 OOD 셀 하나에 의존하면 주장이 약하다.
   205막대(−3.06pp)만으로도 성립하는가?
3. 속도 축이 **단조가 아니다**(0.7 m/s가 5개 밀도 중 4개에서 0.3보다 높음). 이걸 기록만
   해두고 해석하지 않았다 — 설명이 필요한 이상 신호인가?
4. v1(−78pp)과 v2(−11.4pp)를 "단위 밀도당 기울기 2.98 vs 2.02"로 비교했다. 밀도 범위가
   6배 다른 두 격자를 선형 기울기로 비교하는 게 타당한가? (밀도-성능이 절벽형이라면
   기울기 비교 자체가 오도일 수 있다 — v1 07-16 항목이 "절벽형"이라고 적고 있다.)

---

## 2. 2순위 — 결론은 안 바뀌지만 방법이 의심스러운 곳

### (D) dropout 채널 분해 (H2/H3/게이트/H4)

5회 A/B로 LiDAR target association 경로를 분해했다. 요약:

| 개입 | dropout Δ | clean Δ | 판정 |
|---|---:|---:|---|
| P2 mask backfill | +2.15pp (2시드) | **−1.91pp (2시드)** | 상쇄, 기각 |
| H2 association off | **+3.26pp (2시드)** | −0.61pp | 견고하나 미채택 |
| H3 공분산 정직화 | +1.46pp | −0.54pp | 유의 미달 |
| 게이트 0.65/0.35 | +1.51/+1.71pp | — | 기각 |
| H4 플래그만 차단 | +2.36pp | +0.58pp | H2의 69% |

**검수 포인트**
1. **모두 단일 실험 사전등록 게이트(≥4pp)에 미달**인데, 나는 "채널이 실재한다"고 서술했다.
   유의성 없이 메커니즘을 주장하는 것이 정당한가? 근거로 든 것은 (a) 2시드 재현,
   (b) capture와 bar_contact 두 독립 지표가 같은 비율(20.0% / 20.7%)을 가리킴,
   (c) H4+H2 분해가 합으로 맞음 — 이 셋이 충분한가?
2. H4 해석("정책이 읽는 건 공분산이 아니라 visible/time-since-seen 이산 신호")은
   **간접 증거뿐**이다. 관측 벡터에서 해당 성분을 직접 계측하지 않았다.
3. `_associate_lidar_target`의 bearing·z가 **트래커 예측의 되먹임**이라는 지적
   (`navrl_perception.py` 해당 함수) — 이건 코드 사실이므로 확인 요망. 사실이면 이건
   latency와 무관한 **상시 결함**이고, dropout이 아닌 clean에서도 작동한다.

### (E) 대시보드 v1/v2 라벨링

**주장**: `perc100()`이 전역 면적(v2 1600 m²)으로 v1 곡선을 나눠 밀도가 3.3× 과소 표기됐다.
Speed 탭은 키랄리티 수정(07-29) **이전** 데이터를 표시 중이었다.

**검수 포인트**
1. "키랄리티 이전 9개 곡선"의 분류가 맞는가? (`_PRE_CHIRALITY_CURVES` 목록,
   `tools/update_status_snapshot.py`) — 체크포인트 날짜 기준으로 판정했는데 누락/오분류 가능.
2. `corrected_sensorfix_legacy_speed_axis`는 07-29 **이후**라 superseded로 안 찍었다.
   그런데 legacy Gaussian 500 epoch pilot이라 대표성이 없다 — 표시 후보로 남겨둔 게 맞는가?
3. 무효 데이터를 **삭제하지 않고 라벨 보존**하기로 했다. 발표/논문 관점에서 이 방침이 맞는가?

---

## 3. 검수하지 않아도 되는 것 (이미 자체 검증됨)

- 테스트: python 21파일 + JS 1 전량 PASS, pyflakes undefined name 0, 충돌 마커 0
- 평가 무결성: 모든 셀이 SHA-피닝 + receipt. **실행 중 evaluator를 편집했을 때 가드가 실제로
  결과를 거부**한 사례가 있다(2026-08-06 WORKLOG) — 가드 작동 실증
- detector SHA 가드가 죽어 있던 결함 발견·수정 + 회귀 테스트 3건
- 대시보드 정합성: status.json ↔ HTML fallback 동기화, getElementById 대상 누락 0

---

## 4. 검수 결과로 받고 싶은 것

1. §1 (A)(B)(C) 각각에 대해 **동의 / 조건부 동의 / 반박** 중 하나와 근거
2. 특히 (C)의 통계적 주장 강도 — 논문 문장을 어디까지 밀 수 있는가
3. 놓친 confound가 있으면 지적 (나는 이번에 하나를 놓쳤다가 자체 발견했다)
4. 다음 우선순위 판단: learned detector(−13.9pp)로 넘어가는 게 맞는가, 아니면
   dropout 미설명 70%를 더 파야 하는가

---

## 부록 — 재현 명령

```bash
# 전부 inference-only. 각 스크립트는 PREFLIGHT=1로 계약만 검증 가능.
cd aerial_gym/rl_training/rl_games
PREFLIGHT=1 ./eval_navrl_v2_latency_ego_motion.sh      # P3
PREFLIGHT=1 ./eval_navrl_v2_lidar_silent_correct.sh    # H4
PREFLIGHT=1 ./eval_navrl_v2_density_speed_map.sh       # v2 맵
PYTHONNOUSERSITE=1 python ../../../tests/test_navrl_latency_compensate.py   # 33 tests
```
