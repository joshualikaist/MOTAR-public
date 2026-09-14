# 검증 요청 — NavRL 속도 안전필터, 2026-09-05 ~ 09-07 측정 전체 (136 cell)

09-06 패키지(83 cell)에 R-A 드리프트 탐침 2, R-B 시드 반복 50, 교차 트리 확인 1을 더한 것입니다.
09-06 검증에서 받은 정정 세 가지(상호작용 검정, Holm 보정, 독립성 단서)는 반영됐습니다.

## 무엇을 검증해 달라는 건가

우리 결론을 확인해 달라는 것이 아니라 **자료가 그 결론을 지지하는지, 더 단순한 설명이 남아 있는지**를
봐 주세요. 질문은 다섯 개이고 전부 좁고 반증 가능합니다.

1. `governor_cells.csv`의 **개수만으로** 3 시드 결합 arc − riskcap을 다시 계산하면
   **−1.49 pp [−1.90, −1.08]**이 나오는가? `verify.py`가 표준 라이브러리만으로 그 계산을 하는데,
   그 스크립트 없이 손으로 해도 같은가?
2. 사전등록 예측 P5("stopcap − riskcap CI가 모든 밀도에서 0 포함")를 **기각**한 근거 — 70 bars 결합
   −1.19 [−2.01, −0.37], 세 시드 부호 일치 — 가 자료와 맞는가? "노이즈"라는 대안 설명이 남는가?
3. 세 시드가 **두 소스 트리**에서 나왔다(seed 523: `b51c3d2f`, 527·531: `3aa30cd8`). 같은 seed-523 cell을
   새 트리에서 재실행해 비트 동일(306/1699/44)을 얻었다. 이 1 cell이 결합의 정당성 근거로 충분한가?
4. [`docs/plans/confirmation_phase_plan_2026-09-06.md`](../../docs/plans/confirmation_phase_plan_2026-09-06.md)
   §2 표(C1~C6, N1, N2) 중 **근거가 주장을 못 받치는 행**이 있는가?
5. [`docs/plans/master_plan_2026-09-07.md`](../../docs/plans/master_plan_2026-09-07.md)의 기간 추정 중
   근거 없는 숫자가 있는가?

## 자료 (이 디렉터리)

| 파일 | 내용 |
|---|---|
| `governor_cells.csv` | 평가 cell **136개**, 58열. 한 행 = 한 cell. **개수**(`n_crash`/`n_captured`/`n_timeout`, `actual_episodes`)가 있으니 비율·CI를 직접 재계산해 주세요 |
| `roots_manifest.json` | 루트 13개의 커밋·계약·소스 지문 |
| `verify.py` | 개수→결합 재계산. 표준 라이브러리만, `tools/` 임포트 없음 |
| `export_verification.py` | CSV를 만든 스크립트(재현용) |

접촉별 원자료는 각 행의 `contact_records_path`가 가리키는 저장소 내 JSONL입니다(접촉 1건 = 1행,
`hit_lateral_cmd`가 접촉 1 s 전 명령축 기준 측면 오프셋, `pinch_t0`가 접촉 순간 양측 협착).

## 미리 밝히는 한계

- 학습 시드는 **1개**(197). 재현된 것은 *평가* 시드 3개다. 학습 시드 반복(R-C)은 미실행.
- 개별 밀도에서 arc − riskcap CI가 0을 제외하는 것은 5 중 **4**(130 bars: −0.86 [−1.73, +0.02]).
- 결과가 인용하는 체크포인트 4개가 유실돼 seed-911 held-out 결과 6건은 재실행 불가(값은 보존).
- 교차 커밋 동일 조건 재현은 13쌍 중 12쌍 비트 동일, 예외 1건(A8의 T1/riskcap, 원인 미상).
- 전부 시뮬레이션.

## 열 설명 요약

`cond_*` = 평가 조건(전부 기록값). `cond_speed_governor_mode` ∈ {off, riskcap, stopcap, dwa_arc, riskcap_arc,
omni, …}, `cond_speed_governor_half_width_m` = 회랑/튜브 반폭. `checkpoint_sha256` 앞 12자리
`f70221393660`이 ep25000 계보, `197ea269…`가 ref5in ep1900. `runtime_git_commit`이 평가 당시 커밋.
