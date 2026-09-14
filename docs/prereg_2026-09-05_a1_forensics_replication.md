# 사전등록 — A1: 접촉 기하 포렌식 재현 + A2 누락 지표

작성 2026-09-05. 계획: `docs/plans/diagnostic_paper_plan_2026-09-05.md` §A1·A2.
선행: `docs/prereg_2026-09-04_contact_corridor_forensics.md` — **범주·우선순위·lookback은 그대로
승계하며 재정의하지 않는다.**

## 0. 질문

seed 491 단일 시드에서 얻은 "충돌의 77~78 %가 회랑 밖"이 **다른 시드에서 재현되는가?**
재현되지 않으면 그 표를 논문에 내지 않는다.

## 1. 조건

선행 사전등록과 **동일**하되 seed만 바꾼다.

| 항목 | 값 |
|---|---|
| seed | **497** (미사용 확인) |
| arm | `off` · `riskcap` |
| distractor | 0 · bars 70 · 2,049 ep/arm |
| 체크포인트 | ref5in ep1900, sha `197ea269…` |
| lookback | 10 스텝 = 1.0 s (승계) |
| 분류 범주·우선순위 | 승계, 변경 없음 |

## 2. 재현 판정 (결과 이전 동결)

seed 491의 95 % CI를 그대로 게이트로 쓴다. **주지표는 `lateral + no_return` 합계.**

| arm | 지표 | seed 491 | 재현 허용 구간 |
|---|---|---:|---|
| off | **lateral + no_return** | 77.2 % | **[72.5 %, 81.9 %]** |
| off | lateral | 57.1 % | [51.5 %, 62.7 %] |
| off | no_return | 20.1 % | [15.6 %, 24.6 %] |
| off | in_corridor | 21.1 % | [16.5 %, 25.7 %] |
| riskcap | **lateral + no_return** | 78.1 % | **[72.8 %, 83.4 %]** |
| riskcap | lateral | 58.4 % | [52.0 %, 64.7 %] |
| riskcap | no_return | 19.7 % | [14.6 %, 24.9 %] |
| riskcap | in_corridor | 18.5 % | [13.5 %, 23.4 %] |

- `REPLICATED` — 두 arm의 **주지표가 모두** 허용 구간 안
- `PARTIAL` — 한 arm만 통과
- `FAILED` — 둘 다 벗어남

**`FAILED` 또는 `PARTIAL`이면 시드를 5개로 늘린다.** 그래도 흔들리면 E2를 주장하지 않는다.
게이트를 사후에 넓히지 않는다.

## 3. A2 — 추가 계측 (전부 평가 전용, 기존 경로 무변경)

논문이 요구하는데 없던 것을 넣는다. 결과 JSON 감사로 부재를 기계 확인했다.

| 지표 | 정의 | 왜 필요한가 |
|---|---|---|
| `corridor_clearance_hist` | 매 스텝 회랑 clearance의 히스토그램(0–12 m, 0.5 m 빈) | "안전해져서 느려진 것"을 잡는다 |
| `path_length_m` | 에피소드별 누적 이동거리, 결과별 평균 | 우회 비용 |
| `governor_compute_us` | 거버너 1회 호출의 평균·최대 µs | 문헌 관행(HOCBF 90.6/384.5 µs)과 비교 |
| `cmd_vs_actual_deg_all` | **비접촉 프레임 포함** 전 프레임 평균 | 선행 사전등록의 예측 6을 평가 가능하게 |

이들은 **판정 게이트가 아니다.** 서술 지표이며 A1 판정에 관여하지 않는다.

- **예측 6 (재개)**: 접촉 프레임의 `cmd_vs_actual_deg`가 전체 평균보다 크다.

## 4. 기계 무결성

- **M1**: `NAVRL_CONTACT_GEOMETRY` 미설정 시 신규 계측 경로가 실행되지 않는다.
- **M2**: 다섯 범주 합계 = 이력 충분 접촉 수 (승계, 러너가 fail-closed로 강제).
- **M3**: 두 arm이 동일 checkpoint sha·동일 source bundle을 기록한다.

## 5. 산출물

`results/navrl_contact_geometry_seed497/` + `summary.{md,json}`, 판정 키 `verdict_replication`.
WORKLOG 항목.

## 6. 범위 밖

재학습 없음 · 거버너 법칙 변경 없음 · 범주 재정의 없음 · SCP(A3)는 별도 사전등록
