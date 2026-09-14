# 사전등록 — A3: Star-Convex 자유공간의 shadow-mode 처방 반사실

작성 2026-09-05. **초안 — 사용자 승인 전 실행 금지.**
계획: `docs/plans/diagnostic_paper_plan_2026-09-05.md` §A3.
선행: `docs/prereg_2026-09-04_contact_corridor_forensics.md`(범주·lookback 승계),
`docs/prereg_2026-09-05_a1_forensics_replication.md`.

## 0. 질문 (RQ)

접촉 포렌식은 막대 충돌의 **77~78 %가 거버너 회랑 밖**임을 보였다(측면 57~58 % + 무반환 20 %).
그러나 그것은 **서술**이다. 처방의 반사실을 묻는다:

> **전방향 star-convex 자유공간이었다면, 그 접촉들이 사전에 감지되었을까?**

## 1. 설계 — shadow mode (정책 무변경)

S1 shadow와 동일한 구조다. 표준 접촉 포렌식 평가를 그대로 실행하고, 정책은 기존 경로를 그대로
소비한다(**궤적 불변**). 매 스텝 같은 스캔에 대해 병렬로:

1. 72×4 반환을 `scan_to_points`로 3-D 점군으로. **무반환 광선은 버리지 않고 최대사거리 구면에
   놓는다** — 이것이 "미지=자유"를 "미지=경계"로 바꾸는 지점이다.
2. 접촉 시각 기준 **1.0 s 전**(승계) 프레임에서, 부딪힌 막대의 방위에 대해
   `direction_clearance`로 star-convex 경계를 구한다.
3. 그 막대가 **경계 안쪽이었는가**(= 사전 감지)를 기록만 한다.

actor·critic·보상·종료에 어떤 쓰기도 하지 않는다.

## 2. 고정 파라미터 — **본 문서로 동결, 사후 튜닝 금지**

| 항목 | 값 | 근거 |
|---|---|---|
| 방위 cone 반각 | **5.0°** | LiDAR 수평 빔 간격과 동일. 더 좁히면 빔 사이를 보게 되고 넓히면 `omni`로 수렴한다 |
| 최대사거리 | 12.0 m | LiDAR 사거리 |
| 수직 FOV | +20° / −10° | 센서 설정 |
| lookback | 10 스텝 = 1.0 s | 선행 사전등록 승계 |
| 무반환 처리 | **구면에 배치** | 이 실험의 요점 |
| 입력 스캔 | **수직 접힌 `last_scan_nearest`** | 거버너가 소비하는 바로 그 입력(`directional_lidar_clearance`도 수직 amin). 전체 [B,V,H]를 넣으면 두 가지가 동시에 바뀌어 비교가 격리되지 않는다. 이 입력에서 회랑과의 차이는 **오직 ① 모든 방위 참여 ② 무반환이 경계** 둘뿐이다 |

## 3. 조건

접촉 포렌식과 동일. seed **503**(미사용 확인), arm `off`·`riskcap`, N=0, 70 bars, 2,049 ep/arm,
체크포인트 ref5in ep1900.

## 4. 판정 규칙 (결과 이전 동결)

주지표: **재분류율** = (명령 회랑 기준 `LATERAL` 또는 `NO_RETURN`이었던 접촉 중,
star-convex 경계 안쪽이었던 비율).

- `PRESCRIPTION_SUPPORTED` — 재분류율 **≥ 50 %**
- `PARTIAL` — 20 % ~ 50 %
- `INSUFFICIENT` — **< 20 %**

`INSUFFICIENT`이면 논문의 처방 절을 **삭제하고 순수 진단으로 축소한다.** 게이트를 사후에
낮추지 않는다.

부지표(게이트 아님): `NO_RETURN`만의 재분류율, `LATERAL`만의 재분류율, star-convex clearance의
분포, 연산 시간.

## 5. 예측

1. `NO_RETURN`의 재분류율이 `LATERAL`보다 **높다** — 구면 배치가 직접 겨누는 항목이다.
2. 전체 재분류율이 50 %를 넘는다.
3. star-convex clearance가 회랑 clearance보다 **작다**(더 보수적) — 모든 방위를 보므로.
   그렇지 않으면 구현 오류를 의심한다.

## 6. 기계 무결성 (fail-closed)

- **M1**: shadow on/off에서 `NAVRL_OBS_DUMP` 관측 스트림이 **비트 동일**해야 한다(S1과 동일 방식,
  자기참조 필드 제외). 불일치 시 본 측정 금지.
- **M2**: 재분류 대상 접촉 수 ≤ (`LATERAL` + `NO_RETURN`) 접촉 수.
- **M3**: 두 arm이 동일 checkpoint sha·source bundle을 기록.
- 예외는 셀 VOID — 조용한 skip 금지.

## 7. 명시적 범위 밖

재학습 없음 · 정책 경로 변경 없음 · **활성 A/B 없음**(필터를 실제로 켜지 않는다) ·
cone 반각 스윕 없음(대안 비교는 A4에서 `omni`로) · 볼록껍질 전체 구성 없음
(star-convex 성질상 방위별 최근접으로 충분하며, 이 등가성은 단위테스트
`test_widening_the_cone_degenerates_to_omnidirectional`가 고정한다)
