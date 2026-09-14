# 사전등록 — distractor envelope와 False Target Lock Rate (eval-only)

작성 2026-09-01, **distractor 환경에서 어떤 측정도 하기 전**. 결과를 본 뒤 임계·seed·distractor 수·
판정 규칙을 바꾸지 않는다.

## 1. 질문

현재 기본 검출기 `AppearanceTargetSegmenter`(`navrl_perception.py:591`)는 파라미터 5개짜리 1×1
conv로 `3.0·R − 2.0·G − 2.0·B − 0.9`를 계산한다. 문자 그대로 **"빨간가?"** 다. 렌더러가 표적을
평평한 `[0.88, 0.08, 0.045]`로 칠하고 다른 어떤 물체도 빨갛지 않으므로, 그 세계에서
**"빨간 픽셀 = 표적"은 정의상 참**이다.

v7 offline gate는 8개 기준을 전부 통과했다(`results/navrl_detector_offline_gate_v7_confirmatory/`):
frame precision **0.99766**, far 14–20 m recall 0.99688, bearing MAE 0.0244°, 8/8 PASS.
**그러나 그 게이트의 어떤 조건에도 distractor가 없다** — 요약 JSON에 관련 키가 0개다. 즉 이 숫자는
"표적과 닮은 것이 하나도 없는 세계에서 표적을 잘 찾았다"이지 **"표적과 닮은 것과 표적을 구별한다"가
아니다.**

`docs/archive/development_directions_2026-08.md:107`이 이 축을 "D9 (보너스, eval-only) envelope에
distractor 축 추가"로 이미 제안했으나 구현되지 않았다.

**묻는 것: 표적과 같은 색·같은 크기의 물체가 장면에 있을 때 현재 검출기는 무엇을 표적이라고
말하는가.** 고치는 것은 이 사전등록의 범위가 아니다.

## 2. 계산이 예측하는 것 (측정 전 기록)

`_detect_rgbd`는 임계를 넘은 **이미지 전체** 양성 픽셀을 하나의 무게중심으로 축약한다
(`navrl_perception.py:1428`, `:1480`). 연결 성분 분석도, blob 분리도, 최대 성분 선택도 없다:

```
count = mask.sum(dim=(1,2));  u = (mask*u_grid).sum()/count;  range = (depth*mask).sum()/count
```

따라서 표적과 distractor가 **동시에 보이는 프레임에서는 구조적으로 반드시 실패**한다 — 무게중심은
둘 사이 허공을 가리키고 range는 두 거리의 평균이며 count는 합이므로 confidence가 오히려 **올라간다**.

**예측: 동시 가시 프레임의 오귀속률(DISTRACTOR+GHOST)이 90% 이상.** 예측이 빗나가면(예: depth 클립이
distractor를 우연히 배제) 그 자체가 발견이며, 그 경우에도 임계를 사후 조정하지 않는다.

## 3. 실험 계약

| 항목 | 값 |
|---|---|
| 성격 | **평가 전용.** 검출기 수정·재학습·PPO 권한 없음 |
| 검출기 | 현행 기본값(`AppearanceTargetSegmenter`)과 학습형 v7 **둘 다** |
| distractor 형상 | 구(r=0.15, 표적과 **동일 크기**) · 박스 · 기둥 |
| distractor 색 | 표적과 **동일한** nominal `[0.88, 0.08, 0.045]` — 최악 조건부터 |
| **distractor 수 (조작 축)** | **0 / 1 / 3 / 5** |
| 배치 | 막대와 동일한 `footprint_clearance`(비중첩·표면여유), 정적 |
| 아레나·밀도 | 40×40×3, 70막대 고정 |
| **평가 seed** | **479** (전수검색 0건) |
| 에피소드 | 셀당 2,049 |
| 정책 | frozen ref5in D1 ep1900 (`197ea269…`), deterministic, governor off |
| 검출 해상도 | **카메라 해상도 경로**(decoupled 금지, §5) |

`N=0` 셀은 **회귀 검사**다: 현행 계보와 수치가 일치해야 하며, 아니면 distractor 도입이 무관한
것을 바꾼 것이므로 전체를 VOID한다.

## 4. 1차 지표 — False Target Lock Rate

GT 표적·distractor 위치는 **평가 지표 계산에만** 쓴다(actor 관측 유입 금지, `CLAUDE.md` 관측 계약).
검출기가 낸 3D 측정 위치를 프레임마다 3범주로 분류한다:

| 범주 | 정의 |
|---|---|
| `TARGET_LOCK` | 측정 위치가 실제 표적 중심 반경 **0.5 m** 이내 |
| `DISTRACTOR_LOCK` | 어떤 distractor 중심 반경 0.5 m 이내 |
| `GHOST_LOCK` | 둘 다 아님 (무게중심 평균이 만든 허공) |

$$\mathrm{FTLR} = \frac{\#\mathrm{DISTRACTOR\_LOCK} + \#\mathrm{GHOST\_LOCK}}{\#\{\text{visible=true인 프레임}\}}$$

반경 0.5 m 근거: 표적 반경 0.15 m + 검출기 range MAE 0.178 m(v7 실측)의 약 2배 여유. 결과를 보기
전에 고정한다.

**보조 보고(판정 아님)**: 범주별 원값, `count`·`confidence`가 N에 따라 어떻게 변하는지,
never-acquired, capture/crash/timeout 원값.

## 5. 게이트

**Gate 0 (구현 타당성, 판정보다 먼저)** — 셋 다 통과해야 한다:
1. `NAVRL_DISTRACTOR_COUNT` 미설정 시 현행과 **bit-identical**
2. distractor > 0 과 decoupled 검출 해상도의 조합에서 **fail-closed 거부**
   (근거: 고해상도 detect 경로의 항등성은 표적만 칠해진 세계에서만 성립한다 —
   `navrl_detector.py:498,538`. distractor를 같은 색으로 칠하면 perception 분할기는 발화하지만
   detect 마스크에는 없어 항등이 깨지고, 그 경로의 모든 측정이 조용히 무의미해진다)
3. `N=0` 셀이 현행 계보 수치와 일치

하나라도 실패하면 `FAIL_CLOSED_IMPLEMENTATION`이며 검출기에 대한 어떤 주장도 하지 않는다.

**Gate F (1차)** — `N=5`에서:

| 판정 | 조건 |
|---|---|
| `COLOR_SHORTCUT_CONFIRMED` | FTLR **≥ 50.00%** |
| `DETECTOR_ROBUST_TO_DISTRACTORS` | FTLR **≤ 5.00%** |
| `INCONCLUSIVE_DISTRACTOR_ENVELOPE` | 그 외 |

임계 근거: 50%는 "절반 이상의 프레임에서 표적이 아닌 것을 표적이라 말함" — 우연 수준을 훨씬 넘는
명백한 실패다. 5%는 v7이 통과한 frame precision 0.98 기준의 여집합과 같은 자릿수다. 두 값 모두
결과를 보기 전에 고정한다.

**판정 방향 주의**: `COLOR_SHORTCUT_CONFIRMED`는 **예상된 결과이고 실험 실패가 아니다.** 이 실험의
값어치는 검출기 개선이 아니라 **결함의 정량화**에 있다. 반대로 `DETECTOR_ROBUST`가 나오면 그것이
놀라운 결과이며, 그 경우 구현이 의도대로 동작했는지(distractor가 실제로 렌더·검출되는지) 먼저
재확인한다.

## 6. 권한과 한계

- **P2 STRICT FAIL / D1 FAIL / P3 BLOCKED를 변경하지 않는다.** PPO·재학습·Track A/B GPU 권한 없음.
- v7 offline gate의 8/8 PASS 판정을 **소급 변경하지 않는다.** 그 게이트는 distractor 없는 조건에서
  유효하며, 본 실험은 그 조건 밖을 잰다. 두 결과는 모순이 아니라 **서로 다른 질문**이다.
- (L1) distractor가 **정적**이다. 실기의 움직이는 오탐(새 등)은 범위 밖이다.
- (L2) 색이 표적과 **동일**하다. 색 거리에 따른 성능 곡선은 범위 밖이다.
- (L3) 카메라 해상도 160×90에서 잰다. 고해상도에서의 FTLR은 다를 수 있다.
- (L4) 단일 정책·단일 seed·70막대 1조건.
- **(L5) distractor가 일부 코드 경로에서 자유 공간으로 남는다.** 자산 배열이
  `[target?][distractors...][bars...]`이고 `_bar_offset`이 distractor를 건너뛰도록 넓혀졌으므로
  `[_bar_offset : _bar_offset + n_bars_active]`를 읽는 지점들은 distractor를 보지 못한다.
  FTLR을 직접 오염시키는 **두 곳만 고친다** — 드론 스폰 clearance(`navrl_task.py:~1912`, 안 고치면
  드론이 distractor 안에서 스폰되어 그 프레임의 검출이 물리적으로 무의미해진다)와 표적 경로
  planner(`~5858`, 안 고치면 경로가 distractor를 관통해 가림 통계가 틀어진다).

  **고치지 않는 다섯 곳**: 정적 goal 배치(`~4102`), recovery clearance(`~6058`), bar-contact
  probe(`~3081`, `~3127`). 근거는 이들이 "검출기가 무엇을 표적이라 말하는가"라는 1차 지표에
  직접 기여하지 않기 때문이며, 사전등록 §1이 검출기 수정을 범위 밖으로 두었기 때문이다. 결과적으로
  distractor 충돌은 미귀속 contact로 기록되고 정적 goal이 distractor 안에 놓일 수 있다.
  **capture/crash/timeout 원값을 해석할 때 이 사실을 반드시 함께 읽어야 한다** — 그 값들은 이미
  §4에서 판정 대상이 아니라 보조 보고로만 쓰기로 했다.
- 결과가 어떻든 **검출기 교체를 승인하지 않는다.** 교체는 별도 사전등록이 필요하다.

## 7. 하지 않을 것

결과를 본 뒤 임계·seed·distractor 수 변경 · distractor > 0에서 decoupled 강행 ·
`aerial_gym/config/robot_config/**`·`resources/robots/**` 편집(provenance freeze) ·
dirty runtime 실행 · 실행 중인 다른 worktree의 학습 방해.

## 8. 기록 요건

`results/navrl_detector_distractor_envelope_seed479/`. 셀당 조건·FTLR·3범주 원값·검출기 종류,
Gate 0 증거, 무효·VOID 사유. 요약에 `p2_verdict_changed: false`, `d1_verdict_changed: false`,
`p3_unlocked: false`, `decision_authority: "none"`.

---

## 3-b. 검출기 축의 구현 형태 (2026-09-02, **어떤 셀도 실행하기 전** 기록)

§3은 "검출기: 현행 기본값과 학습형 v7 **둘 다**"라고 적었으나 구현 형태를 명시하지 않았다. 구현
과정에서 이것이 **두 번째 인자(factor)**임이 드러났고, 측정 전에 형태를 확정한다.

**2×4 요인설계**: 검출기 ∈ {default, v7} × distractor 수 ∈ {0, 1, 3, 5} = **8셀**.

이는 `VERIFICATION.md` fail-closed 3(한 run에서 여러 축 동시 변경 금지)을 위반하지 않는다. 그 규칙은
**하나의 학습 run** 안에서 airframe·reward·horizon 등을 함께 바꾸지 말라는 것이고, 여기서는 각 셀이
독립된 평가 실행이며 이웃 셀과 정확히 한 인자만 다르다.

**단일축 단언을 방향별로 나눈다**: 검출기를 고정하면 4셀 사이의 유일한 env 차이가
`NAVRL_DISTRACTOR_COUNT`여야 하고, distractor 수를 고정하면 2셀 사이의 유일한 차이가 검출기 선택
변수여야 한다. **양방향 모두 단언한다** — 한쪽으로만 검사하면 다른 축에 딸려온 변수가 숨는다.

**§5 Gate F 판정은 검출기별로 각각** 자기 N=5 셀에서 내린다. 따라서 판정이 **두 개** 나오며,
하나를 다른 하나로 읽을 수 없게 분리해 기록한다. default의 결과는 v7에 대해 아무것도 말하지 않는다.

**Gate 0.3(N=0 회귀, ±3.75 pp)은 default 검출기의 N=0 셀에만 적용한다** — 계보 기준선
(sensor-fidelity `baseline` arm, seed 421, capture/crash/timeout 70.52/19.81/9.66%)이 그 조건에서만
존재하기 때문이다. v7의 N=0 셀은 대응 기준선이 없으므로 **서술적(descriptive)**이며 게이트가 아니다.

### 왜 default만으로는 부족한가 (근거 기록)

default `AppearanceTargetSegmenter`는 파라미터 5개짜리 색 규칙이므로 색 판별에서 실패하는 것이
거의 동어반복이다. **열려 있는 질문은 학습형 v7이 그 지름길을 벗어났는가**이며, 0.99766이라는
frame precision을 문서에 남긴 것도 v7이다. v7을 빼면 실제 연구 질문이 측정되지 않은 채 남는다.

§5의 임계(0.50 / 0.05)와 §4의 지표 정의는 **변경하지 않는다.**

## 3-c. 검출기 간 FTLR은 서로 비교할 수 없다 (2026-09-02, **실행 전** 기록)

§3-b가 검출기를 두 번째 인자로 만들었으나, 두 수준이 **동일한 프레임 분포를 보지 않는다**는 사실이
따라온다. 측정 전에 못박는다.

동결 정책은 `cfg_detector_threshold = 0.55`, `cfg_detector_checkpoint_sha256 = ''`로 학습됐다 —
**학습형 검출기를 한 번도 본 적이 없다.** v7 셀에서는 그 정책이 함께 학습한 적 없는 검출기
(sha `85c7974b…`, threshold 0.700)의 출력을 받아 비행한다. 따라서 v7 셀의 **궤적이 default 셀과
정당하게 다르다.**

궤적이 다르면 관측 기하(거리·방위·가림·distractor 동시 가시성)의 분포가 달라지고, FTLR은 그 분포
위에서 정의된 값이다. 그러므로:

- **검출기별 FTLR은 각자 유효하다** — 각 값은 그 검출기가 실제로 본 프레임에서 계산된다.
- **`FTLR_v7 − FTLR_default`는 의미 있는 양이 아니다.** 두 검출기의 강건성 차이와 두 궤적 분포의
  차이가 분리되지 않는다. 이 차이값을 계산하거나 보고하지 않는다.
- 같은 이유로 v7 셀의 capture/crash/timeout은 **계보와도, default 셀과도** 비교하지 않는다.

§5 Gate F가 애초에 검출기별 판정인 것이 이 제약과 정합한다. 두 판정은 각자의 질문에 답하며
서로에 대해 아무것도 말하지 않는다.

**이것을 해소하려면** 각 검출기로 각각 학습한 정책이 필요하고, 그것은 PPO 권한을 요구하므로
이 사전등록의 범위 밖이다. 여기서는 한계로 기록하고 넘어간다.
