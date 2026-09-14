# Claude 전달용 — 검증 5B까지 반영한 MOTAR PPT 보완·검수 요청

기준일: 2026-08-13 KST
용도: 현재 PPT를 Claude에게 넘겨 **수치·주장·구조를 검수하고, 검증 5B 결과까지 반영한 최종 발표본**을
만들게 할 때 사용하는 요청서다.

> 중요: corrected-semantics 검증 5A와 ref5in P0는 완료됐다. ref5in P1a/P1b는 strict FAIL,
> P1c는 PASS했다. held-out P2는 capture/crash를 통과했지만 timeout 5.56%로 **strict FAIL**했고,
> full-budget P3/검증 5B는 시작하지 않았다. 아래 `[5B 결과 입력]` 자리는 실제 full-budget
> summary/receipt가 생긴 뒤에만 채운다.

---

## 1. Claude에게 그대로 보낼 프롬프트

아래 파일들과 현재 PPT/PPTX를 함께 읽고, MOTAR 연구 발표자료를 전면 검수·보완해 주세요.

핵심 목표는 발표를 화려하게 만드는 것이 아니라, **검증 1–5B의 증거 수준에 맞춰 주장 강도를
정확히 조절하면서 연구 기여가 선명하게 보이도록 재구성하는 것**입니다. 기존 슬라이드를 무조건
유지하지 말고, 중복·구식 수치·오해를 부르는 그림은 과감히 삭제하거나 부록으로 이동해 주세요.

### 반드시 읽을 파일

1. `docs/CLAUDE_PPT_REVIEW_REQUEST_VERIFICATION5B_2026-08-13.md` — 이번 작업의 최상위 계약
2. `docs/codex_review_2026-08-12.md` — 검증 1–4 및 5A 독립 검수 결론
3. `results/navrl_v2_v5a_semantics_smoke_seed197/summary.md` — 검증 5A 확정 결과와 한계
4. `docs/GENSPARK_PPT_BRIEF_2026-08-11.md` — 기존 발표의 시스템·Gate 1–3 수치 원천
5. `docs/development_directions_2026-08.md` — 후속 연구 방향
6. `docs/reference_platform_proposal_2026-08.md`와
   `results/navrl_ref_platform_verification/summary.md` — 5B 기체 계약 후보와 검증 한계
7. `results/navrl_ref5in_smoke_seed197/summary.md`, `p1b/summary.md`, `p1c/summary.md` — P1 계보
8. `results/navrl_ref5in_p2_seed313/summary.md`와 `attestation.json` — P2 strict FAIL과 proof
9. `WORKLOG.md`의 2026-08-12 이후 항목 — 수정 이력과 반증 기록
10. 검증 5B 완료 후 생성될 `summary.md`, `summary.json`, source receipt, seed별 held-out 결과

파일 사이 내용이 충돌하면 날짜가 최신인 독립 검수 문서를 우선하고, 어느 쪽이 맞는지 판단할 수
없으면 슬라이드에 넣지 말고 `검수 필요`로 보고해 주세요.

### PPT 제작 계약

- 한국어, 16:9, 본문 15–17장 + 부록 4–6장, 12–15분 발표.
- 청중은 로보틱스·강화학습 연구자와 대학원 심사자.
- 슬라이드 제목은 주제명이 아니라 **결론형 한 문장**으로 작성.
- 밝은 배경, navy/cyan 기본색. 위험·충돌만 orange/red, 미실행·미확정은 gray.
- 표를 화면에 그대로 붙이지 말고 차트로 다시 그리되, 축·단위·seed·episode 수·95% CI와
  `legacy 601`/`schema-v2 exact-600`, ID/OOD를 표시.
- 모든 차이는 `%`가 아니라 percentage point인 `pp`로 표기.
- 슬라이드마다 40–70초 분량의 한국어 발표자 노트 작성.
- 실험 결과마다 화면 하단에 작은 글씨로 `checkpoint / training seed / eval seed / episodes /
  semantics / detector / governor` 계약을 표시.
- 모든 확정 숫자 옆에 출처 파일명을 발표자 노트에 남길 것.
- 검증 실패나 정정도 숨기지 말고, **감사로 연구 질문이 더 정확해졌다는 흐름**으로 표현.
- 외부 논문·수치·실기 성능을 임의 검색·추정하지 말 것. 참고문헌은 확인된 서지정보만 사용.

### 가장 중요한 편집 원칙

1. **5A와 5B를 합치지 말 것.** 5A는 단일 seed, 1,000 epoch, 70 bars의 engineering smoke다.
   5B는 clean source receipt를 가진 full-budget corrected-semantics baseline이어야 한다.
2. **5A capture 78–79%를 최종 성능으로 쓰지 말 것.** 이는 on-policy training 최근창이다.
3. **5B 미실행 상태에서는 결과 슬라이드를 회색 설계 슬라이드로만 둘 것.** 결과가 생성된 후에도
   사전등록 endpoint와 seed를 바꾸지 말고 실제 CI를 그대로 넣을 것.
4. legacy 601/no-bootstrap 정책의 Gate 1–3 성과와 corrected-semantics fresh lineage를 같은 곡선에서
   연속 학습처럼 연결하지 말 것. 서로 다른 계보로 나란히 표시할 수는 있다.
5. learned-v2 비열등성과 learned-v7 nominal 손실은 모순이 아니다. **다른 detector artifact와 다른
   실험 질문**이다. 각각 라벨을 명확히 표시할 것.
6. contact-time token miss와 negative stopping margin은 상관된 기술 현상이지 독립적 인과 원인으로
   확정되지 않았다. `원인 규명`이 아니라 `후보 채널 분리`로 표현할 것.
7. **5B의 robot contract를 결과보다 먼저 확인할 것.** legacy `navrl_quad`와 신규
   `navrl_ref5in_quad`는 서로 다른 학습 계보다. ref5in은 **hardware-informed simulation
   candidate**다. CPU repository consistency **26/26 PASS**, **canonical same-controller simulator
   gate 21/21 PASS**, 각 기체 **16/16 env survival**을 확인했다. P1a/P1b에서 on-policy learning
   outcome을 관측했지만 둘 다 전체 gate는 FAIL했고 held-out capture·장기 재현성·buildability는
   증명되지 않았다. 실제 5B summary가 선언한 robot만 본문 baseline으로 사용한다.

### 원하는 납품물

1. 수정된 PPTX
2. 슬라이드별 제목·핵심 문장·사용 수치·출처 파일을 정리한 Markdown
3. 기존 PPT에서 삭제/교체/부록 이동한 항목 목록과 이유
4. `확정 주장 / 조건부 주장 / 미확정·향후 검증` 3단계 claim table
5. 숫자 대조표: PPT에 들어간 모든 숫자가 어느 summary/receipt에서 왔는지
6. 검증 5B가 아직 없으면 5B 결과 칸이 비어 있는 draft와, 결과가 생겼을 때 교체할 정확한 위치

---

## 2. 발표의 권장 핵심 메시지

검증 5B 결과가 나오기 전의 가장 안전한 한 문장 결론은 다음이다.

> actor에서 GT 표적·장애물 상태를 배제하고 카메라·LiDAR와 시뮬레이터 ego-state를 쓰는 드론
> 요격 시스템을 고밀도 장애물까지 확장해, 단순 성공률 경쟁을 넘어
> **정책–인지 결합, 시간 정합성, 장애물 표현 용량, 제동 실행가능성**을 계측했다. 독립 감사에서
> legacy 학습의 601-step/no-bootstrap 의미 오류를 발견했고, exact-600/timeout-bootstrap fresh PPO
> engineering smoke는 안정적으로 통과했다. 이제 clean multi-seed 5B가 corrected lineage의 최종
> 성능과 기존 결론의 재현 여부를 결정한다.

5B가 완료된 뒤에는 아래 형식으로만 결론을 갱신한다.

> corrected-semantics fresh baseline은 `[5B 평가 밀도 범위]`에서 `[pooled capture 및 CI]`를 보였고,
> `[seed 간 변동]`이었다. legacy frozen policy와의 차이는 `[직접 비교 가능/계보 차이로 기술적 비교만
> 가능]`하며, `[density ceiling 또는 실패 구성]`이 다음 병목으로 남았다.

5B가 실패하거나 seed 변동이 크면 실패를 감추지 말고 다음처럼 쓴다.

> 5A는 의미론 교정의 실행 안정성을 확인했지만, 5B는 `[실패 내용]`을 보여 engineering correctness가
> full-budget 성능 보장을 뜻하지 않음을 확인했다. 따라서 다음 병목은 `[관측/최적화/커리큘럼 중
> 데이터로 지지되는 항목]`이다.

---

## 3. PPT에서 반드시 정정할 검수 내용

### 검증 1 — learned-v2의 비열등성은 재현됐다

- fresh replication seeds 97/101.
- analytic 3,271/4,098, learned-v2 3,272/4,100.
- learned−analytic **−0.0145 pp**, 95% CI **[−1.752,+1.723]**.
- NI margin −2.0 pp 기준 PASS.
- 기존 PPT의 seeds83/89 결과 `−0.07 pp [−1.79,+1.64]`도 별도 실험으로 유효하지만, 한
  슬라이드에서 두 캠페인을 합치거나 episode를 pooled하지 말 것. 본문은 replication 97/101을
  사용하고 83/89는 부록의 최초 gate로 두는 편이 명료하다.

권장 문장: **“특정 learned-v2 artifact는 synthetic nominal navigation에서 analytic bootstrap에
비열등했다.”**
금지 문장: “learned detector 문제가 해결됐다”, “실세계 비전이 완성됐다.”

### 검증 2 — v7의 nominal 손실은 threshold 문제가 아니다

- original system comparison: nominal analytic 80.92%, v7@0.70 75.90%; envelope analytic
  49.88%, v7@0.70 66.63%.
- threshold-isolated fresh diagnostic seeds191/193:
  - v7@0.55−analytic@0.55: **−5.192 pp [−6.982,−3.401]**
  - v7@0.70−analytic@0.55: **−3.752 pp [−5.523,−1.981]**
  - v7@0.70−v7@0.55: **+1.439 pp [−0.407,+3.285]**
- 따라서 핵심은 threshold 0.70이 아니라 analytic detector 통계로 학습된 actor와 v7 출력 통계의
  결합이다. 5C에서 matched fresh training이 필요한 이유로 연결한다.

권장 시각화: analytic@.55, v7@.55, v7@.70 3개 막대와 CI. `threshold culprit`에 X,
`policy–perception coupling`에 화살표.

### 검증 3 — pose-noise 재평가로 하드웨어 허용오차 주장을 철회했다

- 기존 position/yaw arm은 global PyTorch RNG를 소비해 reset/scene randomness를 바꾸는 confound가
  있었다. isolated RNG seed181로 supersede.
- exact 78.49%; position 1/3/10 cm의 Δ는 +0.20/−0.69/−1.47 pp로 CI가 모두 0 포함.
- yaw 0.5° −0.69 pp(CI 0 포함), yaw 2° **−3.28 pp [−5.86,−0.70]**, yaw 5°
  **−12.75 pp [−15.47,−10.03]**.
- 이것은 step-wise iid Gaussian, 단일 environment seed 결과다. 위치 10 cm가 `안전`하다는 뜻이
  아니며, bias/drift/correlated odometry error는 미검증이다.
- 기존 clock ladder는 constant timestamp offset sensitivity일 뿐 clock skew/jitter 규격이 아니다.

권장 시각화: Δ capture forest plot. CI가 0을 지나는 셀과 그렇지 않은 셀을 색으로 구분.

### 검증 4 — 정적 경로는 있었지만 crash 원인은 아직 인과 분리되지 않았다

- 기존 reachability oracle이 dump 좌표 0..40 m를 −20..20 m로 읽은 오류를 수정.
- corrected static 2-D connectivity: inflation 0.40/0.65/0.766 m 모두 **333/333, 100%**.
- 이 결과가 말하는 것은 spawn→종료 시점 target 위치의 정적 경로 존재뿐이다. 동역학, 이동 target
  trajectory, turn/braking, 600-step 도달가능성은 보장하지 않는다.
- contact-time probe: token association 83.2%, 즉 **16.8% 미연결**; hit-in-FOV 85.3%.
- contact-time executed stopping margin 평균 **−0.157 m**. 둘은 같은 missed-obstacle 채널의 두
  증상일 수 있으므로 독립 원인처럼 더하면 안 된다.

권장 시각화: `static connectivity PASS → dynamics untested → contact-time token/braking warning`의
세 단계 funnel. “geometry acquitted” 같은 강한 표현은 금지.

### 검증 5A — corrected semantics는 1,000 epoch에서 안정적으로 학습됐다

- fresh seed197, 128 env, 1,000 epoch, analytic detector, cluster-sector 8/240°, squashed Gaussian,
  governor/pose-noise off.
- endpoint checkpoint SHA `f53489aa9158…`; exact600와 `time_outs` 기록 확인.
- PPO KL max **0.015831**, behavior-KL audit max **0.022902**(<0.04), rollback 0,
  skipped minibatch 0, 모든 action axis raw OOB 0.
- 실제 episode 가중 결과:

| window | episodes | capture | crash | timeout |
|---|---:|---:|---:|---:|
| 전체 | 44,383 | 73.02% | 25.78% | 1.19% |
| speed ramp 1–300 | 15,533 | 66.95% | 31.63% | 1.42% |
| post-ramp 301–1000 | 28,850 | 76.29% | 22.63% | 1.07% |
| 마지막 100 | 4,134 | 78.69% | 20.30% | 1.02% |

- 그러나 전부 on-policy 70-bar training outcome이며 held-out 성능이 아니다.
- `DENSITY_WARMUP=1000`이므로 epoch1000은 정확히 warmup 경계다. density evidence/gate/promotion은
  미검증. checkpoint distance state는 `[20,28]`이지만 general-spawn sampler의 실제 range는
  `[6,28] m`였다. minimum 20 m mastery로 표현하지 않는다.
- dirty worktree에서 실행돼 핵심 4파일 SHA 불변은 확인했지만 full runtime source manifest는 없다.

권장 문장: **“Corrected semantics survived a real fresh PPO smoke without optimizer failure.”**
금지 문장: “5A가 기존 정책을 능가했다”, “고밀도 학습 성공”, “최종 capture 78.69%.”

### 검증 5B — 결과가 나오기 전에는 설계만 표시한다

5B의 최소 계약은 다음과 같아야 PPT 결과로 사용할 수 있다.

- 사전등록한 robot version을 제외하면 exact-600 + `time_outs` 외에는 5A와 동일한 analytic
  detector/current representation/control.
- **robot은 5B 시작 전에 하나로 동결한다.** ref5in을 채택하면 이는 5A legacy-vehicle baseline의
  연속 실행이 아니라 새로운 corrected-reference-platform lineage다.
- clean commit, dirty=false, 전체 runtime source manifest와 checkpoint receipt.
- full-budget, density warmup 이후 실제 gate/promotion을 통과할 수 있는 길이.
- 최소 2개, 가능하면 3개 matched training seeds.
- 동일한 사전등록 held-out evaluation seeds와 density grid.
- training curve와 held-out 결과를 분리. best-reward가 아니라 canonical final/LKG checkpoint 사용.
- seed별 결과, pooled 결과, training-seed variance/CI를 모두 보고.
- 중도 실패 seed를 삭제하거나 성공 seed만 선택하지 말 것.

#### [5B 결과 입력 — 실행 완료 후 Claude가 summary에서만 복사]

| 항목 | 실제 결과 |
|---|---|
| clean commit / source manifest SHA | `[미실행]` |
| robot / URDF SHA / robot-config SHA | `[미실행]` |
| training seeds / budget | `[미실행]` |
| seed별 최종 밀도와 checkpoint SHA | `[미실행]` |
| held-out density grid / eval seeds / episodes | `[미실행]` |
| density별 capture/crash/timeout | `[미실행]` |
| pooled capture와 95% CI | `[미실행]` |
| seed 간 변동 | `[미실행]` |
| NaN/KL/rollback/early-stop | `[미실행]` |
| 5A 및 legacy와 비교 가능한 범위 | `[미실행]` |
| 최종 판정 | `[미실행]` |

5B 슬라이드는 결과에 따라 세 분기 중 하나로 작성한다.

- PASS: corrected lineage의 multi-seed held-out 성능과 density progression을 중심으로 결론 갱신.
- MIXED: 평균보다 seed variance와 실패 seed 원인을 중심으로 제시.
- FAIL: 5A engineering PASS와 full-budget 실패를 대비해 optimizer/curriculum/representation 중 실제
  로그가 지지하는 병목을 제시. 실패 결과를 숨기거나 seed를 교체하지 않는다.

### 기준 플랫폼 후보 분기 — simulator gate와 학습·제작 가능성을 혼동하지 않는다

- legacy `navrl_quad`는 0.250 kg, 모터 팔 ±0.13 m, collision XY 0.28 m 등 서로 다른 scale의
  개발 파라미터가 섞여 있고 exact BOM/CAD로 단일 실제 플랫폼에 추적되지 않는 simulation model이다.
  어떤 실제 기체에도 대응할 수 없다고 증명한 것은 아니다.
- 신규 `navrl_ref5in_quad`: 1.20 kg, 220 mm motor diagonal, collision XY 0.28 m, collision height
  0.12 m, T/W 3.2617, motor tau 0.04 s. 5인치급 부품 자료를 참고한 **hardware-informed simulation
  candidate**이며, 9.60 N/motor와 0.04 s는 exact propulsion 조합에서 식별한 값이 아닌 synthetic
  prior다. 높이 0.08→0.12 m는 45° tilt에서 한 축 projected support를 약 **+2.83 cm** 바꾸는
  geometry confound다. exact BOM/CAD/CG/prop clearance/sensor FOV로 buildability가 검증된
  reference platform은 아니다.
- CPU repository consistency **26/26 PASS**, schema-2 **canonical same-controller simulator gate
  21/21 PASS**다. legacy/ref5in 각각 **16/16 env survival**, finite state/actuator와 선택한
  hover/step/reversal/yaw/100 Hz fixed-gain roll·pitch gate를 확인했다.
- forward 정상상태/t90은 둘 다 2.490 m/s / 0.8 s, reversal 0-cross/t90은 0.5/1.0 s였다.
  ref5in의 100 Hz peak body rate는 더 낮았으므로 이 폐루프 PASS를 intrinsic plant 동등성이나
  민첩성 증거로 해석하지 않는다.
- held-out 장애물 회피·capture/crash/timeout, 장기·multi-seed 재현성 및 buildability는 증명되지
  않았고 hardware validation도 아니다. P1a/P1b on-policy outcome을 held-out 성능으로 쓰지 않는다.
- 5B가 ref5in을 쓰려면 먼저 P1c fresh 900-epoch learning-viability gate와 held-out P2를 통과하고,
  실제 P3 receipt에 `robot_name`, URDF/config SHA를 기록해야 한다.
- 현재 순서는 **P1a FAIL → P1b FAIL → P1c PASS → P2 seed 313 strict FAIL(timeout 5.56%)
  → P3 seed 211 / full-budget 5B `[차단·미실행]`**이다. on-policy P1 수치로 뒤 단계 칸을 채우지 않는다.

권장 문장: **“The hardware-informed simulation candidate passed repository consistency (26/26) and a
canonical same-controller simulator gate (21/21; 16/16 env survival per model). P1c passed its
on-policy engineering gate, but P2 failed the preregistered held-out timeout ceiling (5.56% > 5%);
full-budget performance and buildability remain unproven.”**
금지 문장: “ref5in은 legacy와 동등한 요격 성능이다”, “기체 현실화가 성능에 영향이 없다.”

---

## 4. 권장 슬라이드 구성

### 본문 17장

1. **MOTAR는 카메라·LiDAR와 시뮬레이터 ego-state로 밀집 장애물 속 이동표적을 요격한다**
   문제 장면, 한 문장 연구 질문, 40×40×3 m와 이동표적.

2. **난점은 추격 속도가 아니라 인지·밀도·제동의 결합이다**
   partial observation, occlusion, moving target, dense bars를 한 그림으로.

3. **Actor에서 GT를 차단하고 critic/reward에만 제한했다**
   RGB-D+LiDAR→detector→tracker→tokens와 simulator ego-state→Transformer PPO→riskcap 흐름도.
   GT 표적·장애물 상태가 actor에는 없고 critic/reward에만 들어가는 경계를 점선으로 분리.

4. **기존 계보는 고밀도에서 80% 수준에 도달했지만 legacy semantics였다**
   Gate 1의 205 bars ep25000/riskcap capture 80.28%와 crash 17.37%; 화면에
   `legacy 601/no-bootstrap` 경고. 이 수치를 corrected lineage와 직접 연결하지 않는다.

5. **Riskcap의 이득은 추가 PPO 적응보다 컸다**
   Gate 1 A/B/C 밀도 곡선. ID pooled riskcap +5.82 pp [4.60,7.04], adaptation +1.56 pp
   [0.43,2.69]. 220 OOD 회색.

6. **표적 속도의 capture 비용은 밀도가 높을수록 커졌다**
   Gate 2 interaction χ²=12.7603, p=0.000354. 130→205 fast-slow −0.64→−5.87 pp.
   crash mechanism까지 확정하지는 않는다.

7. **감사는 높은 성공률보다 먼저 실험 의미를 바로잡았다**
   chirality, reset indentation, RNG coupling, reachability coordinates, 601/no-bootstrap을
   타임라인으로. 오류를 나열하는 슬라이드가 아니라 “각 오류가 어떤 주장을 무효화했는가”를 연결.

8. **Learned-v2는 비열등했지만 detector 일반 문제의 종결은 아니었다**
   검증1 forest/bar. synthetic nominal 한정.

9. **v7의 nominal 손실은 threshold가 아니라 정책–인지 결합이었다**
   검증2 3-arm threshold-isolation 차트. 5C matched training 동기.

10. **시간 정합성은 latency 손실 대부분을 회수하지만 pose 오차에는 한계가 있다**
    legacy capture-time pose recovery(37.82→78.04)는 semantics 라벨을 분리하고, 오른쪽에는
    isolated-RNG yaw forest plot. constant offset와 skew/jitter를 구분.

11. **정적 길은 있었지만 접촉 순간 표현과 제동은 동시에 부족했다**
    corrected connectivity 100%, token miss 16.8%, stopping margin −0.157 m. 인과 미확정 표시.

12. **hardware-informed candidate는 canonical simulator gate를 통과했지만 과제·제작 가능성은 아직 모른다**
    legacy/ref5in 파라미터 비교를 전부 보여주지 말고 mass, motor diagonal, collision XY, T/W와
    CPU **26/26 PASS**, **canonical same-controller simulator gate 21/21 PASS**, 각 기체 **16/16 env
    survival**과 P1a/P1b strict FAIL을 요약. 0.12 m height의 45° projected-support **+2.83 cm**
    confound와 9.60 N/motor·tau 0.04 s가 synthetic prior임을 각주로 표시. `held-out capture /
    buildability unproven` watermark. 실제 5B가 legacy를 쓰면 부록으로 이동.

13. **두 줄의 종료 의미 오류가 corrected fresh lineage를 필요하게 했다**
    `>600→>=600`, `timeouts→time_outs`를 작은 코드/sequence diagram으로. legacy 결과는 소급
    수정되지 않는다는 점 강조.

14. **5A는 corrected semantics의 실행 안정성을 통과했다**
    KL/rollback/finite/checkpoint contract 위주. 최근 capture는 작은 보조 차트로만 표시하고
    `training-only, 70 bars, single seed` watermark.

15. **P1c→P2→P3 gate가 ref5in의 5B 진입을 통제한다**
    `5A 완료 → P1a FAIL → P1b FAIL → P1c PASS → P2 seed 313 strict FAIL
    (timeout 5.56% > 5%) → P3 seed 211/full 5B [차단·미실행]` 타임라인. P1c는 P1b에서
    budget만 750→900으로 바꾼 engineering gate이고, P2의 capture/crash PASS가 timeout FAIL을
    상쇄하지 않는다.

16. **5B는 robot까지 동결한 clean multi-seed baseline으로 최종 결론을 결정한다**
    P2가 실패한 현재는 설계도와 `[차단·미실행]`; 새 gate를 사전등록해 통과한 뒤에만 seed별
    density held-out plot으로 교체.
    실제 receipt의 robot을 표기하고 성공/혼합/실패 분기 중 실제 결과 하나만 사용.

17. **현재 기여는 최고 숫자보다 병목을 측정하고 분리한 방법에 있다**
    (a) dense-clutter moving-target interception, (b) policy–perception coupling 계측,
    (c) collision-causing obstacle representation audit, (d) corrected reproducible lineage.
    다음 단계는 5C perception adaptation, pre-contact timeline, backup-braking, token capacity 순.

### 부록 권장

- A1: v1/v2/legacy/schema-v2 데이터 사용 규칙과 checkpoint lineage tree.
- A2: Gate 1 전체 A/B/C 표와 220 OOD.
- A3: detector v2 최초 NI와 replication을 분리한 전체 표.
- A4: pose-noise CI 전체표, 미검증 bias/drift/skew 목록.
- A5: 5A endpoint receipt·SHA·테스트와 duplicate checkpoint 수정.
- A6: 5B seed별 checkpoint/receipt/held-out raw table 및 실패 seed 포함.
- A7: legacy/ref5in full parameter table, URDF/config SHA와 canonical same-controller simulator gate 한계.

---

## 5. 기존 PPT에서 유지·교체·삭제할 항목

### 유지

- 연구 문제, actor의 카메라·LiDAR + simulator ego-state와 privileged critic/reward의 GT 경계 분리도.
- v2 40×40×3 m 환경과 density/target-speed 계약.
- Gate 1 riskcap 분해, Gate 2 density×speed interaction.
- timestamp-aware pose correction의 공학적 교훈.
- sim-to-real 한계와 bar-contact ceiling.

### 교체

- `learned detector 비열등 → 문제 해결` 서사를 검증1/v7 검증2의 두 단계 서사로 교체.
- `geometry가 원인이 아니다`를 `정적 final-target connectivity는 100%; dynamics는 미검증`으로 교체.
- pose tolerance 사양을 isolated-RNG forest plot과 “hardware spec 아님”으로 교체.
- `다음 fresh run` 계획을 5A 실제 결과 + P1a/P1b FAIL + P1c PASS + P2 strict FAIL + P3 차단 gate와
  5B 설계/결과로 교체.
- legacy 250 g 모델을 traceable BOM/CAD reference처럼 설명한 그림은 mixed-scale development model,
  ref5in 제안과 실제 5B robot 계약으로 교체.
- 자동 run summary의 peak capture나 단일 epoch 값은 가중 창/held-out 값으로 교체.

### 삭제 또는 부록 이동

- v1과 v2의 선형 기울기 직접 비교.
- `riskcap gain grows with density` 유의성 주장.
- H4 69%를 확정적 메커니즘으로 표현한 문장.
- 220 bars를 ID 또는 학습범위로 보이게 하는 그래프.
- 5A 78.69%를 최종 benchmark로 보이게 하는 headline.
- “위치 ≤10 cm 무료”, “yaw ≤1° 허용”, “+20 ms 허용” 같은 하드웨어 규격 문장.

---

## 6. 최종 claim table

| 단계 | PPT에 써도 되는 주장 |
|---|---|
| 확정 | Gate 1에서 riskcap ID pooled +5.82 pp; Gate 2 density×speed interaction p=0.000354; learned-v2 NI replication PASS; v7 nominal coupling loss; corrected static connectivity 333/333; 5A optimizer/semantics engineering PASS; ref5in CPU repository consistency 26/26 및 canonical same-controller simulator gate 21/21 PASS(각 기체 16/16 env survival) |
| 조건부 | capture-time pose correction은 정확 timestamp/pose history 전제; yaw 민감도는 iid Gaussian 단일 seed; token/stopping diagnostics는 contact-time 기술 통계; ref5in은 hardware-informed simulation candidate일 뿐 hardware reference가 아님; 0.12 m height는 +2.83 cm projected-support confound이고 9.60 N/tau 0.04 s는 synthetic prior; legacy와 corrected lineage 비교는 기술적 참고 |
| 미확정 | ref5in held-out/장기 capture와 buildability, 5B 최종 성능, density ceiling의 corrected-lineage 재현, token miss와 stopping margin의 독립 인과, real-world detector 성능, clock skew/jitter, hardware pose tolerance |

---

## 7. Claude 검수 완료 체크리스트

- [ ] 5B 미실행 값을 숫자로 채우지 않았다.
- [ ] 5A에 `engineering-only / 70 bars / single seed / no held-out`가 표시됐다.
- [ ] legacy 601/no-bootstrap과 exact-600/bootstrap이 색 또는 패널로 분리됐다.
- [ ] learned-v2와 learned-v7 artifact가 혼동되지 않는다.
- [ ] 모든 pp, CI, seed, episode 수가 원본 summary와 일치한다.
- [ ] 220 bars가 OOD 회색으로 표시됐다.
- [ ] pose 결과에 iid/single-seed와 hardware spec 아님이 표시됐다.
- [ ] reachability는 static final-target path로 한정됐다.
- [ ] token miss/stopping margin을 독립 인과로 더하지 않았다.
- [ ] 5B 실패 seed나 early-stop이 있다면 숨기지 않았다.
- [ ] 5B의 robot/URDF/config SHA가 표시됐고 legacy/ref5in 결과가 연속 계보처럼 연결되지 않았다.
- [ ] P1a/P1b FAIL, P1c PASS, P2 strict FAIL, P3 차단·미실행이 한 타임라인에서 구분됐다.
- [ ] actor 입력을 카메라·LiDAR + simulator ego-state로 표시하고 sensor-only라고 과장하지 않았다.
- [ ] ref5in의 0.12 m height confound(+2.83 cm)와 9.60 N/tau 0.04 s synthetic prior를 표시했다.
- [ ] ref5in을 hardware-informed simulation candidate로 표시하고, CPU 26/26 및 canonical
  same-controller simulator gate 21/21 PASS(각 기체 16/16 env survival)나 P1 on-policy outcome을
  held-out capture, 장기 재현성, buildability 또는 hardware validation으로 과장하지 않았다.
- [ ] 결론 슬라이드가 최고 단일 epoch capture가 아니라 재현 가능한 기여를 말한다.

이 체크리스트 중 하나라도 실패하면 최종 PPTX로 표시하지 말고 draft로 돌려주세요.
