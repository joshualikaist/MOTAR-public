# 검수 요청 브리프 #3 — 검증 3(P3 전제 민감도) · 검증 4(bar-contact ceiling 분리) + 검증 5 설계 판단

수신: 검수 세션(Codex) / 작성: Claude 세션 / 2026-08-12
대상 커밋: **`589c558..47d836b`** (+ 구현 커밋 `35ac025`), 브랜치 `research/navrl-env`, push 안 함
동반 브리프: `docs/review_brief_2026-08-12_verification_1_2.md` (검증 1·2, 별도 발송됨)

> GPU는 현재 **유휴**다. 실행 중 캠페인이 없으므로 read-only 제약 없음 — 재현 실행 가능.

관례대로 반증 포인트를 앞세운다. §3은 검증 5 시작 전 판단을 요청하는 **설계 질문**이다.

---

## 0. 이번 범위에서 주장하는 것

**검증 3** (`results/navrl_v2_pose_premise_seed163/`): P3의 −2.5 pp latency 잔차는 정확한
timestamp/pose 전제 조건부이며, 그 전제의 예산은 — clock 정렬 **+20 ms 이내**(비대칭:
−50 ms는 −2.8 pp, **+50 ms는 −17.3 pp**), odometry yaw **≤1°**(5°는 −12.5 pp),
위치 **≤10 cm 무료**. 내장 앵커(δ=+τ → 39.30% ≈ naive 수준)로 노브 유효성을 캠페인 안에서 증명.

**검증 4** (`results/navrl_v2_bar_ceiling/`): 205-bar 잔여 crash의 3단 분해 —
① 기하 무죄(정적 연결률: 낙관 100% / governor 97.3% / 비관 94.6%),
② 충돌 막대의 **16.8%가 8토큰 입력에 부재**(4배 과밀, rank 0 = 표현되면 항상 최근접),
③ 접촉 시 stopping margin 음수(executed −0.157 m / requested −1.162 m).

---

## 1. 검증 3 반증 포인트

### (A) 비대칭의 "메커니즘" 서사는 미검증이다
+δ 붕괴를 "오염된 추정이 진행 방향 carve-out을 구동해 경로 위 막대를 지운다(P2 채널 재발화)"로
설명했다. **직접 계측은 없다** — crash 증가 패턴(35.5% vs 21.3%)과 P2 유추뿐. carve-out
발화율/삭제된 bin의 전방 편중을 직접 재지 않았다. 서사를 논문에 쓸 수 있는 수준인지 판정 요망.

### (B) 교란 모델의 형태 선택
- clock offset은 **run 전체 상수**다. 실제 clock skew는 상수+지터인데 지터 성분은 안 쟀다.
- yaw noise는 **스텝별 iid 백색**이다. 실제 odometry yaw 오차는 느린 드리프트/바이어스가
  지배적일 수 있고, 바이어스는 carve-out을 계통적으로 한쪽으로 밀어 백색보다 나쁠 수 있다.
- pose 보간은 sign-aligned **nlerp**다(스텝간 회전 소각 가정). slerp 대비 검증 안 함.
셋 다 "더 현실적인 모델이면 결과가 나빠질 수 있는" 방향이다.

### (C) 일반화 범위
- 단일 시드(163), 셀당 2,049 ep — 작은 효과(−3 pp대)는 CI가 0에 근접.
- **analytic bootstrap 스택에서 측정**했다(검증 2와 분리 목적). v7 스택에서 전제 예산이
  같은지는 미측정 — "실기 스펙"으로 일반화할 때의 단서 누락.
- pos noise 비단조(0.03: −2.78 vs 0.10: −1.42)는 SE 안이지만 문서에 명시했는지 확인 요망.

---

## 2. 검증 4 반증 포인트

### (D) oracle은 정적 연결성이다 — "기하 무죄"가 과대주장인가 (최우선)
- spawn → **최종 표적 위치**의 정적 경로 존재만 확인했다. 동역학(선회반경·제동), 이동 표적의
  궤적, 600-step 예산은 모두 무시한다. 정적 연결은 **필요조건이지 충분조건이 아니다**.
- 막대를 원반으로 근사(3반경 괄호). per-bar 실측 크기 없음.
- 사전등록 판정선(비관 ≥95%)에 94.6%로 **미달**인데 "낙관 100% + 상계 논리"로 무죄를
  선언했다 — 등록 기준을 사후에 완화한 형태다. 이 판정이 유지 가능한지.

### (E) "16.8% 부재"는 접촉 순간의 스냅샷이다
bar probe는 **접촉 시점**에만 토큰 포함 여부를 잰다. 그 막대가 접근하는 동안(2~5스텝 전)
토큰에 있었는지는 모른다 — 있었다면 인과가 "표현 부재"가 아니라 "표현됐는데 회피 실패"다.
시계열 커버리지(접촉 전 k스텝 창)를 재지 않은 것이 이 숫자의 인과 해석을 약화시킨다.

### (F) ③의 내부 불일치 후보
접촉 시 governor 평균 clearance **0.84 m**인데 접촉했다 — clearance 입력(스캔 유래)이 충돌
막대를 못 보고 있었다는 뜻일 수 있고, 그렇다면 ②(부재)와 ③(여유 음수)는 독립 축이 아니라
**같은 원인의 두 증상**이다. 나는 이를 합성이라고 썼지만 결합 구조를 분해하지 않았다.
requested margin −1.16 m도 "정책이 과속 요청" 서사에 썼는데, 정의를 확인해 보니
`stopping_margin = usable(clearance 유래) − (v·reaction + v²/2a)`라 **근접 상태에서는 높은
요청 속도만으로도 정의상 음수가 되기 쉽다**(speed_governor.py:221-226). 즉 −1.16 m 자체는
"과속 요청" 서사를 지지하지만 근접-정의 효과와 분리돼 있지 않다. executed −0.157 m이 더
방어 가능한 수치다 — 서사의 무게를 어디에 둘지 판정 요망.

### (G) 표본
단일 셀, 단일 시드(167), contact 333. dump는 timeout 60건 제외(종료 경로 차이) — contact
분석엔 무영향이지만 outcome 분포 재구성에는 불완전.

---

## 3. 검증 5 설계 판단 요청 (시작 전)

frozen 정책 계보는 legacy **601-action + no-`time_outs`-bootstrap**으로 학습됐다. "수정된
알고리즘의 학습 결과" 주장을 위한 fresh PPO 1회가 남은 마지막 항목이다. 선택지:

- **(A) 최소**: exact-600 + bootstrap만 고쳐 재학습. 주장 최소, 리스크 최소.
- **(B) A + v7 detector + appearance envelope randomization 위에서 학습**: 검증 2의 E1 FAIL
  (정책-인지 결합)을 학습으로 푸는 정공법. 성공 시 "envelope 안 robust end-to-end" 주장.
  실패 시 (A)로 후퇴. **내 추천**: (A)를 스모크로 짧게 → (B) 본학습.
- **(C) B + 검증 4 개입(토큰 capacity/FOV 또는 margin)**: 변수 3중첩 — 분리 불가로 나는 반대.

**질문**: ① (A)/(B)/(C) 판정과 근거. ② (B)라면 사전등록에 무엇을 박아야 하는가 —
비교 기준(legacy 계보와 비교 가능성 없음을 어떻게 다루나), 게이트(무엇으로 성공 판정),
예산(epoch/중단 규칙), envelope 하 평가 프로토콜. ③ 검증 3의 전제 스펙을 (B) 학습·평가
계약에 반영해야 하는가(예: 학습에도 pose 교란 randomization을 넣을지).

---

## 4. 검수 불필요 (자체 검증 완료)

- 검증 3 노브의 산술: 앵커(δ=+τ=naive) 캠페인 내 재현 + 단위테스트 38/38
  (δ=0 정확 pose, 중점 보간, 음수 offset, yaw 단위노름).
- 검증 4 dump의 정보 방화벽: GT는 디스크 전용, actor/critic/reward/종료 무접촉(bar probe와
  동일 원칙), 계측 셀 성능이 앵커와 정합(80.09/16.98 vs 80.54/17.17)해 관측 오염 없음 방증.
- oracle 구현: 0.1 m 격자 + KD-tree + connected components; spawn/goal의 양자화 스냅 0.6 m.

---

## 부록 — 재현

```bash
cd aerial_gym/rl_training/rl_games
PREFLIGHT=1 ./eval_navrl_v2_pose_premise.sh                      # 검증 3 계약
PYTHONNOUSERSITE=1 python ../../../tests/test_navrl_latency_compensate.py   # 38
# 검증 4 oracle 재실행 (CPU ~3분):
cd ../../..
PYTHONNOUSERSITE=1 python tools/analyze_navrl_v2_reachability.py \
  results/navrl_v2_bar_ceiling/episodes_seed167.npz --output /tmp/reach_check.json
```
