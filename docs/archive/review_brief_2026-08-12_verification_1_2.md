# 검수 요청 브리프 #2 — 검증 1(NI 재현) · 검증 2(perception domain shift)

수신: 검수 세션(Codex) / 작성: Claude 세션 / 2026-08-12
대상 커밋: **`9bb5222..16438f9`** (16 커밋, 브랜치 `research/navrl-env`, **push 안 함**)
이전 브리프: `docs/review_brief_2026-08-10.md` → 판정 `docs/codex_review_2026-08-10.md` (완료)

> **동시 실행 주의.** 검증 3(pose-premise 12셀)이 **지금 GPU에서 실행 중**이다. 각 셀이
> runtime source manifest를 해시해 대조하므로, **캠페인 종료 전에 `aerial_gym/**` 소스를
> 수정하면 남은 셀이 거부된다**(2026-08-06에 실제 발생). 검수는 read-only로 진행하고,
> 코드 수정 제안은 diff가 아니라 지적으로 남겨 달라. 완료 여부는
> `results/navrl_v2_pose_premise_seed163/summary.md` 존재로 판단.

이 문서는 성과 요약이 아니라 **반증 포인트 목록**이다. 검수 시간을 위험한 곳에 쓰기 위해
"검수 불필요" 항목도 명시했다.

---

## 0. 이번 범위에서 확정했다고 주장하는 것

1. **검증 1**: Gate 3 navigation NI가 새 시드(97/101)에서 재현. pooled learned−analytic
   **−0.015 pp**, 95% CI [−1.752, +1.723], margin −2.0 pp → PASS. 원 결과(83/89: −0.073 pp,
   CI [−1.790, +1.644])와 **통합하지 않고 별도 보고**(사전등록).
2. **검증 2**: pure-red 외형 가정을 8축 노브로 계측 가능하게 만들고, detector를 envelope
   하에서 재학습(v7, offline gate 14/14 PASS)한 뒤 navigation을 측정했다. 결론:
   **인지 모듈 robustness가 시스템 robustness로 자동 승격되지 않는다** —
   E1(nominal NI) **FAIL −5.02 pp**, E2(envelope 비용) −14.29 pp, E3(bootstrap 붕괴) −31.04 pp.

---

## 1. 최우선 — 여기가 틀리면 검증 2의 결론이 무너진다

### (A) E1 FAIL의 해석: "결합" vs 단순 threshold 오정합

**내 주장**: v7이 nominal에서 −5.02 pp인 것은 정책이 analytic의 픽셀 통계에 결합돼 있기 때문.

**가장 강한 대안 설명(내가 배제하지 못했다)**: learned arm은 threshold **0.70**으로 돌았고
정책은 **0.55**로 학습됐다. 즉 −5.02 pp가 "결합"이 아니라 **운영점 오정합 하나**일 수 있다.
0.70은 v7의 **envelope validation**에서 선택된 값이며 **nominal에서는 한 번도 보정된 적이
없다**(§1-B의 프로토콜 공백과 같은 뿌리).

**검수 요청**: 이 대안이 지배적이라면 "결합" 서사는 과장이다. 판정과, 필요하다면 분리
설계(예: nominal 재보정 threshold arm 추가)의 타당성을 봐 달라. 나는 이것을 WORKLOG에
"미검증 후보 (b)"로만 적었고 본문 서사는 결합 쪽으로 썼다 — 균형이 맞는지가 핵심이다.

### (B) 사전등록의 진정성: v6 FAIL → v7 재등록이 "통과할 때까지"인가

시간순: v3/v4/v5(test 3회 관찰) → **exploratory 격하 선언** → v6 confirmatory 사전등록 →
v6 FAIL(12/14) → **v7 confirmatory 재등록**(새 시드, 용량 증가 + 선택 마진) → v7 PASS.

- 각 confirmatory는 새 시드·1회 개봉·재시도 없음을 지켰다.
- 그러나 **v7은 v6의 실패를 보고 설계됐다.** 엄밀히는 confirmatory 2회이며 type-I 오류가
  누적된다. 나는 이것을 "감사가 지목한 두 지렛대를 고쳤다"로 정당화했다.

**검수 요청**: 이 절차를 (a) 정당한 순차 설계 개선, (b) 보고 시 다중검정 보정 필요,
(c) v7을 exploratory로 격하하고 3차 confirmatory 필요 — 중 무엇으로 볼지 판정해 달라.
논문에 "사전등록 confirmatory PASS"라고 쓸 수 있는지가 여기서 갈린다.

### (C) gate 재정의(pixel precision tolerance)의 사후성

v5가 pixel precision으로 FAIL한 **뒤에** 그 체크를 1px dilation tolerance로 재정의했다.
물리적 근거는 있다(모션 블러가 표적색을 인접 픽셀에 섞는데 GT 마스크는 순간 기하만 라벨).
그러나 **실패한 체크를 완화한 것**이라는 형태는 남는다. recall/IoU는 exact 유지,
임계 0.95 불변, spray FP는 여전히 벌한다(v7 absent FPR 0.0000이 근거).

**검수 요청**: 이 완화가 정당한 metric 수정인지, 아니면 gate 신뢰를 훼손하는지.

### (D) envelope 선택이 같은 측정군에서 나왔다

envelope(hue ±60°)은 **stage-A 사다리에서 bootstrap이 무너지기 시작하는 지점**을 보고
정했다. 즉 "선언된 운용 envelope"이 데이터 독립적으로 정해진 값이 아니다.

**검수 요청**: 이 순환이 결과 해석을 얼마나 약화시키는지. 나는 "선언된 envelope 안에서만
주장"으로 범위를 좁혔지만, envelope 자체가 사후 선택이라는 점은 문서에 명시하지 않았다.

---

## 2. 2순위 — 방법·구현 감사

### (E) `NAVRL_V2_FORCE=1`의 부작용 범위

navigation A/B의 **learned arm에만** force를 걸었다(§A의 0.70 vs 0.55 때문). 그런데 이 플래그는
평가 스크립트에서 **모든** 계약 불일치를 WARNING으로 강등한다(`eval_navrl_v2_density_sweep.sh:656`).

**내가 이미 확인한 것**: 로그의 `WARNING (forced)` 블록에 나열된 항목은
`cfg_detector_threshold` **한 건뿐**이며 arena/pool/placement/episode/governor는 정상 출력됐다.
**검수 요청**: 이 확인으로 충분한지, 아니면 threshold만 예외 처리하는 구조로 바꿔야 하는지.

### (F) 렌더러 8축 구현의 물리적 타당성

- 캘리브레이션 축은 **렌더러만** 교란하고 perception의 intrinsics/extrinsics 사본은
  nominal로 둔다(두 곳을 다 바꾸면 상쇄). 이 모델링이 실제 mis-calibration과 맞는가?
- FOV 오차는 ray table이 `__init__`에서 1회 베이크되므로 **per-run**이다(다른 축은 per-episode).
  이 비대칭이 결과 해석을 왜곡하는가?
- 모션 블러는 RGB만 EMA로 처리하고 **depth는 블러하지 않는다**(명시적 결정). 타당한가?
- hue 회전은 회색축 Rodrigues(휘도 보존)다. 포화 red는 회전 시 sRGB 큐브를 벗어나 clamp된다 —
  이 clamp가 "hue만 바꾼다"는 주장을 훼손하는가?

### (G) v7 stage-A 사다리의 threshold 불일치

사다리는 전 셀 **0.55**로 측정했는데 v7의 운영점은 **0.70**이다. 즉 사다리의 v7 곡선은
운영 threshold가 아닌 값에서의 곡선이다. 비교 대상(bootstrap 0.55)과 맞춘 결과지만,
"envelope 안 96~99%" 수치가 실제 운용 조건의 수치는 아니다. **문서에 이 단서를 안 달았다.**

### (H) albedo 0.5(envelope 밖)에서 learned FPR 7.6%

envelope 안에서는 0.0%인데 밖에서 위양성 모드가 열린다. 나는 "배경 분포 확장 필요"로만
기록했다. 이것이 v7 채택을 막을 정도의 신호인지 판단 필요.

---

## 3. 검수 불필요 (자체 검증 완료, 증거 있음)

- **zero-knob 비트 동일성**: 렌더러 8축 전부 0일 때 rgb/depth가 종전과 `torch.equal`
  (GPU 스모크). 기존 비-appearance 결과는 재기준선 불필요.
- **아키텍처 디스패치**: artifact `meta.architecture`로 1×1/spatial/wide 생성, 긴 prefix 우선
  (Wide가 좁은 태그로 startswith 매칭되는 함정을 테스트로 고정).
- **테스트**: perception 28, latency 38, appearance 12 전량 PASS.
- **검증 1 무결성**: 4셀 policy/detector SHA·동일 source manifest·exact-600 timeout 독립 재계산 PASS.
- **launcher 파생 사고 2건**(summarizer seed 리터럴, `model.classifier` 하드코딩)은 발견·수정·
  기록 완료. 전자는 셀 데이터 무손상이라 요약만 런처 밖에서 재계산하고 그 사실을 파일에 명시.

---

## 4. 받고 싶은 판정

1. §1 (A)~(D) 각각 **동의 / 조건부 동의 / 반박** + 근거
2. 특히 (B): 논문에 "preregistered confirmatory PASS"로 쓸 수 있는가
3. (A)가 지배적이면 검증 2의 헤드라인 문장을 어떻게 고쳐 써야 하는가
4. 놓친 confound (지난 검수에서 seed 42→47과 evaluator SHA 차이를 지적받은 전례가 있다)

---

## 부록 — 재현

```bash
# 전부 inference/offline. 검증 3 실행 중에는 PREFLIGHT도 돌리지 말 것(소스 해시는 안 건드리지만
# GPU 경합). 완료 후:
cd aerial_gym/rl_training/rl_games
PREFLIGHT=1 ./eval_navrl_v2_detector_navigation_ab_replication.sh   # 검증 1
PREFLIGHT=1 ./eval_navrl_v2_appearance_navigation_ab.sh             # 검증 2 E1/E2/E3
PYTHONNOUSERSITE=1 python ../../../tests/test_navrl_detector_appearance.py    # 12
PYTHONNOUSERSITE=1 python ../../../tests/test_navrl_latency_compensate.py     # 38
```

핵심 산출물:
`results/navrl_v2_detector_navigation_ab_replication_seed97_101_schema2/summary.md`,
`results/navrl_detector_domain_shift{,_v7}/summary.md`,
`results/navrl_detector_offline_gate_v{3,4,5,6,7}*/summary.md`,
`results/navrl_v2_appearance_navigation_ab_seed151_157/summary.md`.
