# 논문 삽입용 초안: Method와 Experiments

본 초안은 2026-09-07에 보존된 시뮬레이션 자료의 재분석을 기술한다. 새 GPU 실험은 수행하지 않았다. 수치표는 독립적으로 재계산한 [recomputed.json](recomputed.json)에서 생성한다. 이하 `arc`는 실제 평가에 사용된 arc-clearance speed filter를 뜻하며, 완전한 DWA 탐색기나 충돌 회피를 보증하는 제어기를 의미하지 않는다.

## Method

### M1. 과제, 평가 조건 및 정책

**Setup.** 평가 과제는 정적 막대 장애물이 있는 40 × 40 m 필드에서 움직이는 표적을 요격하는 것이다. 환경 설정 높이는 3 m다. 주분석은 `v2` 계약의 `navrl_quad`와 frozen ep25000 정책 하나를 사용한다. 정책의 SHA-256은 `f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40`이고, 평가 중 가중치를 업데이트하지 않았다. 기체 asset은 `quad_navrl_collide.urdf`, SHA-256은 `ad9a3110037cb01f94caa11054e367e557e9089b4edd520d0aa59b78578988be`다. 별도 D4 실험은 `navrl_ref5in_quad`와 ep1900에서 분기한 재적응 정책을 사용하므로 주분석과 동일한 기체·정책으로 간주하지 않는다.

**Measurement.** 장애물 수는 70, 100, 130, 160, 205였다. 주분석의 기록된 초기 목표거리 범위는 **6–28 m**, 목표 속도는 **U[0.3, 1.5] m/s**였다. `target_pattern=mixed`, `reflection_mode=original`, `target_route_mode=off`, `distractor_count=0`을 사용하였다. 최종 목표거리·속도 분포는 checkpoint의 curriculum clock과 독립적으로 고정했다. 환경 128개를 병렬 실행하고 cell별 목표 종료 episode 수를 2,049로 지정하였다. 동시 종료로 분모가 달라지므로 결과 JSON의 `actual_episodes`를 사용했다. 물리 time step은 0.01 s, policy step당 물리 step은 10회, policy 주기는 0.1 s였다. episode 길이는 최대 600 policy actions, 즉 60 s다.

**Result.** 주분석은 평가 seed 523, 527, 531, 5개 density, 3개 arm의 **45 cell, 실제 종료 episode 기록 92,227개**다. cell당 분모는 2,049–2,051이었다. 정책 checkpoint 수는 1개이며 평가 seed 3개는 독립 학습 seed 3개를 의미하지 않는다. 주분석의 시작거리 6–28 m를 ref5in 실험의 22.5–28 m로 바꾸어 기술하지 않는다.

**Interpretation.** 본 설계는 특정 frozen policy에 평가 시 적용하는 필터를 바꿨을 때의 차이를 추정한다. 새 학습 정책군이나 미관측 기체에 대한 평균 효과를 직접 추정하지 않는다.

**Limitation.** 결과는 시뮬레이션 계약에 종속된다. aggregate CSV에 전체 episode의 layout/scene ID가 없어 같은 RNG seed의 arm을 완전히 paired scenario로 볼 수 없다. 실제 RGB-D 센서 및 LiDAR 로그나 실기 비행시험으로 검증한 결과는 아니다.

### M2. 센서 입력과 관측 표현

**Setup.** actor는 원시 RGB 이미지를 직접 입력받는 end-to-end 영상 정책이 아니다. RGB-D 기반 단일 target detection 및 Kalman track, LiDAR, ego-state에서 구성한 structured observation을 사용한다. 기본 RGB-D 해상도는 160 × 90, 시야각은 87° × 58°, detector range는 20 m이며 배경 depth 경로는 40 × 24다. LiDAR는 수직 4 × 수평 72 rays, 수평 360°, 수직 +20°부터 −10°, 최대거리 12 m다. obstacle selector의 시야는 240°이며 최대 8개 obstacle representation을 유지한다. static scan의 360° 범위와 obstacle selector의 240° 범위는 구분한다.

**Measurement.** 관측 차원은 `288 + 5×8×12 + 5×10 + 5×16 = 898`이다. 각각 current static scan 288, obstacle history 480, ego history 50, target-track history 80차원을 뜻한다. 각 history는 0.5 s 간격의 표본 5개로 약 2 s를 포함한다. Transformer token 구성은 CLS 1개, static 1개, obstacle-history 5개, ego-history 5개, target-history 5개로 총 17개다. embedding dimension은 64, encoder는 4 layers, attention은 4 heads, feed-forward dimension은 128이다. 평가 action selection은 deterministic이다. actor의 x/y 출력은 축별 최대 2.5 m/s이므로 horizontal norm의 요청 상한은 √2 × 2.5 = 3.5355339 m/s다. yaw-rate 명령 상한은 3.0 rad/s다. actor의 z 출력은 prev-action 관측에 유지되지만, 실행 z 명령은 1 m altitude PI가 덮어쓴다. 이후 velocity/attitude loop와 motor allocation은 고정이다.

**Result.** principal narrow-arm 비교에서 바꾼 변수는 governor mode다. policy weight, actor observation layout, target-association 계약 및 vehicle dynamics는 고정하였다. LiDAR의 target-like return 제거는 camera/LiDAR association에 의존하며, simulator의 ground-truth obstacle identity를 governor 입력으로 제공하지 않는다.

**Interpretation.** actor의 센서 기반 입력과 사후 분석용 ground-truth 접촉 계측을 구분한다. 아래 contact geometry는 평가용 위치·기하로 계산한 값이며 policy 또는 governor의 sensor input이 아니다.

**Limitation.** 시뮬레이션의 target 렌더링과 NPS-Drones 실사 detector의 검증은 다른 실험이다. NPS validation AP 0.579를 본 policy detector의 성능으로 대입하지 않는다. 또한 `perception_perturb=false`인 실행의 receipt에 dropout 설정값 0.3이 남아 있다고 해서 실제 30% dropout을 측정했다고 기술하지 않는다.

### M3. 속도 상한과 감시 기하

**Setup.** policy의 horizontal 요청을 **u = (uₓ, uᵧ)**, 요청 속력을 v = ‖u‖₂로 정의한다. governor는 명령 방향을 보존하고 크기만 감소시킨다. 실행 명령은 `u_exec = u · min(1, v_cap/max(v, ε))`다. principal arms의 adaptive-width, lateral-cap 및 yaw-cap 추가 기능은 꺼져 있었다. finite하고 slant range가 `0.995 × 12 m` 미만인 ray를 유효 return으로 인정하고 수평으로 투영했다. sensor association이 target-like로 분류한 return은 cap 계산에서 제외한다. 선택된 장애물 return이 없으면 clearance는 12 m다.

**Measurement.** 직선 회랑에서 명령 방향과의 상대 bearing을 δⱼ, projected range를 rⱼ라 하면 forward xⱼ = rⱼ cos δⱼ, lateral |yⱼ| = |rⱼ sin δⱼ|다. `xⱼ > 0`, `|yⱼ| ≤ w`인 ray의 최소 xⱼ를 d_straight로 사용한다. 기본 반폭 w는 0.45 m다. 세 arm은 다음과 같다.

| arm | clearance | 속도상한 |
|---|---|---|
| riskcap | d_straight | `2.0 + (3.5355339−2.0) · clip((d−3.0)/2.0, 0, 1)` m/s |
| stopcap | d_straight | `clip(√[(aτ)² + 2a·max(d−m,0)] − aτ, 0, 3.5355339)` |
| dwa_arc | d_arc | stopcap과 동일한 정지식 |

정지식의 파라미터는 a = 2.0 m/s², 반응시간 τ = 0.1 s, margin m = 0.45 m다. riskcap은 cap 2.0 m/s의 바닥을 유지하고, stopcap은 usable clearance가 0이면 cap도 0이 된다. 예를 들어 v = 2.5 m/s에서 `vτ + v²/(2a)`는 1.8125 m다. 이는 가정한 감속능력에 따른 운동학 계산이며 실기 제동 성능의 측정값은 아니다.

원호는 **요청 속력 v**와 **측정 body z 각속도 ω**에서 signed radius R = v/ω를 계산한다. 명령 방향을 기준으로 표현한 return (xⱼ, yⱼ)에 대해 `ρⱼ = |√(xⱼ² + (yⱼ−R)²) − |R||`, `ℓⱼ = min(2|R||δⱼ|, 12)`를 사용한다. `ρⱼ ≤ w`이고 `ℓⱼ > 0`인 ray의 최소 ℓⱼ를 d_arc로 선택한다. `|ω| < 10⁻³ rad/s`에서는 lateral = |yⱼ|, along = xⱼ로 직선 극한을 별도로 계산한다. 이 식은 측정에 사용한 프로그램의 정의이며 정확한 swept-volume first-intersection 공식이라는 주장이 아니다.

**Result.** arc−stopcap은 정지법칙을 고정한 상태의 감시 기하 대비다. 반면 arc−riskcap은 **법칙과 기하가 함께 바뀌므로 geometry만의 효과가 아니다**. 이를 분해하기 위한 A7에는 별도 riskcap_arc arm이 존재한다.

**Interpretation.** empirical treatment는 “원호”라는 추상적 원리 자체가 아니라 위 clearance 구현과 속도상한의 조합이다. 관측된 이득을 일반 DWA·CBF·RSS의 이론적 안전성으로 이전하지 않는다.

**Limitation.** no-return은 maximum clearance로 처리되고 잘못된 target association은 장애물 return을 제거할 수 있다. 보존 구현은 `(v, ω, w) = (2 m/s, 1 rad/s, 0.45 m)`와 전방 return (0.5, 0) m에서 튜브 오프셋이 0.06155 m임에도 along = 0을 제외해 clearance 12 m를 반환한다. 이 CPU 반례는 완전한 swept-tube 검출 보장을 반증한다. 다만 저장된 실험에서 이 상태의 발생 빈도 및 crash 기여율은 미측정이다.

### M4. 평가 지표와 종료 규칙

**Setup.** cell은 checkpoint, source version, evaluation seed, density, governor mode 및 반폭이 고정된 실행이다. episode의 terminal outcome은 capture, crash, timeout 중 하나다. capture는 target-relative 이전 위치와 현재 위치를 잇는 선분의 원점 최소거리가 0.5 m 미만인지 검사한다. 같은 step에서 capture와 crash 조건이 함께 발생하면 구현상 capture가 우선한다. crash에는 기체 contact, 높이 0.1 m 미만 또는 4 m 초과, arena XY bounds에 1 m margin을 더한 범위 밖의 종료가 포함된다. timeout은 capture나 crash 없이 600 actions에 도달한 종료다.

**Measurement.** primary metric은 `crash_rate = c_crash / n_actual`이다. mission trade-off를 평가하기 위해 `capture_rate = c_capture / n_actual`, `timeout_rate = c_timeout / n_actual`를 함께 계산했다. governor intervention rate는 요청 horizontal 속력보다 실행 명령 속력이 감소한 기록 frame의 비율이다. contact metric의 분모는 전체 crash가 아니라 기록된 bar-contact case다.

**Result.** principal dataset의 raw aggregate는 다음과 같다. 이 표는 분모를 합친 기술통계이며 IVW effect와 같지 않을 수 있다.

@@RAWTOT@@

primary set의 crash 원인을 합하면 arc의 2,330건은 bar contact 2,050, below 216, out-of-bounds 64건이다. riskcap의 2,806건은 각각 2,514/234/58건이고 stopcap의 2,767건은 2,507/208/52건이다. above는 모두 0건이다. 따라서 arc의 crash 감소를 모두 측면 막대 충돌 감소로 환원하지 않았다.

**Interpretation.** wide stopcap처럼 정지와 timeout이 늘면 crash가 낮아져도 mission은 나빠질 수 있다. 결과는 최소한 crash와 capture의 공동 방향 및 timeout을 함께 제시해야 한다.

**Limitation.** outcome은 해당 simulator의 termination 규칙에 의존한다. capture 우선순위, 표적 처리, 지오펜스 등을 바꾸면 같은 task 이름이라도 다른 metric이다. 시간·거리당 접촉 위험을 주장하려면 별도의 노출분모가 필요하다. aggregate CSV에는 episode의 scene/layout ID와 시간상 순서가 없어 cluster dependence와 paired covariance를 직접 계산할 수 없다.

### M5. 데이터 구성과 포함 기준

**Setup.** 패키지는 13개 result roots의 136 cell을 제공한다. 45 cell을 principal narrow-arm 분석에 사용했고, 보완 분석에는 width sweep, A7 law×geometry, A8/D4 readaptation, R-A drift probe 및 cross-tree repeat를 사용했다.

@@ROOTROWS@@

**Measurement.** principal inclusion에는 root, checkpoint identity, seed, density, mode, width와 추가 기능의 활성화 여부를 사용했다. seed 523의 narrow arm은 D1′, 527/531은 각 R-B root에서 선택했다. 다른 root에 저장된 baseline 또는 재실행을 새 독립 trial로 추가하지 않았다. 전체 136행의 episode 분모 합 278,728에는 이러한 반복이 있어 unique trial 수로 해석하지 않는다. 폭 분석에서 seed 523은 70/205 bars에만 1.2 m reference가 있는 반면 새 seed의 1.2 m arm은 5개 density에 모두 존재한다. 따라서 3-seed width pooling은 공통으로 관측된 70/205 bars에 한정했다.

**Result.** 로컬 raw JSON 136개 및 link된 contact JSONL 114개, 19,298행의 count와 hash 검사가 일치했다. roots manifest는 10개만 포함해 A7의 3개 root가 누락되어 있었다. 이는 count 불일치는 아니지만 portable provenance 패키지의 불완전성이다.

**Interpretation.** narrow main effect, width common-domain effect, contact case-only dataset을 구분하면 분석마다 seed와 n이 달라지는 이유를 재현할 수 있다.

**Limitation.** 제공된 `verify.py`는 adaptive gain을 키에서 제외한 first-root-wins 방식을 사용하여 L8 adaptive arm과 fixed narrow arm을 충돌시킨다. 원래 파일 순서에서는 baseline이 먼저 나와 주효과가 맞지만, 행 순서를 뒤집으면 −1.5552 pp로 바뀐다. 본 분석은 해당 선택기를 사용하지 않았다. 별도 held-out 결과 6건의 유실 checkpoint 4개는 복구하거나 재실행하지 않았으며 principal 45 cell에는 포함되지 않는다.

### M6. 통계 방법

**Setup.** 효과는 risk difference를 percentage points(pp)로 보고한다. cell별 count를 실제 n으로 나누며 반올림 비율을 입력으로 쓰지 않았다. principal effect는 arc−riskcap의 3 seed × 5 density, 15개 대비다. arc−stopcap과 stopcap−riskcap도 별도로 계산했다.

**Measurement.** 차이는 `Δ = 100(p̂_A−p̂_B)`, 분산은 `V = 10⁴[p̂_A(1−p̂_A)/n_A + p̂_B(1−p̂_B)/n_B]`인 독립 이항 Wald 근사를 사용했다. inverse-variance pooled mean은 `Σ(Δ/V)/Σ(1/V)`, 표준오차는 `(Σ1/V)^−1/2`다. CI는 `mean ± 1.959964 SE`이며 양측 normal p를 보고했다. Cochran Q와 I²를 병기했다. 각 arm 대비별 5개 density family에서 Holm 보정을 적용했다. 표의 CI는 보정 전 95% 구간이다. 이 family 정의는 감사의 명시적인 분석 선택이며 원 사전등록에 완전히 고정됐었다고 주장하지 않는다. 세 arm 대비는 표본을 공유하므로 그 p를 독립 증거로 결합하지 않았다.

**Result.** principal arc−riskcap에서 density 70/100/130/160/205의 가중치는 26.29/25.14/21.52/16.91/10.15%였다. 균등 density target을 확인하기 위해 equal-cell weighting, 기존 seed 제외, 각 seed의 동일 density 평균에 대한 t(df = 2) 구간을 민감도 분석으로 추가했다.

**Interpretation.** fixed-effect 결과는 동결된 조건의 정밀도 가중평균이다. 미래의 새 policy, training seed, 미관측 density를 포함하는 population effect를 직접 추정하지 않는다. I² = 0은 seed variation이 없다는 증명이 아니다. [Cochrane Handbook, Chapter 10](https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-10).

**Limitation.** 서로 다른 episode라도 layout pool과 RNG stream을 공유할 수 있고 arm 간 covariance도 미상이다. Wald 구간은 독립성을 가정한 조건부 구간이다. 세 seed의 t 구간은 소표본 정규근사 민감도 분석이며 cluster-robust bootstrap 구간이라고 부르지 않는다. equivalence는 사전 정의한 ±δ에 대해 별도 검정해야 하고 CI가 0을 포함하는 것으로 결론내리지 않는다. [Lakens, 2017](https://doi.org/10.1177/1948550617697177).

## Experiments

### E1. Frozen policy의 narrow-arm 비교

**Setup.** 반폭 0.45 m의 riskcap, stopcap, dwa_arc를 평가 seed 523/527/531 및 density 70/100/130/160/205에서 비교했다. policy는 하나이고 재학습은 없었다. 총 45 cell, 92,227 episode이며 각 pairwise pooled estimate는 15개 seed×density difference를 사용했다.

**Measurement.** primary outcome은 crash rate다. 아래 표는 그 계산에 사용한 cell별 count와 실제 분모를 나타낸다. 같은 표본의 capture, timeout 및 frame-level intervention을 secondary metric으로 평가했다.

@@COUNTS@@

**Result.** 결합 추정치는 다음과 같다.

@@HEADLINE@@

arc의 crash율과 intervention rate는 모두 15/15 seed×density 조합에서 두 narrow 직선 arm보다 낮았다. 전체 기록 frame 가중 intervention rate는 arc 13.0927%, riskcap 16.8183%, stopcap 30.0083%였으며 각 분모는 4,118,784/4,085,632/5,348,096 frames다. 이는 노출량에 가중된 기술통계다. frame을 독립 표본으로 취급한 p는 계산하지 않았다.

@@SEEDTABLE@@

**Interpretation.** 특정 frozen policy에서 현재 구현된 arc+stoplaw는 평가 seed 반복 후에도 낮은 crash를 보였다. raw aggregate capture 또한 riskcap보다 1.5516 pp 높았다. arc−stopcap은 stoplaw를 고정한 geometry comparison이지만, arc−riskcap은 geometry만의 효과로 분해하지 않았다.

**Limitation.** 모든 density에서 통계적으로 두 baseline을 이긴다는 주장은 지지하지 않는다. 아래 표에서 arc−riskcap은 130 bars의 CI가 0을 포함하고, arc−stopcap은 70 bars의 CI가 0을 포함한다. Holm 보정 후 arc−stopcap의 130 bars도 p = .0581이다. 또한 별도 A7의 ep25000, seed 49, 205 bars에서는 arc−stopcap이 +0.439 pp [−1.832, +2.711]로 반대 부호다. 따라서 모든 기존 seed와 계보를 포괄하는 보편 우위라고 쓰지 않는다.

@@DENSITY@@

### E2. 평균 효과의 민감도와 P5

**Setup.** 기존 탐색 seed 523에 대한 의존성을 검사하기 위해 새 seed 527/531만의 10개 대비를 결합했다. 추론의 반복 단위를 episode에서 evaluation seed로 바꾸는 민감도 분석도 수행했다. P5에 대해서는 stopcap−riskcap을 density별로 분석했다.

**Measurement.** 새 seed만의 IVW, equal-cell weighting, 각 seed의 동일 density 평균에 대한 t(df = 2) 구간을 사용했다. P5에서는 70 bars의 seed별 차이와 5-density Holm p를 보고했다.

**Result.** 새 seed만의 arc−riskcap은 −1.6669 [−2.1688, −1.1649] pp였다. equal-cell weighting은 −1.5489 [−1.9820, −1.1158] pp, seed-level t 민감도 구간은 −1.5489 [−2.6482, −0.4496] pp, p = .0261이었다. 70 bars의 stopcap−riskcap은 seed별로 −0.3388, −1.6577, −1.6105 pp였고 IVW 결합은 −1.1904 [−2.0099, −0.3710] pp, Holm p = .02205였다.

@@P5SEED@@

**Interpretation.** 원호 평균 이득은 새 seed와 가중법 변경 후에도 남았다. P5의 “모든 density CI가 0을 포함한다”는 관측 예측은 실패했다. 70 bars의 음의 차이는 독립 episode 모형에서 영효과에 반하는 증거다. 그러나 세 seed 동일 부호만의 양측 sign-test는 p = .25이고, 해당 density의 seed-level t 구간은 −1.2024 [−3.0610, +0.6563] pp, p = .1085다. 따라서 국소 차이에 대한 noise를 배제했다고 해석하지 않았다.

**Limitation.** 100 bars 이상에서 CI가 0을 포함한다는 결과는 추가 이득을 검출하지 못했음을 뜻하며 equivalence나 sharp density breakpoint를 증명하지 않는다. density별 effect variation의 Q_between = 9.8548(df = 4), p = .04295는 탐색적 단서다. 70−100 직접 차이는 −1.0206 [−2.2005, +0.1593] pp로 0을 포함한다. 원래 15개 cell별 p를 한 family에서 Holm 보정하면 0/15개가 통과한다.

### E3. Width × geometry의 mission trade-off

**Setup.** frozen ep25000 policy에서 반폭 0.45→1.2 m를 arc와 straight stopcap에 적용했다. 공통 데이터가 있는 70/205 bars 및 3개 평가 seed를 사용했다. seed 523의 추가 탐색폭은 arc의 경우 0.45/0.6/0.8/1.0/1.2/1.6/2.0/3.0 m, 직선은 0.45/0.6/0.8/1.0/1.2 m였다. 따라서 두 geometry 모두 8점인 완전요인설계가 아니다.

**Measurement.** 각 seed에서 wide−narrow의 crash, capture, timeout 차이를 계산한 뒤 metric별로 IVW 결합했다. 동일 탐색 데이터에서 선택한 폭의 최적성을 사후 입증하는 문제를 피하기 위해 3-seed 재현은 미리 선택한 1.2 m 대 0.45 m 대비로 한정했다.

@@WIDTH@@

**Result.** 205 bars에서 arc 확대는 crash −5.5985 pp, capture +4.4411 pp, timeout +1.1430 pp였다. 같은 확대의 straight stopcap은 crash −13.2693 pp, capture −59.9837 pp, timeout +73.2649 pp였다. 70 bars의 arc crash 감소는 seed 523/527/531에서 −2.7330/−1.3171/−2.4390 pp였다. seed 527은 사전 크기 문턱 −2 pp를 충족하지 못했으므로 P3는 6개 seed×density 중 5개만 통과했다.

**Interpretation.** 원호의 넓은 튜브는 공통 검사 density에서 crash와 capture를 함께 개선했다. 직선 stopcap의 넓은 회랑은 crash 감소를 대규모 timeout 증가와 교환하여 mission을 악화시켰다. “폭의 효과가 geometry에 따라 반대”라는 표현은 capture에 적용하며, crash는 두 geometry 모두 감소했음을 명시한다.

**Limitation.** 세 seed의 capture 점추정이 모두 비감소라는 사실은 각 seed별 noninferiority의 통계적 입증이 아니다. 예를 들어 seed 527, 70 bars의 arc Δcapture는 +0.5854 [−0.8957, +2.0665] pp다. 1.2 m를 모든 환경의 최적폭으로 주장하지 않으며 1.6 m와의 practical equivalence도 검증하지 않았다. 각 metric의 IVW 가중치가 달라 세 pooled 차이의 합이 정확히 0일 필요는 없지만 cell별 outcome 비율의 합은 1이다.

### E4. 접촉 전후 기하와 협착 가설

**Setup.** seed 527/531의 R-B 50 cell에서 bar contact 5,890건의 case-level JSONL을 사용했다. 모든 행에 접촉 시점의 pinch flag가 있었다. 접촉 1 s 전의 명령축에 대한 충돌 막대 중심의 측면 오프셋, 당시 yaw-rate와 actual/requested direction angle, 접촉 시점 좌우 표면거리 근사를 분석했다.

**Measurement.** t−1 s 오프셋은 `hit_lateral_cmd`다. 시점 t의 g_L, g_R는 각각 body-frame bearing [15°,165°], [−165°,−15°]에 있는 막대 중심거리에서 XY 외접원 반경을 뺀 최솟값이며 빈 sector는 12 m다. `pinch_t0 = 1[g_L<0.65 m ∧ g_R<0.65 m]`로 정의했다. 오프셋은 중앙값/IQR, 협착은 count/contact count로 보고했다.

@@PINCH@@

**Result.** 전체 접촉 5,890건 중 97건(1.647%)이 이 pinch 정의에 해당했다. narrow riskcap은 30/1,686(1.779%), narrow stopcap은 15/1,610(0.932%), narrow arc는 21/1,333(1.575%), wide arc는 31/763(4.063%)이었다. pooled narrow arc의 t−1 s 오프셋 중앙값은 0.6911 m [IQR 0.3613,1.0967]였다.

**Interpretation.** 이 정의의 동시 양측 근접은 관측된 bar contact의 주된 분류가 아니었다. 상당한 측면 오프셋을 가진 접촉은 반폭 0.45 m의 직선 감시가 모든 접촉을 포착하지 못할 수 있다는 진단과 양립한다.

**Limitation.** contact를 조건으로 선택한 자료이며 non-contact risk set이나 동적인 reachable free-space를 검사하지 않았다. g_L/g_R는 통로 폭이 아니라 외접원까지의 radial gap이다. 전후 sector의 장애물, 회전 중 swept body, 추력·제동 한계에 따른 회피불가능성을 배제하지 못한다. 따라서 “협착이 아니라 경로 이탈이다” 또는 “접촉 순간 통과 가능한 경로가 있었다”는 배타적·반사실 주장을 하지 않는다. 전체 114개 contact 파일의 cell 중앙값 범위 0.28225–0.88530 m는 “모든 조건에서 0.57–0.76 m”라는 기존 범위를 반증한다.

### E5. 재적응 조건과 필터 법칙의 상호작용

**Setup.** D4는 공통 ref5in D1 ep1900 checkpoint에서 1,000 epoch를 재적응한 T0(off 학습)와 T1(riskcap 학습)을 비교했다. 출발 checkpoint의 SHA-256은 `197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e`다. 재적응 seed 197, 70 bars, goal 22.5–28 m, learning rate 1.5×10⁻⁵를 사용했다. 평가 seed 521에서 각 정책을 riskcap/stopcap으로 평가한 4 cell, 총 8,197 episode다.

**Measurement.** T0의 crash count는 riskcap 319/2049, stopcap 240/2049이고 T1은 343/2049, 353/2050이다. 학습조건×평가필터 상호작용을 `I=(p_T1,stop−p_T1,risk)−(p_T0,stop−p_T0,risk)`로 정의했다. 각 정책 안의 유의/비유의 판정을 비교하는 대신 네 비율의 difference-in-differences를 직접 검정했다.

**Result.** T0의 stopcap−riskcap은 **@@D4T0@@ pp**, T1은 **@@D4T1@@ pp**였다. 상호작용은 **+4.3352 [+1.2227,+7.4476] pp**, p = .006334였다.

**Interpretation.** 이 한 재적응 쌍에서 평가 법칙의 추가 crash 효과가 학습조건에 따라 달랐다는 상호작용을 지지한다. T1의 CI가 0을 포함하는 것은 두 법칙의 동등성 입증이 아니다. R-B의 70 bars 국소 효과는 다른 정책 및 기체 계약에서 측정했으므로 D4의 density interaction을 대체하지 않는다.

**Limitation.** 재적응 seed와 initial checkpoint는 각각 1개다. 예정된 seed 233/239 실험은 같은 checkpoint에서 추가 재적응 seed를 반복한다. 완료되더라도 전체 사전학습의 독립 seed 3개가 되지 않는다. A8의 T1/riskcap 358/2050과 D4의 343/2049 차이는 원인 미상이며, 후속 R-A 두 번이 343/2049였다는 이유만으로 원인이 해결되었다고 주장하지 않는다.

### E6. 소스 재현성 및 전체 한계

**Setup.** seed 523은 old runtime tree, 527/531은 new runtime tree에서 생성되었다. runtime 파일 340개 중 5개가 달랐고 policy와 robot asset은 동일했다. 기본 gain이 0인 width 경로는 같은 scalar 반폭을 반환한다. 접촉 시점 추가 기록과 비활성 adaptive 설정이 주요 변경이었다.

**Measurement.** 보존된 source bundle마다 선언된 340개 파일의 해시를 snapshot 바이트와 대조했다. seed 523, 205 bars, arc 0.45 m의 한 cross-tree repeat에서 outcome aggregate를 비교하였다. 집계 일치와 episode/trajectory/state-level bit identity를 구분했다.

**Result.** 검사한 5개 bundle의 파일 해시는 모두 일치했다. 한 cell의 outcome은 crash/capture/timeout = 306/1699/44, n = 2049로 일치했다. 기록된 변경의 비활성 조건과 이 결과는 기본 조건의 호환성을 지지하며, new tree의 두 seed만으로도 primary effect가 음수였다.

**Interpretation.** 저장된 수치의 산술 재현성과 두 새 seed에서의 반복은 지지된다. 그러나 source version과 평가 seed가 완전히 교차한 설계가 아니므로 seed와 tree 효과를 완전히 분리하지 못한다. 결론은 frozen algorithm, checkpoint, task contract에 조건부인 simulation evidence로 제시한다.

**Limitation.** NPS detector의 validation AP 0.579나 인접 epoch의 AP 평균 0.549±0.015를 본 연구의 독립 generalization metric으로 연결하지 않는다. 인접 epoch의 SD는 독립 test/seed CI가 아니며 실제 거리오차는 range GT 없이 식별할 수 없다. 별도 checkpoint 4개 유실과 그에 의존한 held-out 6개 결과의 재실행 불가 한계도 남아 있다. 현 자료는 실기 안전성, 새 학습 계보에서의 보편 우위, adaptive width의 불필요성, 정확한 swept-tube 검출 보장을 지지하지 않는다.

## 제출 전 반영 원칙

본 문장의 결과는 보존 자료의 재분석이다. source migration이나 수정 원호의 성능을 아직 실험한 것처럼 삽입하지 않는다. 각 claim의 policy, 기체, density, evaluation seed, adaptation seed 범위를 유지한다. episode-cluster 분석을 추가하면 해당 추정량과 구간을 별도로 명시한다. principal narrow dataset, width trade-off, case-only contact를 하나의 n으로 합치지 않는다.
