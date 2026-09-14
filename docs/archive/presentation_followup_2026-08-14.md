# 발표 전 추가 확인 4건 — 2026-08-14

## 1. Detector coupling gate-passing rerun

재사전등록: `docs/prereg_2026-08-14_detector_coupling_binbias.md`.

거리 quartile별 v7 mean bias `+0.205/+0.305/+0.224/−0.554 m`를 기존 AR(1)+이분산 주입에
추가했다. 결과를 보기 전에 evaluation seed 431, detector-noise seed 9431, 3 arms, ±10% 품질
게이트와 판정 규칙을 고정했다.

### 품질 게이트

| 항목 | v7 profile | bin-bias injection | 차이 | 판정 |
|---|---:|---:|---:|---|
| pooled range-error std | 0.7053 m | 0.7085 m | +0.45% | PASS (허용 ±10%) |

### 3-arm 결과

| arm | episodes | capture | crash | timeout | Δ capture vs clean (95% CI) |
|---|---:|---:|---:|---:|---:|
| analytic clean | 2,050 | 80.29% | 17.22% | 2.49% | baseline |
| analytic + v7 noise 1.0× + bin bias | 2,049 | 76.72% | 21.03% | 2.24% | −3.57 pp [−6.09, −1.06] |
| learned v7 real output | 2,049 | 76.04% | 21.43% | 2.54% | −4.26 pp [−6.78, −1.73] |

고정 규칙에 따라 **output-coupling hypothesis supported; matched retraining required**다. 합성
arm CI가 v7 실물 점추정 −4.26 pp를 포함한다. 합성 출력오차가 이 frozen analytic-trained policy를
해친다는 것은 확인했지만, 같은 출력분포로 재학습한 policy가 견디는지는 측정하지 않았다. 따라서
발표의 짧은 표기는 `frozen-policy 출력분포 결합 지지`가 적절하고, `인과적 결합 확정`은 과도하다.

출처:

- `results/navrl_detector_coupling_binbias_seed431/quality_gate.json`
- `results/navrl_detector_coupling_binbias_seed431/summary.md`
- 각 arm의 `205bars.json`과 `205bars.receipt.json`

## 2. Bar footprint와 nominal gross area occupancy

생성 계약은 width와 depth를 **각각 독립적으로** `Uniform(0.4, 0.8) m`에서 뽑는다. 사각형이며
height는 v2 `bars_h3` pool에서 3.0 m다. 이론값:

- 한 변 평균 0.6000 m, 표준편차 0.11547 m
- 단면적 `E[W×D] = 0.3600 m²`, 표준편차 0.09888 m²
- 205 bars / 40×40 m arena의 gross occupancy = `205×0.36/1600 = 4.6125%`

실제 seed-42 40개 URDF pool의 전수 실사값:

- pooled side 평균 0.60262 m, population std 0.10806 m
- 단면적 평균 0.365313 m², population std 0.101583 m²
- pool 평균 기준 gross occupancy = `205×0.365313/1600 = 4.6806%`

runtime은 이 40개 파일을 동일확률·복원추출한다. 발표에는 생성분포의 이론값 **약 4.61%**를
주값으로 쓰고, 필요하면 실제 finite pool 값 4.68%를 괄호에 둔다.

주의: navrl_band는 touching/merge 배치를 허용하므로 위 값은 각 footprint를 합한 **nominal gross
occupancy**다. 실제 layout union area는 overlap만큼 작고 layout별로 달라 별도 geometry dump 없이
고정값으로 말할 수 없다.

YOPOv2 0.6 m 원형 나무의 nominal gross occupancy 1.13%(4/100m²)–1.77%(6.25/100m²)와 같은
방식으로 비교하면 MOTAR는 면적 기준 약 **2.6–4.1배**다.

출처:

- `tools/generate_bar_assets.py` (`MIN_WD=0.4`, `MAX_WD=0.8`, independent draws, seed 42)
- `resources/models/environment_assets/bars_h3/bar_*.urdf` (실제 40개 pool)
- `aerial_gym/env_manager/asset_loader.py` (`random.choices`, equal-weight replacement)
- `aerial_gym/config/env_config/navrl_bars_env.py` (40×40 m runtime arena contract)

## 3. RA-L 2025 exception bibliography

문헌 스캔의 “장애물 속 요격은 다중기체 1편”은 아래 OPEN 논문이다.

Jiayu Chen, Chao Yu, Guosheng Li, Wenhao Tang, Shilong Ji, Xinyi Yang, Botian Xu, Huazhong Yang,
and Yu Wang, **“Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep
Reinforcement Learning,”** *IEEE Robotics and Automation Letters*, vol. 10, no. 8, pp. 8196–8203,
2025. DOI: **10.1109/LRA.2025.3583620**.

이는 multi-UAV cooperative pursuit-evasion이며 unknown obstacle environments와 real quadrotor
deployment를 다룬다. 따라서 `선행연구는 정지 목표점 중심`은 절대명제가 아니라
`고밀도 sensor-only 단일기체 이동표적 요격의 직접 비교군은 제한적이며, 예외로 OPEN의 다중기체
pursuit-evasion이 있다`로 쓰는 것이 안전하다.

로컬 스캔 출처: `docs/development_directions_2026-08.md` lines 18, 61.

## 4. Verification 5B status

**2026-08-14 01:34 KST 기준 NOT RUN / PENDING이 맞다.** 실행 중인 training/evaluation process가
없고, `results/`에 5B full-budget summary/receipt/seed lineage가 없다. 존재하는 최신 단계는 5A
engineering smoke와 ref5in P2 strict-fail diagnostics다. 따라서 발표에서 `5B: NOT RUN / PENDING`
표기를 유지한다. 중간 수치는 없다.

출처:

- `docs/CLAUDE_PPT_REVIEW_REQUEST_VERIFICATION5B_2026-08-13.md`
- `results/navrl_v2_v5a_semantics_smoke_seed197/summary.md`
- `results/navrl_ref5in_p2_seed313/summary.md`
