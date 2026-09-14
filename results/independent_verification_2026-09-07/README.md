# NavRL speed-filter independent verification (2026-09-07)

이 디렉터리는 `results/verification_export_2026-09-07/`의 raw count와 manifest를 독립적으로 재계산하고, 관련 계획·주장의 강도를 감사한 최종 산출물이다.

## 먼저 읽을 문서

- [`AUDIT_REPORT.md`](AUDIT_REPORT.md): 오류, 근거의 한계, 필요한 수정, 후속 실험 시간 추정을 포함한 최종 감사 보고서
- [`PAPER_METHOD_EXPERIMENT.md`](PAPER_METHOD_EXPERIMENT.md): 논문 본문에 옮길 수 있는 Method/Experiment 초안
- [`tables.md`](tables.md): 재계산된 핵심 수치 표

## 핵심 판정

- 45개 사전 지정 cell을 inverse-variance fixed-effect 방식으로 결합하면 `arc − riskcap = −1.4903 percentage points`, 95% CI `[−1.8981, −1.0826]`가 재현된다.
- 70 bars에서 `stopcap − riskcap = −1.1904 pp`이며 episode-level raw p-value는 `0.00441`, 5개 density에 대한 Holm 보정 p-value는 `0.02205`이다. 그러나 seed를 독립 반복 단위로 취급한 t 검정은 3 seed만으로 95% CI가 0을 포함하고(`p=0.108`), 세 seed가 모두 음수라는 부호검정도 양측 `p=0.25`이다. 따라서 평가 episode 독립성을 전제로 한 신호는 있으나 seed 일반화와 noise 배제 주장은 성립하지 않는다.
- 두 source tree의 기본 제어 경로가 같다는 정적 근거와 한 cell의 outcome-count 재현은 pooling의 보조 근거다. 이것만으로 bit-identical 실행이나 전체 45-cell 교환가능성을 입증하지는 않는다. 다만 새 tree의 두 seed만으로도 주효과는 `−1.6669 pp`, 95% CI `[−2.1689, −1.1649]`이다.
- archived 구현에는 실제 swept tube 안의 return을 제외하는 반례가 존재한다. 현재 알고리즘은 `implemented arc-clearance speed filter`로 기술해야 하며, 완전한 DWA 또는 충돌안전 보장으로 표현하면 안 된다.

## 재현

저장소 루트에서 다음을 실행한다.

```bash
python3 results/independent_verification_2026-09-07/recompute.py
python3 results/independent_verification_2026-09-07/evidence_audit.py
/home/fair/miniconda3/envs/aerialgym/bin/python \
  results/independent_verification_2026-09-07/check_numerics_and_witness.py
```

첫 두 스크립트의 핵심 집계는 Python 표준 라이브러리만 사용한다. 세 번째 스크립트는 SciPy 수치 교차검증과 archived PyTorch 구현의 기하 반례 실행에 사용한다.

기계 판독 결과는 [`recomputed.json`](recomputed.json), [`evidence_audit.json`](evidence_audit.json), [`primary_cells.csv`](primary_cells.csv)에 저장되어 있다.
