# R3 독립 renderer validation — PASS

사전등록: [preregistration_renderer_r3_2026-09-11.md](../../docs/preregistration_renderer_r3_2026-09-11.md).
실행 소스: 커밋 `83c62c30f132d740515c3ab0ea99d6837299e294`, tracked tree clean.
2026-09-11 재검증에서 README의 전체 SHA 오기를 receipt와 대조해 교정했다. 원본 receipt는 변경하지 않았다.
환경: RTX 3070, Python 3.8.20, torch 2.4.1+cu121, Warp 1.0.0.

## 최종 판정

**`PASS`**. seed 173을 두 별도 Python 프로세스에서 실행했다. 두 run의 metric, check,
11개 output-array SHA-256과 사전등록 runtime 축이 전부 exact 일치했다.

| 측정값 | 결과 | 고정 기준 |
|---|---:|---:|
| 유효 geometry coverage | 0.21344 | ≥ 0.05 |
| visible instance / face | 2 / 8 | =2 / ≥4 |
| uniform flat luminance variance | 0.0 | ≤ 1e-12 |
| Lambertian luminance variance | 0.018891 | ≥ 1e-4 |
| light-direction 변경 RGB MAE | 0.201582 | ≥ 0.02 |
| 선택 material visible pixels | 2,196 | ≥ 50 |
| 선택 material 변경 비율 | 1.0 | =1.0 |
| 선택 외 영역 최대 변화 | 0.0 | =0.0 |
| depth range | 2.8204–4.1569 m | 기술 기록 |

동일 입력 재렌더의 range/depth/normal/face/instance/valid는 전부 exact 일치했고,
instance ID를 +100으로 재번호화해도 Lambertian RGB는 exact 일치했다. 모든 RGB 출력은 3채널이다.

## 해석 제한

PASS는 이 일반 상자 fixture와 고정 조건에서 flat/Lambertian 구현, geometry 보존, debug-ID 격리와
프로세스 간 재현이 확인됐다는 뜻뿐이다. photorealism, 실영상 전이, detector/association 성능,
shortcut 감소, autonomous task 개선을 뜻하지 않는다. 학습 모델·label·loss가 없으므로 학습은
실행하지 않았고 이 결과에 학습 판정은 없다. 이 R3 실행 자체에는 R4 background와 R5 throughput이 포함되지 않는다.
후속 R4/R4b FAIL과 R5 계측은 [전체 재검증 보고서](../renderer_reverification_2026-09-11_1951/README.md)에 있다.

근거는 [run 1](../renderer_r3_seed173_run1_2026-09-11/run.json),
[run 2](../renderer_r3_seed173_run2_2026-09-11/run.json), [exact 비교](summary.json)에 있다.
