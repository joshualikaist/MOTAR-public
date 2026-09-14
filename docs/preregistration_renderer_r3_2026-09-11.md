# R3 독립 renderer validation 사전등록

2026-09-11. **실행 전 작성·커밋한다.** R2 seed 0 smoke는 이미 관측했으므로 이 판정에서 제외한다.
정식 실행 seed는 173이며 변경하지 않는다. 학습 모델·detector·task·policy를 사용하지 않는다.

## 고정 표본과 실행

- 커밋된 tracked-clean source에서 RTX 3070, `aerialgym` 환경, Warp GPU로 실행한다.
- 일반 상자 2개 fixture, camera 160×120, HFOV 60°, far ray range 20m, scene 1개다.
- 같은 커밋·seed 173을 **서로 다른 Python 프로세스 2개**에서 실행한다.
- 각 프로세스는 같은 geometry를 2회 ray cast한다. 모든 geometry buffer가 exact 일치해야 한다.
- 두 실행의 모든 metric·check·array SHA와 지정 runtime 축이 exact 일치해야 최종 PASS다.
- 결과를 본 뒤 threshold, seed, fixture, resolution, light/material arm을 바꾸지 않는다.

## 고정 arms와 판정

| 검사 | arm | 성공 기준 |
|---|---|---|
| fixture | GPU G-buffer | coverage ≥ 0.05, instance 정확히 2개, face ≥ 4개 |
| flat | 모든 material의 linear RGB=0.55 | 유효 픽셀 luminance population variance ≤ `1e-12` |
| normal shading | ambient=.2, kd=.8, directional=.8, light=(-1,0,-1) 정규화 | luminance variance ≥ `1e-4` |
| lighting | 위 arm과 light=(1,0,-1)만 다른 arm | 유효 RGB MAE ≥ `0.02` |
| material | flat arm에서 material 0만 RGB=(.8,.2,.2) | 해당 material visible pixel ≥50, 전부 변화, 그 밖은 max change=0 exact |
| depth/geometry | 같은 입력의 두 번째 ray cast | range/depth/normal/face/instance/valid 전부 exact |
| debug leakage | 유효 instance ID에 +100, shading 재계산 | Lambertian RGB exact, 모든 RGB output의 마지막 차원=3 |

linear-light luminance는 BT.709 가중치 `(0.2126, 0.7152, 0.0722)`를 쓴다.
하나라도 실패하면 해당 run은 FAIL이고, 두 run 중 하나가 실패하거나 exact 비교가 실패하면 최종 FAIL이다.
실패 원인을 본 뒤 본 판정의 데이터를 다시 만들지 않는다. 후속 수정은 새 버전·새 사전등록으로 분리한다.

## 해석 제한

PASS는 제한된 fixture에서 flat/Lambertian 구현과 출력 격리·재현 계약이 작동했다는 뜻뿐이다.
photorealism, 실제 영상 전이, perception shortcut 감소, detector/association 성능 향상을 뜻하지 않는다.
FAIL도 학습 시스템의 실패를 뜻하지 않는다. R4 background와 R5 throughput은 이 판정에 포함하지 않는다.

## 실행 명령

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/fair/miniconda3/envs/aerialgym/bin/python -B tools/eval_renderer_r3.py \
  --output results/renderer_r3_seed173_run1_2026-09-11

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/fair/miniconda3/envs/aerialgym/bin/python -B tools/eval_renderer_r3.py \
  --output results/renderer_r3_seed173_run2_2026-09-11

python3 -B tools/compare_renderer_r3.py \
  --run results/renderer_r3_seed173_run1_2026-09-11/run.json \
  --run results/renderer_r3_seed173_run2_2026-09-11/run.json \
  --output results/renderer_r3_seed173_comparison_2026-09-11
```
