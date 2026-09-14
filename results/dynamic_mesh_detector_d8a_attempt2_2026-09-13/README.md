# D8-A — mesh-derived observation technical gate (attempt 2)

사전등록: [`docs/preregistration_dynamic_mesh_detector_d8_2026-09-13.md`](../../docs/preregistration_dynamic_mesh_detector_d8_2026-09-13.md),
commit `e538052`. 실행 소스: clean commit
`c50a26d86f820611da879a84e3a252c787bd3dd9`. 원자료와 모든 runtime/source hash는
[`receipt.json`](receipt.json)에 있다.

## 판정: **`TECHNICAL_GO`**

D8-A가 묻는 좁은 질문에는 GO다. v3 visual mesh를 target-local Warp mesh로 한 번 만들고,
움직이는 표적의 ray를 local frame으로 변환해 기존 private target mask/depth에 연결할 수 있다.
정적 장면 occlusion, 결정성, debug buffer 비유출, 비용 gate가 모두 통과했다.

이 판정은 detector 정확도, association, 색 지름길 감소, frozen-policy 견고성, adaptation,
실기 성능을 측정하지 않았다. 이 항목들을 GO로 읽으면 안 된다.

## 사전등록 비용 gate

128 env × 160×90, warm-up 50, measured 500, arm마다 새 process 3회. 표는 세 반복의
step-median 중앙값이다.

| arm | step median | steps/s | baseline 대비 | throughput 손실 | torch reserved 증가 | NVML 증가 |
|---|---:|---:|---:|---:|---:|---:|
| `analytic_flat` | 43.656 ms | 20.825 | — | — | — | — |
| `mesh_flat` | 43.911 ms | 20.672 | **+0.256 ms / +0.585%** | 0.734% | +44 MiB | +44 MiB |
| `mesh_shaded` | 44.173 ms | 20.526 | **+0.517 ms / +1.184%** | 1.436% | +66 MiB | +64 MiB |

두 treatment 모두 사전등록 GO 경계(상대 ≤10%, 절대 ≤5 ms, throughput 손실 ≤10%, torch
reserved 증가 ≤256 MiB)를 만족했다. NVML도 모든 셀에서 사용 가능했다. latency의 음수나 작은
P95 역전을 성능 개선으로 해석하지 않는다.

## 무결성과 기하 효과

21개 visible pose(거리 4.5/5.5/6.5 m × 자세 7개)와 별도 bar-occluded pose를 두 번 실행했다.

- 두 pose run의 전체 output hash가 같다.
- `mesh_flat`과 `mesh_shaded` mask/depth가 같다.
- 세 거리 band 모두 mesh pixel이 0이 아니며 analytic OBB와 다른 기하 효과가 있다.
- visible pose의 `mesh / analytic` 픽셀 면적비 중앙값은 **0.556**, 범위 0.429–0.889다.
- 거리별 면적비 중앙값은 4.5 m 0.571, 5.5 m 0.583, 6.5 m 0.500이다.
- flat target 내부 red variance는 0이고 shaded arm은 0보다 크다.
- target-local raw hit 164개 중 12개가 정적 bar에 가려졌고, published occluded survivor는 0개다.
- visible hit 152개에서 invalid depth/face/material/normal은 모두 0개다.
- 세 arm의 fixed-action robot/target trajectory hash가 arm과 반복에 걸쳐 같다.
- 각 arm의 500-step output hash는 세 독립 process에서 반복된다.

전체 step에서 누적 target pixel은 analytic 726,178, mesh-flat/mesh-shaded 409,825로 반복마다
정확히 같았다. 이는 visual mesh가 analytic OBB보다 작은 silhouette를 만든다는 측정이지, 더
정확한 detector나 더 좋은 policy를 뜻하지 않는다.

## 실행 환경과 attempt 1 경계

Python `/home/fair/miniconda3/envs/aerialgym/bin/python3.8`, torch 2.4.1+cu121, Warp 1.0.0,
RTX 3070을 사용했다. Isaac Gym extension용 ninja는 같은 conda prefix의 executable을 직접
바인딩했고, version은 `1.13.0.git.kitware.jobserver-pipe-1`, SHA-256은
`696f9628a79d9ce50314cf9556d7cd1a1d1ec52b8fd52828f6f9db1719565b67`이다.

attempt 1은 이 바인딩 전에 task import에서 멈춘 0-cell `VOID_EXECUTION`이며 별도 디렉터리에
보존한다. attempt 2와 합치지 않는다.

## 재현

```bash
cd /home/fair/workspaces/aerial_gym_ws/src/aerial_gym_simulator
PATH=/home/fair/miniconda3/envs/aerialgym/bin:$PATH \
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
/home/fair/miniconda3/envs/aerialgym/bin/python -B \
  tools/probe_dynamic_mesh_treatment.py \
  --output results/<new-empty-d8a-directory>
```

기존 결과 디렉터리를 재사용하지 않는다. 이 결과 뒤의 다음 gate는 별도 사전등록된 D8-B
frozen-policy sensitivity다. 학습은 여전히 승인되지 않았다.
