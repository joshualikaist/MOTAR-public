# Claude handoff — streaming 재현성 및 다음 단계

> **해소됨 (2026-09-09).** §1 재현성 조사는 끝났다. 원인은 비결정성이 아니라 **실행 환경 불일치**다 —
> frozen cache는 `detector_runs/venv`(torch 2.10 / cuDNN 9.10.2)에서 생성됐는데 RGB 재실행이
> `datasets/detenv`(torch 2.4 / cuDNN 9.1)로 돌았다. 올바른 환경으로 전체 2,296프레임을 다시 돌리면
> `rank_mismatches 0`으로 frozen output이 그대로 재현된다. 아래 §작업 경계의 "RGB 실행은
> `datasets/detenv/bin/python`"은 **이 사고의 원인이므로 따르지 않는다.**
> 결과: [`perception_streaming_findings_2026-09-09.md`](perception_streaming_findings_2026-09-09.md).

## 현재 상태

저장소: `/home/fair/workspaces/aerial_gym_ws/src/aerial_gym_simulator`.
[결과](../../results/perception_p7e_streaming_2026-09-08/README.md),
[API](../specs/perception_streaming_v1.md),
[원본 artifact 경로와 SHA](../../results/perception_p7e_streaming_2026-09-08/summary.json)를 먼저 읽는다.
P7c v2가 선택 모델이다. P7e crop verifier는 utility 0.52352로 미채택했다.
cached replay 2,296프레임 logits/rank 정확 일치; full RGB 두 번은 rank 5개가 달랐다.
decode 포함 약 14.4 FPS이며, 아직 실제 camera/ROS/PPO 연결은 없다.

## 1. 먼저: historical candidate 수치 재현성 조사

`tools/run_perception_candidate_producer.py`와 `tools/perception_streaming.py`의 detector를
비교하라. 현재 환경에서 첫 프레임의 두 구현은 정확히 같지만 과거 candidate cache와 다르다.
rank 차이가 5개라는 이유만으로 tolerance를 완화하거나 gate를 PASS로 바꾸지 마라.

1. 원본 cache/receipt/checkpoint/source SHA를 확인하고 그대로 보존한다.
2. 당시 실행환경 증거와 현재 Python/torch/CUDA/cuDNN/OpenCV, YOLO revision,
   TF32/benchmark/determinism 설정, dtype/tile/NMS/descriptor를 비교한다.
   TF32 off, benchmark on, deterministic on을 각각 첫 프레임에서 시험했지만
   historical 첫 confidence를 재현하지 못했다. 원인 확정 증거로 해석하지 않는다.
3. 첫 차이가 발생하는 단계를 분리하고, 원인 가설을 작은 재현 실험으로 검증한다.
4. 원인을 고쳤다면 모델·threshold 변경 없이 전체 2,296 validation frames의
   replay/RGB audit를 미존재 output 경로로 다시 실행한다. 과거 환경을 복원할 수 없다면
   근거와 한계를 보고하고, 새로운 baseline/cache 버전 채택은 사용자 결정으로 남긴다.

완료 조건: 원인을 증거로 특정하고 frozen output을 재현하거나, 재현 불가의 구체적 경계와
버전 전환안을 제시한다. 모델 성능 개선 실험과 섞지 않는다.

## 2. 그다음: 성능 보존 latency 개선

현재 mean은 detector/descriptor 27.84 ms, motion/GMC 33.62 ms, selector 2.74 ms,
decode 포함 69.38 ms다. 우선 motion resize/flow/GMC/후보 특징 단계별 시간을 분리한다.
동일 출력을 보존하는 중복 계산 제거부터 검토한다. flow 해상도/빈도 변경, FP16,
다른 detector나 optical-flow 알고리즘은 모델 입력 분포를 바꾸므로 별도 실험 계약이 필요하다.
완료 조건: 동일 validation 입력의 전후 latency mean/P50/P95, ranks, hit/false-lock/no-lock/
reacquisition을 함께 보고한다. 빠르다는 이유만으로 품질 저하를 숨기지 않는다.

## 3. 이후: P8-lite/P9 준비

실기 정보 없이 가능한 pixel-center error, miss/false-lock burst, reacquisition, latency의
sequence-aware error model과 downstream 입력 schema부터 설계한다.
camera intrinsics/거리 GT/track identity 없이 degree bearing·metric range·실제 ID-switch를
만들어내지 않는다. P9/PPO 연결 전 stale/no-lock/reset 처리를 명시하고 별도 승인을 받는다.

## 작업 경계

- NPS test 열람, 재학습, threshold tuning, 원본 cache 덮어쓰기는 이 재현성 조사에 필요 없다.
- `results/independent_verification_2026-09-06/`은 인용 근거 보존을 위해 2026-09-09 추적 전환됐다.
- ~~RGB 실행은 `datasets/detenv/bin/python`~~ → **`detector_runs/venv/bin/python`**(2026-09-09 정정: cache를 만든 환경이며, 다른 환경에서는 재현되지 않는다).
- verifier CLI는 strict rank parity FAIL일 때 JSON 저장 후 exit 1을 반환한다. 예상된 실패도 기록한다.
- 30 FPS, 실기 탑재, identity tracking, ROS/PPO 연결 완료를 현재 결과로 주장하지 않는다.
