# P7e 및 streaming perception 결과

상태: **P7e NOT SELECTED; streaming IMPLEMENTED; cached replay PASS;
historical RGB rank parity FAIL.** NPS test는 이번 작업에서 사용하지 않았다.

## 구현

[사전등록](../../docs/plans/perception_p7e_streaming_2026-09-08.md)은 결과 관찰 전
`f71079c`에서 고정했다. [crop verifier](../../tools/perception_crop_verifier.py)는
NPS train 35 clips의 clip-disjoint 3-fold BCE로 epoch를 정하고 전체 train으로 재학습했다.
fold best epoch는 8/7/8, 최종 8 epoch다. validation threshold는 고정 0.5이며 사후 튜닝하지 않았다.

[파이프라인](../../tools/perception_streaming.py)은 BGR frame → frozen tiled detector →
Top-5 descriptor → backward flow/GMC → clip-local T16 → P7c 선택이다.
sequence reset, timestamp/geometry guard, checkpoint hash 검사를 포함한다.
온라인 API는 GT를 받지 않는다. rank는 프레임 내 후보 번호이지 target identity가 아니다.
[API와 실행법](../../docs/specs/perception_streaming_v1.md).

## 모델 선택 — 기존 P7c 유지

| NPS validation 2,296 frames | P7c v2 cached | P7e crop verifier |
|---|---:|---:|
| Frame hit | 1,848 (0.80488) | 1,500 (0.65331) |
| Proxy false locks | 274 | 298 |
| No-lock frames | 174 | 498 |
| Selection utility | **0.68554** | 0.52352 |

P7e는 이번 작은 crop-validity 설계에서 개선되지 않아 미채택이다. 이 결과가 learned
appearance 전체의 불가능성을 뜻하지는 않는다. 64D learned embedding을 기존 histogram
대신 사용하려면 temporal selector도 재학습해야 한다. 이번에는 그런 교체를 하지 않았다.

## Streaming 검증과 미통과 항목

- Cached candidate+motion replay: 전체 2,296프레임에서 offline logits와 최대 차이 **0**,
  rank mismatch **0**. 기존 P7c metrics를 정확히 재현했다.
- RGB 전체 재검출: 두 번 실행 모두 historical candidate cache와 rank **5/2,296** 차이.
  candidate dictionary의 exact byte-value 비교는 전 프레임에서 달랐다. 이는 전 프레임의
  검출 실패를 뜻하지 않으며, 작은 실수 차이도 mismatch로 집계한다.
- RGB 재실행 hit 1,848, false locks 273, no-lock 175, utility 0.685976.
  이 미세 차이를 성능 향상으로 해석하거나 새 checkpoint를 선택하지 않았다.
- 기존 P4 producer를 현재 환경에서 그대로 실행한 **첫 프레임**의 후보는 새 pipeline과
  전부 같지만 historical cache와 달랐다. [진단 원문](producer_diagnostic.json).
  따라서 streaming 변환만의 오류라고 단정할 수 없다. 과거/현재 실행의 수치 차이를
  만드는 정확한 환경·연산 설정 원인은 미확정이다. 이 점검을 전체 producer parity로
  확대 해석하지 않는다.
- RGB audit CLI는 결과 JSON을 남긴 뒤 strict rank parity 미통과로 **exit 1**을 반환했다.
  최초 실패 결과는 보존했고, 진단 정보 보강 후 재실행 결과도 별도 파일로 보존했다.

## RTX 3070 직렬 처리 시간

두 번째 전체 실행, validation JPEG decode 포함, cold first frame 포함. 모든 값 ms/frame.

| 단계 | Mean | P50 | P95 |
|---|---:|---:|---:|
| Detector + descriptor | 27.84 | 25.69 | 32.59 |
| Motion/GMC | 33.62 | 36.43 | 39.67 |
| T16 selector | 2.74 | 2.67 | 3.13 |
| Pipeline 합계 | 64.20 | 64.52 | 67.78 |
| Decode 포함 | **69.38** | 69.44 | 72.08 |

Decode 포함 **14.41 FPS**. 과거 motion cache와 같은 grayscale을 얻으려고 JPEG의
BGR/gray를 각각 decode했다. 실제 카메라 입력, ROS 전송, queue, PPO/control 비용은 제외했다.
30 FPS 실시간이나 실기 탑재 성능을 주장하지 않는다. motion과 detector가 주 비용이므로
VRAM 추가 사용이나 Transformer 확대만으로 해결되는 병목은 아니다.

## 검증·재현 자료

[summary.json](summary.json)에 fold별 BCE, checkpoint/source/원본 artifact SHA-256,
전체 metrics, 두 RGB 실행의 latency와 runtime 설정을 보존했다. 대용량 checkpoint와
전체 validation ranks는 저장소 밖 원본 경로에 있으며 summary에는 ranks만 생략했다.
Python 1,221 tests PASS (4 skipped). 사이트는 이번에 수정하지 않았다.
당시 미추적이던 `results/independent_verification_2026-09-06/`은 이 작업에서는 수정하지 않았고,
인용 근거 보존을 위해 2026-09-09 별도 커밋에서 추적 전환됐다.

다음 순서: 수치 재현성 설정 고정 → motion/GMC latency 개선 → P8-lite/P9 입력 계약.
P7d track identity, 실기 camera calibration/range GT, 실제 flight/ROS/PPO 연결은 미완료다.


---

## 후속 (2026-09-09) — RGB parity 해소

이 디렉터리의 `summary.json`은 `status: IMPLEMENTED_WITH_HISTORICAL_RGB_PARITY_FAILURE`로 동결돼
있고 그대로 보존한다. 그 실패의 **원인은 모델이 아니라 실행 환경이었다.**

frozen candidate cache는 `detector_runs/venv`(Python 3.10 / torch 2.10.0+cu128 / CUDA 12.8 /
cuDNN 9.10.2)에서 생성됐는데, 당시 RGB 재실행은 `datasets/detenv`(3.8 / 2.4.1+cu121 / 12.1 / 9.1.0)로
돌았다. 두 스택은 서로 다른 컨볼루션 커널을 골라 같은 60프레임에서 confidence가 최대 45 % 어긋난다.

올바른 환경으로 전체 2,296프레임을 다시 돌린 결과가
`detector_runs/results/nps_detfly_joint_final/stream_rgb_venv_v1.json`이며
`rank_mismatches 0` / `rank_parity_pass true` / `candidate_frame_mismatches 0`이고, selected 2122 /
no_lock 174 / false_lock 274 / center error 1.941796 / loss 106이 cached replay와 전부 일치한다.

조사 전문: [`docs/plans/perception_streaming_findings_2026-09-09.md`](../../docs/plans/perception_streaming_findings_2026-09-09.md).
