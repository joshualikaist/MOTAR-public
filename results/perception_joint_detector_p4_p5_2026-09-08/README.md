# Joint detector + P4/P5 결과

작성: 2026-09-08

상태: **JOINT DETECTOR COMPLETE · P4 PASS · P5 KF v1 REJECTED**

대용량 prediction과 checkpoint는 workspace 외부 결과 디렉터리에 보존하고, 이 디렉터리에는 결과와
무결성 hash만 기록한다. test를 학습·checkpoint 선택·threshold 조정에 사용하지 않았다.

## 합동 detector

| 평가 | P | R | AP30 | 비고 |
|---|---:|---:|---:|---|
| Det-Fly `020` native 4K | 0.27291 | 0.86400 | **0.73887** | 6,913 images, TP/FP/FN 5,972/15,911/940 |
| Det-Fly `020` scale-matched ÷4 | 0.59319 | 0.58478 | 0.59719 | 크기 진단 arm |

validation-selected epoch 26의 P/R/mAP50/mAP50-95는
`0.77634 / 0.65901 / 0.69167 / 0.32339`다. 동결 checkpoint SHA-256은
`aab12f39bdc79f201657002f12d2096fab916600b546481964f8fcacdb3300ac`이다.

## P4/P5 NPS test

P4는 1,725 frames/8 clips에서 최대 5개의
`[u,v,w,h,confidence,appearance_64D]` 후보를 생성했고 schema·semantic·source manifest·payload hash
검증을 통과했다. appearance는 parameter-free descriptor이며 learned ReID 성능 주장이 아니다.

| arm | frame hit | proxy false lock | center error mean | reacquisition mean |
|---|---:|---:|---:|---:|
| CNN-only top-1 | **0.61449** | **0.24501** | **1.9127 px** | **0.7201 s** |
| CNN + KF v1 | 0.46551 | 0.48426 | 3.9748 px | 0.8692 s |

KF v1은 lock을 더 자주 유지했지만 오검출을 지속해 CNN-only보다 낮았다. 판정은 **REJECTED**다.
실제 ID switch/FTLR는 source track ID와 designated target ID가 없어서 식별 불가능하다. degree bearing
error도 camera intrinsic 부재로 unavailable이다. producer mean/p95 `61.43 / 122.81 ms`는 다른 Det-Fly
평가가 같은 RTX 3070을 사용하던 contended offline 측정이며 배포 latency가 아니다.

## 외부 원자료와 SHA-256

원자료 root:
`/home/fair/workspaces/aerial_gym_ws/detector_runs/results/nps_detfly_joint_final`

| 파일 | SHA-256 |
|---|---|
| `detfly_native/report.json` | `b786eba1f8268d8562aa8474407788780bfebb107338d11ef5e6416bb8f97db7` |
| `detfly_native/predictions.jsonl.gz` | `e01d398a70bb1769015581bd8202763b78e52828817f843ba0490732adba01f2` |
| `detfly_native/receipt.json` | `1ccf9dd755badde1df3b880985dd1b86af8edb9727e71b698daed4f2271ffcb8` |
| `detfly_scale_matched_4x/report.json` | `8295f09cf9cda17b97b378ac520d347f13e5da5a5fa73daed94e60e36723a5ce` |
| `detfly_scale_matched_4x/predictions.jsonl.gz` | `ae991c131ff35e28feba5949a5e29916eae971ee48c21e450f545b7aa2b12c43` |
| `detfly_scale_matched_4x/receipt.json` | `a12f62c5fe69a18f53e449cb4b1d646b0fee0118e66ce27d1f4314ef9534c9e3` |
| `nps_test_candidates.jsonl.gz` | `fbad60aa33961b403f253a914a70915b43568e69d7c5ec625fb4331edbacd417` |
| `nps_test_candidates.jsonl.gz.receipt.json` | `ff191460fefa0d4815d6a94a6cac751e56504951cd284943d1f3065daa036696` |
| `nps_test_kf/report.json` | `324eb9c4d8ce85f5f82cf9662510de78f188de8aa653e593e0ce00cb8dcf4b0c` |
| `nps_test_kf/tracks.jsonl.gz` | `351fb3a6443d612cbbd40e1d77f8ccf2fa0b288d83e3104362fb93635bee79fe` |
| `nps_test_kf/receipt.json` | `9d3a810bd6eb02f474698b92f80441513f7fbc55ca8c570a0ad1dd3250200a2b` |

기계 판독 요약은 [`summary.json`](summary.json)이다.
