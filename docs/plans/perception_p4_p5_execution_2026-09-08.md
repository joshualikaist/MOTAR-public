# P4 candidate producer + P5 Kalman association 실행 계약

작성: 2026-09-08  
상태: **COMPLETE — P4 PASS; P5 KF v1 REJECTED**

P4는 NPS 원본 clip의 container FPS와 원 frame 번호로 capture timestamp를 복원하고, 각 frame에서 native
640/128 sliding-window detector 결과를 image-level NMS한 뒤 confidence 순 Top-K=5를 기록한다. 후보마다
box·confidence와 재현 가능한 parameter-free RGB-grid/histogram 64D L2 descriptor를 출력한다. 낮은 score도
후속 비교에서 같은 입력을 쓰도록 producer floor는 `0.001`이다.

P5 상태는 `[u,v,log(w),log(h),du,dv,dlog(w),dlog(h)]` constant-velocity KF다. 4D innovation의
chi-square p=0.99 gate, appearance cosine cost, confidence penalty를 결합해 최대 5×5 exact assignment를
수행한다. 새 track threshold 0.25, update threshold 0.05, 2 hit confirm, 최대 coasting 0.5초를 고정한다.
기계 판독값은 `configs/perception_kf_v1.json`이 정본이다.

최종 detector hash를 동결한 뒤 NPS validation에서 pipeline 무결성을 확인하고, 같은 설정으로 NPS test를
한 번 평가한다. CNN-only는 confidence≥0.25인 top-1, KF arm은 같은 P4 후보의 confirmed track을 쓴다.
IoU≥0.3 frame hit, no-lock, non-overlap proxy false lock, center error, loss/reacquisition, detector P/R,
producer/KF latency를 보고한다.

NPS YOLO 변환 당시 원 annotation의 track ID가 보존되지 않았고 일부 frame에는 여러 UAV box가 있다.
따라서 실제 FTLR와 ID switch는 측정 불가능하며 proxy를 해당 지표로 이름 바꾸지 않는다. 카메라 intrinsic도
없으므로 degree bearing error는 보류한다.

## 실행 결과

P4는 NPS validation 2,296 frames/7 clips와 test 1,725 frames/8 clips를 모두 생성했다. 두 split 모두
schema, semantic condition, source manifest, compressed payload hash 검증을 통과했다. test candidate
receipt SHA-256은 `ff191460fefa0d4815d6a94a6cac751e56504951cd284943d1f3065daa036696`이다.

validation에서 고정 KF v1은 이미 CNN-only보다 낮았다(frame hit `0.81490 → 0.62631`, proxy false lock
`0.15492 → 0.36652`). 유한한 validation-only 대안을 확인했지만 CNN-only의 hit/false-lock 균형을
넘는 설정이 없어 v1을 그대로 test에서 한 번 평가했다. test 결과는 다음과 같다.

| arm | IoU≥0.3 frame hit | proxy false lock / selected | center error mean | no-lock frames |
|---|---:|---:|---:|---:|
| CNN-only top-1 | 0.61449 | 0.24501 | 1.9127 px | 321 |
| CNN + KF v1 | 0.46551 | 0.48426 | 3.9748 px | 168 |

KF가 lock을 더 자주 유지하지만 틀린 후보를 지속해 hit와 false-lock 모두 악화했다. 따라서
**P5 KF v1은 REJECTED**다. 실제 ID switch/FTLR는 원 track ID와 designated target ID가 없어 식별할 수
없으며, 표의 false lock은 모든 annotation과 IoU<0.3인 selected box의 proxy다. camera intrinsic 부재로
degree bearing error도 unavailable이다.

candidate producer mean/p95 latency는 `61.43 / 122.81 ms`, KF 추가 mean/p95는
`0.327 / 0.577 ms`였다. producer 측정 당시 Det-Fly GPU 평가가 동시에 실행 중이었으므로 이는
**contended offline RTX 3070 measurement**이지 배포 latency가 아니다. test KF receipt SHA-256은
`9d3a810bd6eb02f474698b92f80441513f7fbc55ca8c570a0ad1dd3250200a2b`이다.
