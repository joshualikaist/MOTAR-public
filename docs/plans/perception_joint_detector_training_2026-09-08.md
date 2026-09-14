# NPS + Det-Fly detector 학습·최종 평가 계약

작성: 2026-09-08  
상태: **COMPLETE — validation-selected checkpoint frozen before sealed test**

## 목적과 격리

P3에서 동결한 NPS-only checkpoint를 NPS+Det-Fly v1 train/validation으로 미세 조정한다. Det-Fly
`020` 전체 6,913 frame은 sealed final test이고 학습 YAML에 존재하지 않는다. 학습 종료, validation 기반
`best.pt` 선택, checkpoint SHA-256 동결이 끝난 뒤에만 test manifest를 읽는다.

## 동결 학습 조건

| 항목 | 값 |
|---|---|
| 초기화 | NPS `best.pt`, SHA-256 `ccd65dc3…f580` |
| 구현 | YOLOv5 commit `35b48237aef6d71ca9de2c5dea345d7536eb7fa7` |
| data | `nps_detfly_joint_v1/joint_detector.yaml`; train 29,824 / val 9,307 samples |
| 해상도 | base 640, multi-scale 320–960 |
| 최적화 | SGD, batch 8, 최대 30 epochs, patience 10, seed 0, workers 8 |
| augmentation | `configs/perception_joint_detector_hyp_v1.yaml` |
| anchor | train labels에 대한 YOLOv5 AutoAnchor 활성화; validation/test 미사용 |
| layer | 전체 fine-tuning, freeze 없음 |
| 선택 | YOLOv5 validation fitness의 최고 checkpoint; test/운영 threshold 미사용 |

RTX 3070에서 CUDA OOM이 발생하면 이 run을 실패로 보존한다. batch size를 몰래 바꾸어 재개하지 않고,
별도 이름과 계약 수정으로 batch 4 실험을 시작한다.

## 최종 test 계약

선택된 `best.pt`의 SHA-256을 먼저 기록한다. 그 다음 sealed Det-Fly `020` manifest 전체에 아래 두 arm을
각각 한 번 실행한다.

1. **primary native-4K:** 640 px / 128 px overlap sliding window
2. **diagnostic scale-matched:** 원 영상을 1/4로 축소 후 동일 detector 적용

두 arm 모두 inference confidence floor `0.001`, 보고용 operating threshold `0.25`, NMS IoU `0.45`,
평가 IoU `0.3`과 `0.5`를 쓴다. threshold/anchor/model은 test 결과를 보고 조정하지 않는다. 전체 P/R/AP,
pixel-size bin, frame 수, raw prediction receipt를 보존한다. Primary native arm이 최종 detector 성능이고
scale-matched arm은 크기 민감도 진단일 뿐 모델 선택에 사용하지 않는다.

## P4/P5 경계

P4 후보 생성과 P5 KF 평가는 위 `best.pt`를 detector로 동결한 뒤 NPS test source clips에서 수행한다.
NPS는 단일 표적 영상이므로 miss/FP/center error/track loss/reacquisition/latency는 측정하지만 실제
multi-target ID-switch는 `NOT_IDENTIFIABLE_SINGLE_TARGET`로 명시한다.

## 실행 결과

- 30/30 epochs를 계약대로 완료했다. AutoAnchor BPR은 1.000이어서 anchor 변경이 필요 없었다.
- validation fitness로 선택된 epoch는 zero-based 26이다. P/R/mAP50/mAP50-95는
  `0.77634 / 0.65901 / 0.69167 / 0.32339`이다.
- test 접근 전에 동결한 `best.pt` SHA-256은
  `aab12f39bdc79f201657002f12d2096fab916600b546481964f8fcacdb3300ac`이다.
- sealed Det-Fly `020` native 4K의 IoU 0.3 P/R/AP는
  `0.27291 / 0.86400 / 0.73887`이다. confidence 0.25에서 TP/FP/FN은
  `5,972 / 15,911 / 940`이다.
- scale-matched ÷4 diagnostic의 IoU 0.3 P/R/AP는
  `0.59319 / 0.58478 / 0.59719`이다.

NPS-only native AP30 `0.00354`에 비해 joint detector가 표적 크기·domain gap을 크게 회복했다.
그러나 native operating-point precision `0.27291`은 낮다. test 결과를 본 뒤 threshold를 바꾸지 않았으며,
운영 threshold 보정은 별도 validation 계약이 필요하다. 원자료 경로와 hash는
`results/perception_joint_detector_p4_p5_2026-09-08/`에 기록한다.
