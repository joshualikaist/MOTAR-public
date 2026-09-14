# NavRL detector offline gate v2

- candidate: `pixel_1x1+focal_dice`
- validation-selected threshold: `0.550`
- artifact SHA-256: `eeb332ec770368e13d89f78da267b1bea5511780006bd6c7408768dff7d80aa7`
- held-out test decision: **FAIL**

| metric | held-out test |
|---|---:|
| frame precision | 0.9860 |
| frame recall | 0.8498 |
| absent FPR | 0.0048 |
| full-occlusion FPR | 0.0062 |
| partial-occlusion recall | 0.8604 |
| small-target recall | 0.8249 |
| far 14-20 m recall | 0.8223 |
| pixel precision / IoU | 0.6566 / 0.5918 |
| bearing MAE | 0.159 deg |
| range MAE | 0.027 m |

## Gate checks

- [ ] frame_recall>=0.95
- [x] frame_precision>=0.98
- [x] absent_fpr<=0.01
- [x] full_occlusion_fpr<=0.01
- [ ] far_recall>=0.85
- [x] partial_recall>=0.85
- [x] small_recall>=0.80
- [ ] pixel_precision>=0.95
- [x] bearing_mae<=1.5deg
- [x] range_mae<=0.25m
- [x] test_absent_frames>=500
- [x] test_full_occlusion_frames>=100
- [x] test_partial_frames>=50
- [x] test_small_frames>=100
