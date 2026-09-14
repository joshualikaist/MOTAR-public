# NavRL detector offline gate v2

- candidate: `balanced_bce`
- validation-selected threshold: `0.425`
- artifact SHA-256: `c5d8b1780108e999833502ea5d0029c40fc790756583eae115f3042f7f311cc3`
- held-out test decision: **FAIL**

| metric | held-out test |
|---|---:|
| frame precision | 0.9749 |
| frame recall | 0.9366 |
| absent FPR | 0.0072 |
| full-occlusion FPR | 0.0134 |
| partial-occlusion recall | 0.9955 |
| small-target recall | 0.8805 |
| far 14-20 m recall | 0.8539 |
| pixel precision / IoU | 0.1717 / 0.1712 |
| bearing MAE | 0.374 deg |
| range MAE | 0.068 m |

## Gate checks

- [ ] frame_recall>=0.95
- [ ] frame_precision>=0.98
- [x] absent_fpr<=0.01
- [ ] full_occlusion_fpr<=0.01
- [x] far_recall>=0.85
- [x] partial_recall>=0.85
- [x] small_recall>=0.80
- [ ] pixel_precision>=0.95
- [x] bearing_mae<=1.5deg
- [x] range_mae<=0.25m
- [x] test_absent_frames>=500
- [x] test_full_occlusion_frames>=100
- [x] test_partial_frames>=50
- [x] test_small_frames>=100
