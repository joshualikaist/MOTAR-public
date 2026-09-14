# NavRL detector offline gate v2

- candidate: `spatial_cnn+focal_dice`
- validation-selected threshold: `0.500`
- artifact SHA-256: `b700778e3d686ed88e6a93d99e07a17ae9b15174bf5181911146a2dbc0a29a9b`
- held-out test decision: **FAIL**

| metric | held-out test |
|---|---:|
| frame precision | 0.9975 |
| frame recall | 0.9340 |
| absent FPR | 0.0012 |
| full-occlusion FPR | 0.0010 |
| partial-occlusion recall | 0.9457 |
| small-target recall | 0.8547 |
| far 14-20 m recall | 0.8571 |
| pixel precision / IoU | 0.9997 / 0.8929 |
| bearing MAE | 0.040 deg |
| range MAE | 0.274 m |

## Gate checks

- [ ] frame_recall>=0.95
- [x] frame_precision>=0.98
- [x] absent_fpr<=0.01
- [x] full_occlusion_fpr<=0.01
- [x] far_recall>=0.85
- [x] partial_recall>=0.85
- [x] small_recall>=0.80
- [x] pixel_precision>=0.95
- [x] bearing_mae<=1.5deg
- [ ] range_mae<=0.25m
- [x] test_absent_frames>=500
- [x] test_full_occlusion_frames>=100
- [x] test_partial_frames>=50
- [x] test_small_frames>=100
