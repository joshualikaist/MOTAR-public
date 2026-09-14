# NavRL detector offline gate v2

- candidate: `balanced_bce`
- validation-selected threshold: `0.550`
- artifact SHA-256: `8da32d6f21bfbd3bdd5ec5de9ef9cb09e8deb4bd5ce511630e19afee33f26f10`
- held-out test decision: **PASS**

| metric | held-out test |
|---|---:|
| frame precision | 1.0000 |
| frame recall | 1.0000 |
| absent FPR | 0.0000 |
| full-occlusion FPR | 0.0000 |
| partial-occlusion recall | 1.0000 |
| small-target recall | 1.0000 |
| far 14-20 m recall | 1.0000 |
| pixel precision / IoU | 1.0000 / 1.0000 |
| bearing MAE | 0.000 deg |
| range MAE | 0.000 m |

## Gate checks

- [x] frame_recall>=0.95
- [x] frame_precision>=0.98
- [x] absent_fpr<=0.01
- [x] full_occlusion_fpr<=0.01
- [x] far_recall>=0.85
- [x] partial_recall>=0.85
- [x] small_recall>=0.80
- [x] pixel_precision>=0.95
- [x] bearing_mae<=1.5deg
- [x] range_mae<=0.25m
- [x] test_absent_frames>=500
- [x] test_full_occlusion_frames>=100
- [x] test_partial_frames>=50
- [x] test_small_frames>=100
