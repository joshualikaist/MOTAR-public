# NavRL detector offline gate v2

- candidate: `spatial_cnn+focal_dice`
- validation-selected threshold: `0.075`
- artifact SHA-256: `354da116c82ab78571f742b394c7dd150903331ffc57e405f286ce74b54c7bdc`
- held-out test decision: **FAIL**

| metric | held-out test |
|---|---:|
| frame precision | 0.9837 |
| frame recall | 1.0000 |
| absent FPR | 0.0072 |
| full-occlusion FPR | 0.0083 |
| partial-occlusion recall | 1.0000 |
| small-target recall | 1.0000 |
| far 14-20 m recall | 1.0000 |
| pixel precision / IoU | 0.7995 / 0.7987 |
| bearing MAE | 0.160 deg |
| range MAE | 0.878 m |

## Gate checks

- [x] frame_recall>=0.95
- [x] frame_precision>=0.98
- [x] absent_fpr<=0.01
- [x] full_occlusion_fpr<=0.01
- [x] far_recall>=0.85
- [x] partial_recall>=0.85
- [x] small_recall>=0.80
- [ ] pixel_precision>=0.95
- [x] bearing_mae<=1.5deg
- [ ] range_mae<=0.25m
- [x] test_absent_frames>=500
- [x] test_full_occlusion_frames>=100
- [x] test_partial_frames>=50
- [x] test_small_frames>=100
