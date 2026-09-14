# NavRL detector offline gate v2

- candidate: `spatial_cnn_wide+focal_dice`
- validation-selected threshold: `0.700`
- artifact SHA-256: `85c7974bcd85c627170c5bd63030144d1c5dc2a11e5d64829cad38f615c5d5d7`
- held-out test decision: **PASS**

| metric | held-out test |
|---|---:|
| frame precision | 0.9977 |
| frame recall | 0.9938 |
| absent FPR | 0.0000 |
| full-occlusion FPR | 0.0015 |
| partial-occlusion recall | 0.9919 |
| small-target recall | 0.9873 |
| far 14-20 m recall | 0.9969 |
| pixel precision / IoU | 1.0000 / 0.9691 |
| bearing MAE | 0.024 deg |
| range MAE | 0.178 m |

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
