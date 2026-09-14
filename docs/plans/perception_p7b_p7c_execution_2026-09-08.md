# P7b candidate-preserving + P7c frame-dynamics 실행 계약

작성: 2026-09-08

상태: **COMPLETE — CORRECTED P7c v2 SELECTED ON VALIDATION**

최종 utility: 기존 P7 0.67247, P7b 0.67944, corrected P7c v2 0.68554.
P7c v2는 false lock이 감소했지만 no-lock과 재획득 시간이 증가했다.
체크포인트 재로드 prediction bytes는 원 결과와 일치했다. 상세 결과는
[결과 정본](../../results/perception_p7bc_2026-09-08/README.md)을 참조한다.

## 2026-09-08 구현 정정

첫 P7c run의 utility 0.681184668989547은 무효다. 직전 후보 선택에서 GMC로 변환한
축소 좌표를 원본 후보 좌표와 비교하는 단위 오류가 발견됐다. mapped point를 원본
좌표로 복원하도록 수정하며 기존 cache/run/selection은 감사용으로 보존한다.
수정 cache와 run은 `v2` 경로에 새로 생성한다. architecture, seed, optimizer,
checkpoint 선택과 utility 규칙은 그대로이며 P7b는 재학습하지 않는다.
이 정정은 validation 결과를 관찰한 뒤 수행되므로 독립적인 새 사전등록 결과로 주장하지 않는다.

## 목적과 격리

기존 P7 Transformer의 validation utility 이득은 CNN-only 대비 `+0.00697`로 작다. 이 후속 실험은
Transformer 용량을 키우지 않고, 과거 frame의 Top-5 후보를 하나로 pooling하면서 소실된 후보별 정보와
ego-camera motion을 입력에 복원한다. 기존 P7을 포함한 세 arm은 동일한 frozen P4 candidate, NPS train
13,510 frames/35 clips, validation 2,296 frames/7 clips만 사용한다. **NPS test manifest/candidate/report는
architecture 선택 입력으로 읽지 않는다.**

원 NPS track ID가 변환 과정에서 보존되지 않았으므로 세 arm 모두 designated-target tracker가 아니라
UAV candidate selector다. IDF1, ID switch, fragmentation, 실제 FTLR는 P7d의 track-ID 데이터 전까지
보고하지 않는다.

## 고정 arm

### Current P7

기존 `perception_transformer_v1.json`과 checkpoint를 그대로 사용한다. 최근 16 frame 각각의 후보를
confidence-weighted frame token 하나로 pooling한 baseline이다. 재학습하지 않는다.

### P7b — candidate-preserving Transformer

- history 16, Top-K 5, 최대 80 candidate 위치를 유지한다.
- 현재 frame을 제외한 최대 15×5 후보와 항상 유효한 cold-start token을 2-layer Transformer encoder에
  넣는다.
- 현재 frame의 각 후보가 encoded history에 4-head cross-attention한다.
- token 입력은 기존 69D `[u,v,w,h,confidence,appearance_64d]`와 current frame까지의 age seconds다.
- hidden 128, FFN 256, dropout 0.1이며 current P7과 같은 크기 계열을 유지한다.

### P7c — P7b + frame dynamics/GMC

P7b token마다 아래 고정 12D motion feature를 추가한다.

1. GMC 보정 후 가장 가까운 직전 후보와의 `du,dv,dlog(w),dlog(h)` 4D
2. candidate crop의 dense backward optical flow 2D
3. global camera-motion backward flow 2D
4. local−global residual flow 2D
5. GMC RANSAC inlier ratio와 valid flag 2D

Optical flow는 원 RGB를 최대 폭 640 px로 축소한 뒤 Farneback
`pyr_scale=.5, levels=3, winsize=21, iterations=3, poly_n=5, poly_sigma=1.2`로 current→previous
방향을 계산한다. GMC는 16 px grid의 flow correspondence에서 후보 box를 1.5배 확장해 제외하고,
`estimateAffinePartial2D` RANSAC threshold 1.5 px로 추정한다. feature cache는 manifest와 candidate
SHA-256에 결박한다. train/validation cache만 만들며 test cache는 만들지 않는다.

## 학습·선택 규칙

P7b/P7c 모두 AdamW, learning rate `1e-3`, weight decay `1e-4`, batch 256, seed 17, 최대 40 epochs,
gradient clip 1.0, early-stopping patience 8을 사용한다. checkpoint는 validation cross-entropy 최소,
exact tie면 더 이른 epoch를 선택한다.

최종 arm utility는 기존과 같은

```text
(hit_frames - proxy_false_lock_frames) / frames_with_ground_truth
```

이다. utility 최대 arm을 선택하고 exact tie는 복잡도가 낮은
`Current P7 → P7b → P7c` 순이다. frame hit, proxy false lock, no-lock, center error와 temporal-only
latency도 함께 기록한다. P7b/P7c가 이기지 못하면 current P7을 유지한다.

## 다음 경계

선택 architecture를 freeze한 뒤에만 P6/P7 계열 NPS test 평가를 한 번 수행할 수 있다. P7d learned
appearance/ReID는 MOTChallenge 형식의 실제 air-to-air track ID 데이터 receipt와 identity-disjoint split을
별도 사전등록한 뒤 시작한다. 신뢰 가능한 identity/motion gate가 생기기 전에는 memory bank를 붙이지 않고,
Mamba/대형 Transformer는 현재 범위에서 제외한다.
