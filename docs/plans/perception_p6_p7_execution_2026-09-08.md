# P6 GRU + P7 Temporal Transformer 실행 계약

작성: 2026-09-08

상태: **COMPLETE — P6 GRU NOT SELECTED; P7 TRANSFORMER SELECTED ON VALIDATION**

## 범위와 데이터 격리

P6와 P7은 P4의 동일한 frozen detector와 Top-K=5 candidate record만 사용한다. detector를 재학습하거나
candidate를 모델별로 다시 만들지 않는다. NPS train clip으로 weight를 학습하고 NPS validation clip으로
epoch와 association arm을 선택한다. 이 단계에서는 NPS test candidate/report를 읽지 않는다.

NPS 원 annotation의 target track ID가 YOLO 변환에서 보존되지 않았고 한 frame에 UAV가 여러 개일 수 있다.
따라서 supervision은 **현재 frame에서 어떤 annotation과든 IoU≥0.3인 candidate**이며, 가장 높은 IoU
candidate를 정답으로 쓴다. 그런 candidate가 없으면 `NO_LOCK`이 정답이다. 이것은 temporal UAV-candidate
selection baseline이지 designated-target ReID 또는 ID-switch 평가가 아니다.

## 공통 입력과 학습 조건

후보 feature는 image 크기로 정규화한 `[u,v,w,h]`, confidence, parameter-free P4 appearance 64D의
69차원이다. 한 clip 안에서 최근 sampled frame만 시간순으로 사용하고 clip 경계를 넘지 않는다. 원 source
frame 간격은 capture timestamp로 계산해 frame representation에 포함한다. 미래 frame은 사용하지 않는다.

- optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`
- batch: 256, seed: 17, 최대 40 epochs, gradient norm clip: 1.0
- dropout: 0.1, early stopping patience: 8
- checkpoint 선택: validation cross-entropy 최소; exact tie면 더 이른 epoch
- detector candidate floor/top-K와 appearance encoder는 P4 v1 그대로 유지

## P6 GRU

- history: 최근 8 sampled frames
- candidate embedding: 128D
- confidence-weighted frame pooling 후 timestamp gap을 결합
- single-layer GRU hidden 128
- 현재 frame의 5 candidate와 `NO_LOCK`을 6-way 분류

## P7 Temporal Transformer

- history: 최근 16 sampled frames
- model dimension 128, 4 heads, 2 encoder layers, feed-forward 256
- confidence-weighted frame token과 timestamp gap, learned temporal position을 사용
- 현재 frame의 5 candidate와 `NO_LOCK`을 6-way 분류

두 모델은 같은 candidate encoder/scoring head 의미를 갖지만 weight를 공유하지 않는다. Transformer는
현재 frame까지의 window만 입력받으므로 미래 정보가 없다.

## validation 비교와 완료 조건

각 모델은 선택 checkpoint로 validation 전체를 한 번 출력하고 prediction/report/receipt SHA-256을 남긴다.
P5 report의 CNN-only와 KF도 같은 candidate/manifest hash인지 검증한 뒤 아래 utility로 네 arm을 비교한다.

```text
selection_utility = (hit_frames - proxy_false_lock_frames) / frames_with_ground_truth
```

utility 최대 arm을 선택한다. exact tie 순서는 `CNN-only → KF → GRU → Transformer`로, 더 단순한 arm을
우선한다. frame hit, no-lock, proxy false lock, center error, loss/reacquisition, association latency를 모두
함께 보고한다. 실제 FTLR/ID switch로 이름을 바꾸지 않는다.

P6/P7 완료 조건은 코드·단위 테스트, train candidate receipt, 두 모델 checkpoint/report/receipt,
validation-only selection receipt가 모두 존재하고 hash 검증되는 것이다. test 평가는 별도 승인 단계다.

## 실행 결과

P4 train 후보는 NPS train 13,510 frames/35 clips에서 67,550개가 생성됐고 schema·semantic·manifest·hash
전수 검증을 통과했다. train candidate SHA-256은
`bd28f4e296d622984c416eb05b2b343f3f38501ccaddf80047b3b7eaddb19061`이다.

GRU는 validation cross-entropy가 첫 epoch 뒤 악화해 zero-based epoch 0이 선택됐다. Transformer는
epoch 5가 선택됐다. 두 checkpoint 모두 clean source commit `5b3ee5e`에서 만들어졌고 test는 입력으로
사용되지 않았다.

| arm | frame hit | proxy false lock / selected | no-lock | utility |
|---|---:|---:|---:|---:|
| CNN-only | **0.81490** | 0.15492 | 82 | 0.66551 |
| CNN + KF v1 | 0.62631 | 0.36652 | 26 | 0.26394 |
| CNN + GRU T=8 | 0.78659 | 0.16466 | 134 | 0.63153 |
| CNN + Transformer T=16 | 0.80880 | **0.14424** | 126 | **0.67247** |

고정 rule로 **CNN + Temporal Transformer T=16**을 선택했다. CNN-only보다 hit는 14 frame 낮지만
proxy false lock이 30 frame 적어 utility가 `+0.00697` 높다. validation 2,296 frames/7 clips에서의
선택 결과일 뿐 test 우월성이나 실제 ID-switch 개선 주장이 아니다.

GRU/Transformer checkpoint SHA-256은 각각
`e377413635f1b8fa2640f2e846152ce565b23827dd03f76236f5b22ed9b6a86c`,
`9c3b91b977d8b6325d2064cf09b88bf3e75fe10b3656439896272f48bac83f1d`이다. frozen Transformer를
독립 evaluator로 다시 읽은 validation prediction은 압축 해제 내용 SHA-256이 원 training output과
동일한 `b6ca57c0f3e63946c31eb4be45ee88620543fcd3c0bddf386bd73f21ba8bba60`이었다.
