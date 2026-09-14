# P6/P7 temporal association 결과

작성: 2026-09-08

상태: **P6 COMPLETE · P7 TRANSFORMER SELECTED ON VALIDATION · TEST NOT RUN**

결과를 보기 전 commit `5b3ee5e`에서 architecture, optimizer, checkpoint selection, association selection
rule을 고정했다. NPS train 13,510 frames/35 clips으로 학습하고 validation 2,296 frames/7 clips으로만
checkpoint와 arm을 선택했다.

## Validation 결과

| arm | frame hit | proxy false lock / selected | center error | no-lock | utility |
|---|---:|---:|---:|---:|---:|
| CNN-only | **0.81490** | 0.15492 | 1.9496 px | 82 | 0.66551 |
| CNN + KF v1 | 0.62631 | 0.36652 | 4.2229 px | 26 | 0.26394 |
| CNN + GRU T=8 | 0.78659 | 0.16466 | 2.0556 px | 134 | 0.63153 |
| CNN + Transformer T=16 | 0.80880 | **0.14424** | **1.9259 px** | 126 | **0.67247** |

고정 utility `(hit_frames − proxy_false_lock_frames) / frames`에 따라
**CNN + Temporal Transformer T=16**을 선택했다. CNN-only보다 hit가 14 frame 적지만 proxy false lock이
30 frame 적어 utility가 `+0.00697` 높다. 차이가 작고 validation clip이 7개뿐이므로 test superiority
주장이 아니다. test는 P6/P7 학습·선택 입력이 아니며 별도 단계에서 한 번만 평가한다.

원 track ID와 designated target ID가 없어 실제 ID switch/FTLR는 식별할 수 없다. 이 결과는 현재 frame의
UAV candidate selection baseline이다. appearance 64D도 parameter-free descriptor라 learned ReID 주장이
아니다.

## 산출물과 SHA-256

외부 run root: `/home/fair/workspaces/aerial_gym_ws/detector_runs`

| 산출물 | SHA-256 |
|---|---|
| train manifest | `6d6c4b68d1fb2df9d019185e20cc816beab2617b656d7cb6a36b65dbed465170` |
| train candidates | `bd28f4e296d622984c416eb05b2b343f3f38501ccaddf80047b3b7eaddb19061` |
| GRU checkpoint | `e377413635f1b8fa2640f2e846152ce565b23827dd03f76236f5b22ed9b6a86c` |
| GRU report | `2d7845cd98fa57505bf1066e450c3253302f6d23977e174625c2f9bea2fd00ff` |
| Transformer checkpoint | `9c3b91b977d8b6325d2064cf09b88bf3e75fe10b3656439896272f48bac83f1d` |
| Transformer report | `c2952999da296d80871854d37bce9c6e7c12f7b2f3eddef6353b93b56ad2b0fa` |
| validation selection | `14214486fd600ff6625666b0a1e09707887547aa8d0f46dbe4e44d25cef79146` |
| selection receipt | `3cf5656c299dfd71ef7cf6eec8fb7f48859a1486cd7502db8844a13c814076b6` |
| independent reload report | `719b0e1ba2b649aaab74a23bb92ed8e59103633700dbd49884d8e9af2fc02cb4` |

독립 evaluator의 압축 해제 prediction content SHA-256은 원 training validation output과 같은
`b6ca57c0f3e63946c31eb4be45ee88620543fcd3c0bddf386bd73f21ba8bba60`이다.

기계 판독 요약은 [`summary.json`](summary.json)이다.
