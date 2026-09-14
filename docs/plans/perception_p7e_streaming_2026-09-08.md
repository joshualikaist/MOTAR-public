# P7e crop verifier 및 streaming 실행 계약

상태: P7e COMPLETE — NOT SELECTED; streaming IMPLEMENTED, cached replay PASS,
historical RGB rank parity FAIL (5/2,296). 아래 실험 규칙은 결과 관찰 전 고정했다.

결과: [평가·한계](../../results/perception_p7e_streaming_2026-09-08/README.md),
[streaming API](../specs/perception_streaming_v1.md).

P7e는 frozen detector의 현재 Top-5 crop을 UAV-validity로 분류하는 별도 비교군이다.
ReID가 아니며 기존 P7c checkpoint의 histogram 입력은 유지한다. 학습된 embedding을
기존 checkpoint에 바로 교체하면 feature 의미가 달라지므로 이번에는 verifier를 별도
선택기로 평가한다. 개선되면 스트리밍 선택기로 채택하며 아니면 P7c v2를 유지한다.

원 RGB candidate box의 1.25배 context를 32x32 RGB로 변환한다. 3-layer CNN
(16,32,64 channels, stride 2), global average pooling, 64D embedding, validity head.
IoU>=0.3 positive, IoU<0.1 negative, 중간 후보는 loss에서 제외한다.
seed 17, AdamW lr 1e-3, weight_decay 1e-4, batch 256, 8 epochs.
35 train clips를 이름 SHA-256 순서로 정렬해 index modulo 3으로 3 folds를 만든다.
각 fold의 최소 BCE epoch (동점은 이른 epoch)를 선택하고 그 epoch 수의 중앙값으로
전체 train을 fresh 학습한다. 외부 validation은 최종 1회 평가에만 사용한다.
max sigmoid>=0.5면 최고 후보, 아니면 NO_LOCK. 기존 utility가 더 높으면 P7c 유지,
동점도 P7c 유지. threshold tuning과 test 열람은 하지 않는다.

스트리밍 경로: BGR frame + sequence/frame/timestamp → frozen tiled detector →
candidate descriptor → online backward flow/GMC → clip-local T16 buffer → selector.
선택된 verifier가 있다면 별도 crop branch를 사용한다. detector와 selector hash를 검사한다.
sequence 전환 시 buffer를 비우며 비단조 timestamp와 geometry 변화를 거부한다.
GT는 metric evaluator에만 전달하며 online selector에는 제공하지 않는다.
cached-candidate replay는 offline logits와 tolerance 1e-5 및 rank 일치를 검사한다.
전체 validation RGB에서 batch-one 처리 시간의 mean/p50/p95와 FPS를 측정한다.
이미지 decode, detector, motion, selector 및 합계 비용을 구분한다.
실제 비행/카메라 입력 드라이버·PPO 연결·P8/P9는 이번 범위 밖이다.
