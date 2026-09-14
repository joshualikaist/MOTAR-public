# Candidate validity 학습 비교 계약

상태: COMPLETE — NOT SELECTED. utility 0.683362 vs corrected P7c 0.685540.
hit +25 frames, false lock +30 frames로 기존 선택을 유지한다.
[결과·receipt](../../results/perception_p7bc_2026-09-08/validity_loss_result.json).

진단: Top-5 oracle hit 2071/2296=90.20%; P7c의 정답 후보가 있는 abstention 86,
잘못 선택 137, 후보 자체가 없는 frame 225. oracle은 GT 기반 상한으로 실현 성능이 아니다.
NPS refined annotation에 ID 열은 있지만 동봉 renderer는 raw annotation에 ID가 없다고
명시한다. refined ID 생성 출처를 검증하기 전 실제 identity supervision으로 취급하지 않는다.

다음 우선순위는 learned crop appearance와 검증된 track-ID 데이터다.
MOT-FLY 공식 저장소 https://github.com/CZC-123/MOT-FLY 는 MOTChallenge 형식,
Apache 2.0 및 Google Drive/Baidu 경로를 제공한다. 이번 단계에서는 다운로드하지 않았다.
고정 source-sequence split과 identity 의미를 감사한 후 별도 학습 계약을 고정한다.

Validation 진단 후 설계한 후속 개발 실험이다. test는 사용하지 않는다.
기존 corrected P7c v2와 architecture, 후보, motion v2 cache, seed 17,
batch 256, AdamW, 최대 40 epoch, patience 8을 동일하게 고정한다.
기존 one-best CE 대신 IoU>=0.3인 모든 후보를 positive로 하는 masked BCE를 사용한다.
현재 후보 중 최대 validity logit이 0 이상이면 그 후보를 고르고, 아니면 NO_LOCK이다.
학습하지 않는 no-lock head 대신 고정 logit 0을 사용한다. 동점은 candidate 우선이다.
epoch는 validation BCE 최소, arm은 기존 utility 최대; 동점은 기존 P7c를 유지한다.
출력 probability는 independent sigmoid이며 합이 1인 분포가 아니다.
기존 report cross_entropy 필드는 BCE arm에서는 binary CE이며 loss_kind로 구분한다.
기존과 같은 7 validation clips를 재사용하므로 confirmatory 성능 주장은 하지 않는다.
