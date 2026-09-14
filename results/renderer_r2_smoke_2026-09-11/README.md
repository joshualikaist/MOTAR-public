# R2 실제 Warp/GPU 기술 smoke — 비정식

2026-09-11 RTX 3070에서 독립 renderer를 처음 실행했다. seed 0, scene 1개, 160×120,
2개 정적 frame이다. 정상 상태 `RENDERED_UNASSESSED`; 이 결과는 R3 사전등록 전에 관측했고
실행 당시 source tree가 dirty였으므로 정식 실험 판정에는 사용하지 않는다.

두 별도 프로세스의 배열 hash가 전부 일치했고, 각 frame은 유효 픽셀 4,098개와 두 instance를 담았다.
`frame_*.npz`는 RGB와 debug buffer의 실제 산출물이고 request/receipt는 입력·runtime·hash를 기록한다.
정식 판정은 [R3 비교 결과](../renderer_r3_seed173_comparison_2026-09-11/README.md)를 사용한다.
