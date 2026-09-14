# R2 실제 Warp/GPU 기술 smoke 반복 — 비정식

첫 smoke와 같은 seed 0·scene·camera를 별도 Python 프로세스에서 반복했다.
두 frame의 모든 배열 hash가 첫 실행과 exact 일치했다. 실행 당시 source tree가 dirty였고
R3 사전등록 전 결과이므로 기술 smoke의 재현 확인에만 사용하며 정식 판정에는 포함하지 않는다.

정식 판정은 [R3 비교 결과](../renderer_r3_seed173_comparison_2026-09-11/README.md)를 사용한다.
