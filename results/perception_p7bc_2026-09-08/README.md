# P7b/P7c validation 결과

상태: COMPLETE — corrected P7c v2 selected; test 미실행

사전등록 코드 `0a4cb77`, 좌표 버그 정정 코드 `3ce7d8d`.
동일 NPS train 13,510 frames/35 clips, validation 2,296 frames/7 clips를 사용했다.

| arm | hit | false lock / selected | no-lock | utility |
|---|---:|---:|---:|---:|
| 기존 P7 | 0.80880 | 0.14424 | 126 | 0.67247 |
| P7b 후보 보존 | 0.81751 | 0.14448 | 102 | 0.67944 |
| P7c motion/GMC (수정 v2) | 0.80488 | 0.12912 | 174 | 0.68554 |

고정 utility로 P7c v2를 선택했다. 기존 P7 대비 hit −9, proxy false lock −39,
utility +0.01307이다. 평균 재획득 시간은 0.32391→0.47033초, 최대는
5.20868→19.93322초로 악화했다. validation 7 clips의 선택 결과이며 보편적 우월성은 아니다.

첫 P7c v1 utility 0.68118과 이전 selection은 축소/원본 좌표 혼용 오류로 무효다.
원 산출물은 감사용으로 보존했으며 v2만 현재 선택 근거다. 정정 전에 validation을
관찰했다는 사실을 유지한다. seed/architecture/optimizer/선택 규칙은 변경하지 않았다.

P7c v2 checkpoint SHA-256: `9b1f52299eea83f89ed95e8bc7bcf26c0b51140d388887a83937486cc6113cca`.
독립 재로드 prediction content SHA-256: `769ff55a3ff53497380c5865a3486f4fa7f886324ee3e30124dbc3dcdb51a8cd`.
원본 prediction과 압축 해제 bytes가 동일했다. 상세 지표·경로·receipt는 [summary.json](summary.json).

latency의 batched tensor 수치는 optical flow/GMC·detector·이미지 로딩을 제외한다.
전체 온라인 latency와 실기 성능은 측정되지 않았다. 실제 ID switch/FTLR/ReID는
track ID 부재로 미측정이다. P7d 착수 조건은
[identity boundary](../../docs/plans/perception_p7d_identity_boundary_2026-09-08.md)에 정리했다.

검증: 전체 Python 1,219 tests OK (4 skipped), 축소 좌표 회귀 테스트와 checkpoint 재로드 PASS.
