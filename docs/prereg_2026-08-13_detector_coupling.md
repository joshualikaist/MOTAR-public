# 이동됨 — `docs/archive/prereg_2026-08-13_detector_coupling.md`

이 파일은 **경로 스텁**이다. 내용은 아카이브에 있다:
[`docs/archive/prereg_2026-08-13_detector_coupling.md`](archive/prereg_2026-08-13_detector_coupling.md).

## 왜 지우면 안 되는가

소스 5곳이 **이 경로**(archive가 아닌)를 사전등록 출처로 인용한다:

- `aerial_gym/config/task_config/navrl_task_config.py:329`
- `aerial_gym/task/navrl_task/navrl_perception.py:825`
- `aerial_gym/rl_training/rl_games/eval_navrl_v2_detector_coupling.sh:5`
- `tests/test_navrl_detector_noise.py:3`
- `tools/analyse_navrl_detector_coupling.py:4`

2026-08-20 문서 통합 때 원본이 `docs/archive/`로 옮겨졌으나 소스의 경로 문자열은 갱신되지 않아,
그 시점부터 다섯 인용이 전부 존재하지 않는 파일을 가리키고 있었다(2026-09-05 발견).
경로를 소스에서 고치는 대신 스텁을 두는 이유는
`docs/reference_platform_proposal_2026-08.md`와 같다 — 그 소스 파일들 중 일부가
provenance-frozen이거나 영수증 해시에 묶여 있어 바이트 변경 비용이 크다.
