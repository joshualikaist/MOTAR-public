# 이 문서는 이동했습니다 → [`docs/archive/reference_platform_proposal_2026-08.md`](archive/reference_platform_proposal_2026-08.md)

이 리다이렉트는 지워도 되는 편의 파일이 아니다.

`aerial_gym/config/robot_config/navrl_ref5in_quad_config.py`의 docstring이 이 경로를 가리키는데,
그 파일은 frozen ref5in 체크포인트에 대해 **provenance-frozen**이다 — 바이트가 하나라도 바뀌면
`eval_navrl_v2_density_sweep.sh:240-242`가 `robot config source drift`로 무조건 `exit 2`를 내고,
`NAVRL_V2_FORCE`로도 우회할 수 없다.

2026-08-20 커밋 `921fb1d`가 바로 그 docstring의 경로를 `docs/archive/`로 갱신했고, 그 결과 8개
브랜치 전부에서 ref5in 체크포인트의 byte-exact 평가가 막혔다. 2026-08-21에 그 한 줄을 되돌리고
대신 이 스텁을 두어, 참조를 참으로 유지하면서 해시되는 바이트는 건드리지 않도록 했다.

**따라서 이 파일을 지우고 소스의 경로를 고치는 방향으로 "정리"하지 말 것.** 잠금은
`tests/test_navrl_ref5in_provenance_freeze.py`가 걸고 있다.
