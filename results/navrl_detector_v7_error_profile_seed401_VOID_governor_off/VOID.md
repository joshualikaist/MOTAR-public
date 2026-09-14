# VOID — 사전등록 계약 위반 (governor off)

폐기 사유: 평가기를 직접 호출하면서 `NAVRL_SPEED_GOVERNOR=riskcap` 지정을 빠뜨렸다.
결과 JSON의 `condition.speed_governor_mode = "off"`가 이를 증명한다.

사전등록(`docs/prereg_2026-08-13_detector_coupling.md` §3)은 riskcap을 계약으로 고정했다.
governor가 꺼지면 궤적 분포가 달라지고(capture 71.50% vs riskcap 계약의 ~80%대), 프로파일링은
정의상 "정책이 방문하는 상태 위에서의 검출기 오차"를 재므로 이 자료는 계약에 해당하지 않는다.

분석에 사용하지 않았다. 시드 401은 소진 처리하고 프로파일링을 **시드 419**로 재실행한다.
재발 방지: 계약을 ad-hoc 환경변수로 넘기지 않고 런처 스크립트
`eval_navrl_v2_detector_coupling.sh` 한 곳에서 export 한다.
