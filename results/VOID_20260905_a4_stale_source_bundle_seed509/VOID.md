# VOID — A4 seed 509, 소스 번들이 수정 전 셸 스크립트를 고정

**무효 사유**: 첫 A4 시도가 `eval_navrl_v2_density_sweep.sh`의 **수정 전** 버전으로 공유 소스
번들을 생성했다. 그 시도는 `omni`에서 `exit 2`로 죽었다(셸의 별도 모드 화이트리스트가 새 모드를
거부). 스크립트를 고치고 재개하자 이번엔 평가기가
`runtime source changed during evaluation: eval_navrl_v2_density_sweep.sh`로 `exit 1` 했다 —
번들에 박힌 해시와 런타임 파일이 다르기 때문이다.

**완료된 arm**: off, fixed2p0, riskcap, stopcap, omni(산출은 났으나 검증 실패).

화이트리스트 변경은 `case` 분기에 두 항목과 오류 문구를 더한 것뿐이라 앞선 네 arm의 동작에
영향을 줄 수 없다. 그럼에도 채택하지 않는다:

1. 같은 날 A3에서 "실행 중 소스가 바뀌었으니 값이 같아도 VOID"라고 판단했다. 여기서 기준을
   완화하면 일관성이 없다.
2. 사전등록(`prereg_2026-09-05_a4_geometry_baselines.md` §2·M3)이 **6 arm 동일 시드·동일 소스
   번들**을 요구한다. 네 arm이 구 스크립트, 두 arm이 신 스크립트로 돌면 그 요구가 깨진다.

**교훈**: 공유 소스 번들은 **첫 arm**에서 고정된다. 스윕 도중 소스를 고치면 남은 arm만 재개할
수 없고 전체를 다시 돌려야 한다. 새 모드를 추가할 때는 파이썬과 셸 화이트리스트를 **함께**
고친 뒤 스윕을 시작한다 — `tests/test_navrl_speed_governor.py::WhitelistsAgree`가 이제 그것을
강제한다.

버려진 수치(참고용, 채택 금지): off 21.13 % / fixed2p0 15.67 % / riskcap 18.77 % /
stopcap 12.30 % crash.
