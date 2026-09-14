# VOID — A8 평가 2차 시도, 4/10 cell에서 중단 (코드 결함)

2026-09-05 18:22. **판정·수치 없음. S_off·S_riskcap·S_dwa_arc 결과는 인용하지 않는다.**

## 무슨 일

소스 정책 3 cell은 통과했고 **재적응 체크포인트 첫 cell(T0_off)에서 복원이 실패**했다:

```
navrl_task.py:4300 in set_env_state
ValueError: could not convert string to float: 'off'
```

`cfg_search_state`(값 `"off"`, 모드 문자열)가 `abs(float(saved) - current)`로 비교하는
config-drift 루프에 들어 있었다. 옛 체크포인트에는 이 키가 없어 `saved is None`으로 건너뛰었기
때문에 지금까지 아무도 걸리지 않았다.

**이건 A8 고유 문제가 아니다.** `cfg_search_state`가 생긴 뒤 학습한 체크포인트는 **전부 평가
불가**였다. A8의 세 arm이 그 조건을 만족하는 첫 정책이라 처음 드러났을 뿐이다.

## 수정

`cfg_search_state`와 `cfg_search_state_force_invalid`를 float 루프에서 빼내 값 비교 루프로 옮겼다.
회귀 테스트 `tests/test_navrl_env_state_guard_types.py`는 개별 키가 아니라 **불변식**을 검사한다:
float 비교 루프의 어떤 항목도 `str(...)`나 문자열 리터럴을 current 값으로 줄 수 없다.
수정 전 코드에서 2건 실패, 수정 후 통과를 확인했다.

## 왜 부분 재개가 아닌가

수정이 런타임 소스(`navrl_task.py`)를 바꾸므로 완료된 3 cell과 나머지 7 cell의
`runtime_source_manifest_sha256`가 달라진다. 요약기 M3가 루트 내 단일성을 요구하므로 섞인 루트는
판정 불가다. 사전등록 §7의 부분 재개 금지가 그대로 적용된다.

## 보존·비용

학습 3 arm(ep2900) 무손실. 잃은 것은 cell 3개, 약 19분.
