# VOID — export 가드의 타입 버그로 toward 셀이 중단됨 (결과 미산출)

## 무슨 일이 있었나

seed 359 first-acquisition 진단의 첫 셀(toward)이 평가는 끝냈지만 export 직전에 자체 가드에서
중단됐다:

```
RuntimeError: NavRL first-acquisition outcome mismatch: [1932, 115, 2] != (1932, 115, 2)
```

값은 같다. `fa_outcomes`를 list로 만들고 tuple인 `expected[1:]`과 비교해서, 파이썬에서 list는
tuple과 절대 같지 않으므로 **카운트와 무관하게 항상 발화**하는 가드였다. 즉 데이터 문제가 아니라
내가 새로 쓴 검증 코드의 타입 버그다.

## 이 디렉터리에 무엇이 있고 없나

`205bars.json`은 **없다**. 가드가 export 이전에 발화했기 때문에 결과 JSON도 receipt도 쓰이지
않았다. 남은 것은 source_manifest / source_snapshot / checkpoint_snapshot / 실행 로그뿐이다.
away 셀은 **아예 실행되지 않았다**.

## 시드 재사용을 허용한 근거 (명시)

같은 seed 359로 재실행한다. 근거:

1. 결과가 export되지 않았으므로 어떤 셀의 요약도 존재하지 않는다.
2. 사전등록 primary screen은 **away**의 never-acquired rate 차이인데, away는 실행조차 되지
   않아 primary에 대한 정보가 전혀 노출되지 않았다.
3. 오류 메시지를 통해 내가 본 유일한 수치는 toward의 outcome 카운트 `1932/115/2`
   (= capture 94.29%)이며, 이는 seed 353 toward의 94.29%를 재현한 보조 지표다. 새로운 정보가
   아니고 판정에 쓰이지 않는다.
4. 실패 원인이 실험 계약이나 데이터가 아니라 assertion의 타입 비교였고, 그 버그를 고친 뒤
   회귀 테스트(`test_outcome_sum_guard_compares_like_with_like`)로 고정했다.

"결과를 보고 나서 재시도"가 아니라 "산출물이 없는 실행 실패의 재실행"이다. 판단 근거를 남기니
동의하지 않으면 새 시드로 재사전등록하면 된다.

## 고친 것

- `navrl_task.py`: `fa_outcomes`를 tuple로 생성.
- `tests/test_navrl_ref5in_run_contract.py`: 이 버그 유형을 고정하는 회귀 테스트 추가.
