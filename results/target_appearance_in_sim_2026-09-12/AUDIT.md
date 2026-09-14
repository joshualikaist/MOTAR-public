# 2026-09-12 독립 감사 — 저장된 수치 일치, 해석·출처는 제한

대상 커밋: `23e681897f1c8603db87b8eb919ea365ac41186e`.
새 simulator 실행, 학습, 제어 변경 없이 기존 JSON과 소스만 검사했다.
판정은 **PARTIAL_EVIDENCE**이며 새로운 성능 PASS가 아니다.

> 후속 확인: `5cea0e4`가 one-link 테스트를 수정했고 `ef59832`에서 전체 회귀 1,533개 실행,
> 실패·오류 0, skip 4를 재확인했다. 현재 테스트 상태는 **TEST_CONTRACT_RECONCILED**다.
> 아래 실패 수는 최초 감사 당시 기록이다. 저장된 probe의 근거 누락은 여전히 남아 있다.
> [후속 검사](../../docs/repository_followup_2026-09-12.md).

## 1. 독립 재계산

`raw.json` SHA-256:
`0589df29f51cace0ac7c756c29531f6f53021aaf739bf6793eac4d95046700a6`.

| 항목 | 재계산 결과 |
|---|---|
| 각 arm의 거리/픽셀 수 | 각각 3,840개, 전부 유한값 |
| 0픽셀 표본 | 두 arm 모두 0개; 비율 계산에서 제외한 표본 없음 |
| 동일 인덱스의 거리 차이 | 전부 0.0 m |
| 동일 인덱스의 픽셀 수 차이 | 전부 0 px |
| 픽셀 수 비 v3/v2 | 최솟값·Q1·중앙값·Q3·최댓값 모두 1.000 |
| 저장된 거리 범위 | 4.068412780761719–6.842845439910889 m |

| 거리 구간 (왼쪽 포함, 오른쪽 제외) | n | v2 중앙값 px | v3 중앙값 px |
|---|---:|---:|---:|
| [4, 5) m | 1,196 | 18.0 | 18.0 |
| [5, 6) m | 1,804 | 6.0 | 6.0 |
| [6, 7) m | 840 | 6.0 | 6.0 |

3,840은 저장된 관측 수이지 독립 실험 반복 수가 아니다. 신뢰구간·일반화·동등성 검정을
추가하지 않는다. 이 구간표는 탐색적 기술통계이며 사전등록된 평가가 아니다.

## 2. 보고에서 정정한 것

- **픽셀 수 동일 ≠ 마스크/영상 동일.** 원자료에는 마스크 위치나 RGB가 없다.
- **거리 동일 ≠ 전체 궤적 동일.** 3D pose, 속도, camera pose, frame/environment ID가 없다.
  저장 순서로 나란히 비교할 수 있지만 완전한 trajectory pairing을 검증할 수는 없다.
- **자산은 로드됐다는 보고와 계약 PASS가 다르다.** 현재 URDF는 13 links인데 기존 계약 및
  `test_it_is_still_one_base_link`는 1 link를 요구한다. 최신 HEAD 전체 회귀는
  **1,516개 실행, 1개 실패, 4개 skip**였다. 테스트 기준을 바꾸거나 자산을 수정하지 않았다.
  이번 문서/재계산 테스트 추가 후에는 **1,524개 실행, 같은 실패 1개, 4개 skip**이다.
- `warp_asset.py`는 visual mesh를 읽는다. "시뮬레이터는 collision만 읽는다"는 문장은
  특정 물리 기하 추출기 두 곳에만 해당한다.
- foreground mask의 analytic proxy와 독립 prototype mesh rendering을 구분한다.
  scene mesh 가림 검사는 남아 있으므로 모든 visual 변경의 무효를 일반화하지 않는다.
- 같은 색인 여러 물체는 색만으로 **지정된 한 물체의 identity**를 구별할 수 없다.
  foreground segmentation cue와 identity cue를 혼동하지 않는다.

## 3. 실행 출처의 한계

원자료의 각 arm에는 `appearance`, `ranges`, `pixels`만 있다. 실행 commit/dirty 상태,
runtime fingerprint, effective environment, seed, env 수, step 수, 시각은 기록되지 않았다.
`23e6818`은 자료가 들어간 **보관 커밋**이며 실행 source commit이라는 증거가 아니다.
첫 두 실패 실행의 원자료와 loader traceback도 이 폴더에는 없다. 소스의 mesh/link 인덱싱은
보고된 실패 설명과 부합하지만, 여기서 Isaac Gym 로드를 재실행해 인증한 것은 아니다.

보관된 probe는 seed를 task 생성/reset **뒤에** 설정하고 설정값에 `setdefault`를 사용한다.
따라서 CLI 기본값만으로 초기화 난수와 실제 실행 환경이 완전히 고정됐다고 말할 수 없다.
소스에 선언된 `RANGE_EDGES`/`MIN_SAMPLES_PER_BIN`은 출력 검증에 사용되지 않는다.
기록된 4–7 m 표는 별도 기술통계이며 probe가 최소 표본 수를 강제했다는 증거가 아니다.
누락 정보를 추정해 historical raw receipt에 채워 넣지 않는다.

## 4. CPU 재현 — 새 simulator 실행 불필요

저장소 루트에서 Python 표준 라이브러리만으로 실행한다.

```bash
python3 -B -m unittest discover -s tests -p 'test_appearance_report_evidence.py' -v
```

검사는 원자료 SHA, 길이·유한값·양수 픽셀 수, 거리/픽셀 수의 항목별 일치, 비율,
거리 구간별 n·중앙값을 재계산한다. 이 명령은 파일을 쓰지 않으며 GPU와 simulator를 import하지 않는다.

## 5. 최초 감사 시점의 후속 상태 — 현재 테스트 상태는 상단 참조

- README·사이트·VERIFICATION·계획의 0.587과 1.000 결과 계보를 분리했다.
- 독립 일반 renderer의 raw-array 공개 재현성과 비교 도구 검증은 별도 결과 폴더에서 다룬다.
- V1 자산 계약 불일치: **CONTRACT_MISMATCH**, 해결됐다고 표시하지 않는다.
- detector 입력 교체, 정책·제어 실험, 실제 배치: 이번 감사에서 실행하거나 구현하지 않았다.

원자료와 과거 실험의 판정 기준은 보존한다. 숫자가 같다는 이유로 빠진 출처와 해석 한계를
자동으로 해소하지 않는다.
