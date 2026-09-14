# 독립 일반 renderer CPU 신규 설치 — 기술 검증 완료

판정: **CPU_INSTALL_TECHNICAL_PASS**. 설치 재현과 작은 정적 일반 장면에 한정한다.
설치 당시 전체 회귀의 실패는 `summary.json`에 보존했다. V1 테스트 충돌은 이후 별도 수정으로
해소됐다([후속 확인](../../docs/repository_followup_2026-09-12.md)). R4/R4b FAIL은 유지한다.
[계획/고정 완료 기준](../../docs/renderer_cpu_install_plan_2026-09-12.md) ·
[신규 설치 명령](../../docs/renderer_cpu_quickstart_2026-09-12.md) · [요약 JSON](summary.json).

## 실패를 먼저 구분했다

| 시도 | 측정 결과 | 처리 |
|---|---|---|
| PyPI urdfpy 0.0.22, 새 venv | 설치와 pip check 성공; 기존 146 tests 중 errors 8 | 원기둥 `len(None)` 오류 확인, 채택 안 함 |
| 같은 버전의 upstream VCS 설치 | pip exit 0, 실제 install 목록은 비어 있음 | no-op으로 판정; 성공으로 보고하지 않음 |
| 진단 venv에서 source 강제 교체 | 기존 146 tests 모두 PASS | 수정된 진단 환경이지 신규 설치 성공은 아님 |
| 두 번째 새 venv에서 profile 설치 | 다운로드 read timeout, 설치 시작 전 중단 | pip/setuptools만 남았음을 확인하고 동일 profile 재시도 |
| 같은 profile 재시도 | pip check PASS; 기존 146 + 패키징 6 = 152 tests PASS, skip 0 | 신규 설치 검증 완료 |

공개 wheel SHA는 `ec9d944bd8a28fe0914b6038e8119ebd53905bc4c507744e35b46a3fa02529bd`.
버전 문자열만 같고 구현은 달랐다. 정상 source는 upstream
`5466842899b33bd549e8f9e2a9a987bd5e37373b`이며 설치 후 `direct_url.json`에서 이 commit을
확인했다. 기존 simulator conda나 로컬 urdfpy checkout은 수정하지 않았다.

`pypi_test_final_output.txt`는 실패한 테스트 프로세스의 **마지막 출력 청크**로, 설치 전체 로그가
아니다. `source_install_noop.json`은 install 배열이 빈 원본 pip report다. 네트워크 실패의 전체
로그는 이 폴더에 저장하지 않았고 관찰한 오류/단계만 summary에 기록했다.

## 실제 CPU 렌더링

깨끗한 별도 checkout `cdc8113652008b0b856f056601c13ca4e25e6632`에서 실행했다.
renderer의 모든 source SHA가 receipt에 있으며 source code는 이미 커밋돼 있었다.
설치 profile은 실행 당시 후보 파일의 SHA로 식별했다:
`936b7c1de8e65e2817b3b36502dcd1c5f28caf2a9ab733debb3e01388823f93f`.
profile의 Git 보관 commit과 renderer 실행 source commit을 혼동하지 않는다.

- Linux x86-64, Python 3.8.20, Torch 2.4.1+cpu, Warp 1.0.0.
- 일반 상자 fixture, scene 1, 160×120, seed 0, 독립 프로세스 2개 × 각 2프레임.
- `smoke1/`, `smoke2/`에 request/receipt 및 NPZ 4개를 함께 추적한다. 사용자 실사 데이터가 아니다.
- 프레임당 8개 배열의 SHA가 두 실행에서 일치하고 파일 SHA·dtype·shape·decoded SHA를 확인했다.
- receipt는 `RENDERED_UNASSESSED`, 실험 판정은 `NOT_EVALUATED`, 학습은 `NOT_SUPPORTED`다.
  `CPU_INSTALL_TECHNICAL_PASS`는 이 설치 점검의 판정이며 renderer 실험 결과를 변경한 것이 아니다.

Warp 초기화에서 CUDA 장치 미발견 메시지가 있었지만 실행 장치는 CPU이고 정상 종료했다.
CPU JIT cache가 사용될 수 있으며, 신규 환경 성공을 cache-cold latency 측정으로 읽지 않는다.
진단 venv의 별도 smoke 두 번과 두 번째 venv의 사전 smoke 한 번도 수행했으며, 공개하는
비교 원자료는 위 `smoke1/2` 두 실행이다. GPU 결과와의 byte 동일성은 검사하지 않았다.

## 환경·출처·재현 검사

`environment.json`은 실행 프로세스에서 수집했다. user-site 비활성화, 모든 distribution의 venv
내 설치 경로, Torch CUDA metadata `null`, urdfpy source commit/비-editable 설치를 확인했다.
25개 runtime 배포 + pip/setuptools = 27개가 기록돼 있다.
`install_sources.json`은 최종 pip report에서 package 이름·버전·다운로드 출처/hash를 추린 것이다.
긴 description은 생략했으므로 원본 pip report 자체가 아니다. OS/build 의존성을 잠근 lockfile도 아니다.

NumPy가 있는 환경에서 저장소 루트 기준:

```bash
python -B -m unittest discover -s tests -p 'test_cpu_install_evidence.py' -v
```

3개 테스트가 profile SHA/설치 출처, 네 NPZ의 파일·배열 hash, 두 실행의 배열 일치 및 historical
Git source SHA를 검증한다. shallow clone에 실행 source object가 없으면 실패하는 것이 맞다.

## 설치 당시 전체 회귀 — 보존 기록

설치 검증 당시 전체 기존 환경 회귀는 **1,533 tests, 실패 1, errors 0, skip 4**였다. 실패는 기존
`test_it_is_still_one_base_link`이며 자산/검출기/정책/제어 코드나 그 테스트를 수정하지 않았다.
이 기록으로 전체 PASS, V1 통합 완료, shortcut 감소, GPU 처리량, 실제 배치를 주장하지 않는다.
현재 raw 결과·사전등록·기존 학습 환경을 덮어쓰지 않았고 push는 수행하지 않았다.

후속: `5cea0e4`에서 테스트가 수정됐으며 `ef59832`의 새 회귀는 실패·오류 없이 끝났다.
별도 [followup.json](followup.json)이 검사 출처와 수를 기록한다. 당시 `summary.json`을
덮어쓰지 않았으며 CPU smoke의 실행 source도 계속 `cdc8113`이다.
