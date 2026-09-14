# 독립 일반 렌더러 CPU 설치 검증 계획

2026-09-12. 출발 commit `cdc8113`. 대상은 `tools/renderer_validation`의 독립 일반 geometry/
shading 도구이며 detector·정책·제어·자산 통합 코드는 변경하지 않는다.
이 문서는 패키징 기술 점검 기준이지 기존 R4/R4b 실험의 재사전등록이 아니다.

## 먼저 알고 있는 실패 위험

- 계획 작성 당시 기존 전체 테스트의 V1 one-link 계약 실패는 별개였다. CPU 설치 성공으로
  해소하지 않는다. 이후 별도 테스트 수정 및 [후속 검사](repository_followup_2026-09-12.md)로
  충돌이 해소됐으며 아래 설치 완료 기준과 당시 실패 receipt는 변경하지 않았다.
- Python 3.8.20 / NumPy 1.23.0 / networkx 2.2를 사용한 기존 환경과 최신 패키지를 섞으면
  삭제된 API 때문에 실패할 수 있다. 기존 환경은 변경하지 않는다.
- 기존 urdfpy는 `0.0.22`라는 버전만 같을 뿐 PyPI artifact와 동일성이 확인되지 않은
  upstream checkout `5466842899b33bd549e8f9e2a9a987bd5e37373b`다.
- 앞선 pip 23 + CPU-only index 설치는 의존성 해석/build 단계에서 실패했다.
  pip 25.0.1과 공식 CPU Torch wheel 직접 URL을 사용하고 일반 의존성은 PyPI에서 해결한다.
- CPU unit tests는 Warp를 mock하므로 이것만 통과해도 실제 렌더링 성공은 아니다.

## 순서와 고정 완료 기준

1. **설치 조사:** 별도 venv(`--system-site-packages` 없음), `PYTHONNOUSERSITE=1`.
   먼저 공개 urdfpy artifact의 동작을 확인한다. 실패하면 실패를 기록하고 검증된 upstream
   source pin을 후보로 삼는다. 버전 문자열만으로 package 교체를 인증하지 않는다.
2. **별도 설치 명세:** root `setup.py`나 root requirements를 재사용하지 않는다.
   Linux x86-64 / Python 3.8 전용 CPU profile과 정확한 의존성/배포 출처를 기록한다.
   `pip check`가 0이어야 하며 Torch의 CUDA build metadata는 `None`이어야 한다.
   urdfpy의 `direct_url.json`에서 정확한 Git commit과 비-editable 설치를 확인한다.
   같은 버전이면 VCS 설치 명령도 no-op일 수 있으므로 exit 0만으로 인정하지 않는다.
3. **기존 계약 검사:** 기존 `test_renderer*.py` 146개를 전부 실행해 실패·오류·skip 0을 요구한다.
   새 패키징 테스트는 별도 집계한다. 기존 테스트 삭제/skip으로 성공을 만들지 않는다.
4. **실제 일반 상자 smoke:** `run_renderer_validation.py --device cpu`, scene 1개,
   160×120, 2프레임, seed 0을 서로 다른 output으로 두 번 실행한다.
   두 실행 모두 `RENDERED_UNASSESSED`, 모든 저장 배열의 hash 일치, 파일 hash 검증을 요구한다.
   `NOT_EVALUATED`/`NOT_RUN`/`NOT_SUPPORTED` 상태를 성능 PASS로 바꾸지 않는다.
5. **재설치 확인:** 확정한 설치 파일로 두 번째 깨끗한 venv를 구성해 `pip check`와 기존
   146개 검사를 반복한다. 실제 smoke도 해당 환경에서 확인한다.
   실패 후 수정한 첫 진단 환경을 신규 설치 성공으로 부르지 않는다. VCS 설치에 필요한
   `git`도 Python 패키지와 별개인 시스템 전제조건으로 기록한다.
6. **기록:** 설치 source/pip report, runtime 및 패키지 경로, 테스트 수, smoke receipt/hash,
   실패 시도, 재현 명령과 한계를 새 결과 디렉터리에 보존한다. 기존 결과는 덮어쓰지 않는다.

패키징 파일과 새 테스트는 `apply_patch`로 편집하고 diff 검토 후 논리적으로 커밋한다.
작업 중 source commit이 바뀌면 실행 receipt의 실제 commit을 따로 기록한다.
위 단계가 실패하면 다음 단계의 성공을 예측해 보고하지 않고 실패 지점을 알린다.
기존 GPU 수치/전체 simulator 설치 재현/shortcut 감소/실제 시스템 성능은 완료 기준 밖이다.
