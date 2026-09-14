# 공개 근거의 CPU 재현 점검 — 2026-09-12

> 후속 갱신: 아래 NumPy-only 검사 이후 독립 renderer의 전체 CPU 신규 설치와 실제 일반 장면
> smoke도 별도로 완료했다. [설치 결과](../results/renderer_cpu_install_2026-09-12/README.md).
> 아래 미완료 목록은 첫 점검 시점 기록이다. V1 테스트 충돌도 이후 별도 수정 `5cea0e4`와
> [후속 검사](repository_followup_2026-09-12.md)로 해소됐다. GPU/전체 simulator 실행 검증은
> 별개이며, 설치·테스트 성공으로 전체 연구 gate를 PASS로 바꾸지 않는다.

## 완료 범위

검증 checkout은 `ce7d191`이다. 원 저장소를 `git clone --local --no-hardlinks`로 새 디렉터리에
복제했다. 기존 작업 디렉터리의 ignored NPZ·source_snapshot·데이터셋을 복사하지 않았다.
이는 **로컬 커밋의 독립 checkout 검증**이며, 아직 push하지 않았으므로 원격 배포 검증은 아니다.

새 venv는 Python 3.8.20, NumPy 1.23.0이며 `--system-site-packages`를 쓰지 않았다.
Torch와 Isaac Gym은 이 환경에서 import 대상으로 검색되지 않았다.

| 검사 | 결과 |
|---|---|
| 보관된 appearance JSON 기술통계 재계산 | 6 tests PASS |
| 독립 합성 배경 receipt/배열 비교 테스트 | 24 tests PASS |
| 실제 run1/run2 비교 CLI | exit 0, `PASS`, evidence_verified `[true, true]` |
| Node 사이트 계약/로컬 링크 검사 | PASS |
| 검사 후 checkout | clean |

실제 비교 CLI는 NPZ archive SHA, 디코딩한 14개 배열 SHA, historical Git object의 source
14개 SHA를 확인한다. source commit `2a497f87601cddab687649db504097001b4d8b03`가 필요하므로
shallow clone에 해당 object가 없으면 실패하는 것이 맞다.

## 재현 명령

검증된 커밋과 source 이력이 있는 full checkout의 저장소 루트에서 실행한다.
아래는 **CPU 근거 검사 전용**이며 전체 renderer/시뮬레이터 설치 지침이 아니다.

```bash
audit_venv=$(mktemp -d /tmp/motar-evidence-venv-XXXXXX)
python3.8 -m venv "$audit_venv"
"$audit_venv/bin/python" -m pip install --no-cache-dir numpy==1.23.0

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' "$audit_venv/bin/python" -B \
  -m unittest discover -s tests -p 'test_appearance_report_evidence.py' -v
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' "$audit_venv/bin/python" -B \
  -m unittest discover -s tests -p 'test_renderer_background_comparison.py' -v
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' "$audit_venv/bin/python" -B \
  tools/compare_renderer_background_validation.py \
  --run results/renderer_background_v1_2026-09-12/run1/run.json \
  --run results/renderer_background_v1_2026-09-12/run2/run.json
node tests/test_status_site.js
```

비교 CLI의 `PASS`는 저장된 기술 근거의 무결성·일치이다. R4/R4b의 통계적 FAIL을 뒤집지 않으며,
새 GPU 렌더링, shortcut 감소, detector/정책 성능을 입증하지 않는다.

## 첫 점검 당시 미완료 목록 — 현재 상태는 상단 후속 기록 참조

- **V1 CONTRACT_MISMATCH:** 13-link 자산과 one-link 계약/테스트 불일치. 기존 전체 환경에서
  최종 스위트는 1,524 tests / 1 failure / 4 skipped. 이 문서는 전체 PASS를 주장하지 않는다.
- **probe PARTIAL_EVIDENCE:** 보관된 픽셀 수/거리만 확인 가능하다. pose·영상·실행 provenance
  누락은 남아 있다. [감사](../results/target_appearance_in_sim_2026-09-12/AUDIT.md).
- 전체 renderer quickstart의 **새 환경 설치부터** 재현하는 검증은 미완료다. 기존 환경에서
  renderer tests 146개가 통과한 것과 이번 NumPy-only 근거 검사를 합쳐 그 완료로 쓰지 않는다.
  앞선 CPU Torch 설치 시도는 pip 의존성 해석/build 단계에서 실패했으며 여기서는 재시도하지 않았다.
- README 전면 구성 변경, 외부 인용·배포 라이선스의 남은 확인, 전체 Git history 공개 감사는
  이 변경에서 완료하지 않았다. 기존 WORKLOG의 체크리스트 전체가 완료됐다는 뜻이 아니다.
- 새로운 detector 연결·정책·제어·실제 배치는 이 점검 범위 밖이다. 캐시·사용자 데이터 삭제 및
  원격 push도 수행하지 않았다.

후속 문서/패키징 작업은 위 미완료 항목을 별도로 닫아야 한다. 사전등록이나 historical 결과의
FAIL을 변경해서 공개 준비 완료로 표시하지 않는다.
