# 2026-09-12 후속 확인 — 테스트 충돌 해소와 남은 근거를 분리

## 확인한 상태

검사 시작 source는 `ef598328c5a2edb54080e13124a9208d7ed9c9c7`, 작업 트리는 clean이었다.
로컬 `origin/main` 참조 `eaa7f73`보다 15 commits 앞섰다. 원격 push는 하지 않았다.

- `5cea0e4`: V1 테스트만 수정됐다. 링크 개수 대신 물리 속성 보존을 검사하고 부품 위치
  검사에 joint 원점을 반영했다. 이번 후속 작업에서는 이 테스트나 자산을 변경하지 않았다.
- `ef59832`: 독립 CPU 설치 profile·검사·실제 상자 렌더링 근거를 보관했다.
  urdfpy의 버전 번호만으로 구현 동일성을 주장하던 설명을 철회했다.

| 이번 재검사 | 실제 결과 | 범위 |
|---|---|---|
| 기존 전체 회귀, aerialgym Python 3.8 | 1,533개 실행, 실패 0, 오류 0, skip 4; 37.416초 | 1,529개 통과 + 4개 건너뜀; 새 simulator 실험 아님 |
| 독립 CPU renderer 계약 | 152개 통과, skip 0; 3.012초 | 기존 146 + 설치 계약 6 |
| 보관된 CPU 설치·렌더링 근거 | 3개 통과, skip 0; 0.060초 | 출처·NPZ·decoded 배열 SHA·두 실행 일치 |
| Node 사이트 검사 | exit 0, PASS | 정적 사이트 계약·링크 |

전체 회귀에는 의도적으로 오류를 유발하는 테스트의 traceback과 기존 dependency 경고가
출력된다. 최종 unittest 결과는 `OK (skipped=4)`이고 프로세스 exit code는 0이었다.
위 수는 문서 후속 수정 **전의 깨끗한 ef59832**를 식별한다. 후속 문서 검사 추가분과 혼동하지 않는다.
기계 판독 기록은 [followup.json](../results/renderer_cpu_install_2026-09-12/followup.json)이다.

후속 문서 검사 2개를 추가한 작업 트리에서도 전체 회귀를 다시 실행했다: **1,535개 실행,
실패·오류 0, skip 4**, 37.898초. 이 후속 실행은 미커밋 변경 위에서 수행했으며 위 clean
ef59832의 1,533개 기록과 구분한다. 발표 자료 11개·문서 주장 12개·Node 사이트 검사도 통과했다.

## 해소한 것과 해소하지 않은 것

현재 테스트 상태는 **TEST_CONTRACT_RECONCILED**다. 과거 CONTRACT_MISMATCH는 실제로
발생했고 그때의 실패 수와 [설치 summary](../results/renderer_cpu_install_2026-09-12/summary.json)는
그대로 보존한다. README·VERIFICATION·사이트가 같은 현재 상태를 가리키도록 정정했다.

테스트가 통과했다는 사실만으로 일반적인 모든 joint 회전·임의 자산에 대한 기하 검증을
인증하지 않는다. 보고된 5 mm 변조 검사는 이번 후속 점검에서 별도로 재실행하지 않았다.
보관된 3,840쌍의 probe는 계속 **PARTIAL_EVIDENCE**다. 영상·전체 궤적·실행 출처의 누락을
이번 테스트 결과로 메울 수 없다. R4/R4b의 FAIL과 인지 shortcut 감소 미측정도 유지한다.

## 새 경로의 비용은 미측정 — INTEGRATED_RENDER_COST_UNMEASURED

인용된 기하 패스 3.09 / 9.14 ms는
[독립 URDF smoke의 기존 장면 비교](../results/renderer_urdf_smoke_2026-09-11/README.md)다.
R5의 음영 추가 비용과도 측정 대상이 다르다. 이것만으로 다른 렌더링 경로에 붙는 추가 비용을
“몇 ms”라고 확정하거나 삼각형 개수로 보간할 수 없다. 전체 step FPS와 VRAM도 미측정이다.
기존 원자료를 변경하거나 새 성능 추정치를 만들어 넣지 않았다.

## 작업 경계와 다음 순서

이번 작업은 문서 정합성·설치·독립 일반 상자 렌더러의 재현성으로 제한한다. 요격 시스템의
표적 검출 개선을 준비하는 커널은 격리 prototype을 포함해 구현하지 않았다. detector 연결,
충돌·정책·제어 코드, 실기 동작도 변경하지 않았다.

안전하게 이어갈 수 있는 저장소 작업은 다음 순서다.

1. 이 후속 변경의 문서·근거 회귀 및 diff를 확인한다. 과거 결과는 덮어쓰지 않는다.
2. 독립 CPU quickstart를 공개 입구로 명확히 분리하고, 남은 배포 체크리스트를 정리한다.
3. 이미지 재배포 고지·인용 metadata·공개 이력의 미해결 사항을 별도 검토한다.
   이 확인이나 원격 배포가 완료됐다고 현재 테스트 성공으로 대신 주장하지 않는다.

## 재검사 명령

저장소 루트에서 각 환경의 interpreter를 사용한다. 설치 방법은
[독립 CPU quickstart](renderer_cpu_quickstart_2026-09-12.md)를 따른다.

```bash
# 기존 전체 회귀: historical aerialgym 환경에서만
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python -B -m unittest discover -s tests -q

# 별도 CPU profile 환경에서: simulator 실행 없음
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -p 'test_renderer*.py' -q
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' python -B -m unittest discover -s tests -p 'test_cpu_install_evidence.py' -q
node tests/test_status_site.js
```
