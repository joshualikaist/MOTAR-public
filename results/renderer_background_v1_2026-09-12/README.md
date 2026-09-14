# R4 background v1 기술 구현 및 검증

기존 R4/R4b의 `FAIL`을 덮어쓰지 않고, 누락됐던 일반 배경 geometry와 재질 intervention 계약을
별도 경로로 구현·검증했다. 이 결과는 일반 object renderer의 기술 계약만 말하며 detector,
tracking, association, PPO, shortcut 감소 또는 실제 시스템 성능을 주장하지 않는다.

## 구현 범위

- 절차적 floor, wall, 12각 column, 2-link hinged panel
- surface patch/material ID와 part/instance debug metadata
- 동일 geometry에서 material 0 albedo 변경, 모든 material 순환 변경, X축 조명 반전,
  instance debug ID 재번호화 intervention
- RGB 3채널/유한값/범위, geometry exact rerender, material·lighting·debug ID 격리 검사

상세 계약은 [renderer_background_v1_contract.md](../../docs/renderer_background_v1_contract.md),
구현은 [background_scene.py](../../tools/renderer_validation/background_scene.py)와
[background_validation.py](../../tools/renderer_validation/background_validation.py)에 있다.

## 결과

| 실행 | 결과 |
|---|---|
| run1 | `TECHNICAL_PASS` |
| run2 | `TECHNICAL_PASS` |
| 과거 comparison v1 | `PASS` — 당시 비교한 필드에 한정; 검증기 결함은 아래 교정 참고 |
| 기존 R4/R4b | `FAIL_UNCHANGED` |

두 실행의 `arrays.npz` SHA-256은
`bd4c7d4664b78d76683b1a89760ebdfeecf8427fe064a96ae8a08c3074736492`로 동일하다.
각 view의 모든 instance는 32개 이상 valid pixel, material 0은 64개 이상을 확보했다.
재질 변경·순환 intervention은 보이는 픽셀에서 실제 RGB 변화가 확인됐고, debug instance ID는 RGB를
바꾸지 않았다. 결과 receipt에는 입력, camera pose, runtime, Warp/driver, source-file snapshot,
배열 SHA가 들어 있다.

## 검증기 교정 — 2026-09-12

기존 comparison/summary.json은 보존한다. 그 v1 도구는 양쪽 필드가 누락되면 None == None으로
통과할 수 있었고 per_view와 일부 runtime 항목을 비교하지 않았다. 정상 receipt의 내용이
틀렸다는 뜻은 아니지만 당시 검증기 자체는 불완전했다.

v2는 필수 필드·형식·유한값·중복 JSON 키를 검사하고 per_view와 runtime 전체를 비교한다.
원 NPZ 파일과 디코딩한 각 배열의 SHA, 실행 당시 Git source object의 SHA도 검증한다.
두 정상 실행의 read-only v2 검증은 PASS다. 새 단위 테스트는 누락·변조·뷰 불일치·실패 gate를 검사한다.
이는 저장된 근거의 무결성·일치 검사이며 실험의 진정성이나 원래 통계 계산의 재실행을 보증하지 않는다.

실행 source commit은 두 receipt 모두 2a497f87601cddab687649db504097001b4d8b03이다.
저장소가 추적하는 것은 **run1/run.json, run2/run.json, 두 run의 arrays.npz,
comparison/summary.json, 이 README**다. NPZ는 각각 1,573,320 bytes이며 SHA가 같다.
처음에는 전역 `*.npz` ignore 때문에 누락됐고, 추적한다고 적었던 설명을 한 차례 정정했다.
후속 공개 재현성 교정에서 **이 두 합성 배경 배열만** 명시적으로 예외 처리해 포함했다.
데이터셋·비행 프레임 dump·체크포인트에 대한 전역 ignore는 유지한다.
receipt의 `arrays_npz_sha256`과 디코딩한 배열별 SHA를 모두 확인한다.
`source_snapshot`은 `results/**/source_snapshot` ignore 규칙으로 추적하지 않는다. receipt의 **14개** source 파일은 위 Git
commit에 존재하며 파일별 SHA로 검증하므로 full clone에서 해당 Git object로 재구성할 수 있다.
shallow clone으로 그 commit이 없다면 검증은 실패한다. historical receipt는 수정하지 않았다.

첫 구현 실행의 `TECHNICAL_FAIL`은 2026-09-11 결과에 보존한다. 원인은 CUDA float `mean()`의
1 ulp 오차로 all-changed 판정이 1.0보다 작아진 것이었고, 정수 개수 판정으로 수정한 뒤 두 실행을
새 디렉터리에서 완료했다.

## 재현

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/fair/miniconda3/envs/aerialgym/bin/python -B \
  tools/run_renderer_background_validation.py \
  --output results/renderer_background_v1_2026-09-12/new_run

python -B tools/compare_renderer_background_validation.py \
  --run results/renderer_background_v1_2026-09-12/run1/run.json \
  --run results/renderer_background_v1_2026-09-12/run2/run.json
```

비교 명령은 NumPy가 있는 환경에서 실행하며 기본적으로 파일을 쓰지 않는다. 결과를 새로 보존하려면
--output에 아직 존재하지 않는 경로를 명시한다. renderer 재실행도 위 new_run이 이미 있으면
새 경로를 사용해야 한다. 이 문서는 추가 실험 권한을 열지 않으며 기존 R4 기준을 바꾸지 않는다.

## 독립 checkout 확인

`ce7d191`의 새 full local clone과 NumPy-only Python 3.8.20 venv에서 비교 테스트 24개 및 실제
run1/run2 CLI 검증이 PASS했다. ignored 파일을 복사하지 않았고 검사 후 checkout은 clean이다.
이는 저장된 원배열의 CPU 검증이며 GPU 재렌더링/전체 설치 검증이 아니다.
[환경·재현 명령·미완료 항목](../../docs/public_evidence_reproduction_2026-09-12.md).
