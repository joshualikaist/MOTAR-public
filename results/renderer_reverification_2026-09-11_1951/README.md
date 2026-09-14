# R3–R5 전체 재검증 — 2026-09-11

검증 상태: **VERIFIED_WITH_LIMITATIONS**. R3/R4/R4b 수치와 출력은 재현된다.
**R4와 R4b의 실험 판정은 계속 FAIL**이며 검증 PASS로 덮어쓰지 않는다.
[기계 판독 summary](summary.json), [CPU 재계산 도구](recompute_summary.py),
[실행 규약](../../docs/renderer_reverification_protocol_2026-09-11.md).

## 범위와 출처

- 일반 상자 2개를 쓰는 독립 renderer만 GPU로 실행했다. 학습·실제 detector/tracker·PPO·정책 평가는 없다.
- 원래 요청한 floor/wall/column/articulated scene 검증은 여기서 구현하거나 실행하지 않았다.
- 기존 R3/R4/R4b 및 기존 R5는 tracked-clean `7c62dbfd1d58a67aa9e3e1ce2362cdd833456312`에서 실행했다.
- 보완 도구 `tools/verify_renderer_measurements.py`와 보완 규약은 실행 당시 **미커밋**이었다.
  이를 커밋된 소스라고 주장하지 않는다. 실제 소스 전체 SHA, HEAD에서의 추적/일치 여부와 규약 SHA를
  보완 receipt에 저장했고 각 실행 전후 및 11개 셀 사이에서 동일함을 검사했다.
- 보완 도구 SHA: `64cbf6d0c8aeb9d41f0e3245402660e4c287fbc71edcbf1bda66c54b5a9ece3f`.
- 규약 SHA: `c7ca44104243c6d34bbb118d7599490e049d2234dbeef8a0936f4b6cbb47dadf`.
- Python 3.8.20, torch 2.4.1+cu121, Warp 1.0.0, NVIDIA driver 570.133.07, RTX 3070.
  runtime의 legacy `driver_cuda`는 torch build CUDA 별칭이다. 실제 드라이버는 별도 필드에 기록했다.
- 실행 후 문서만 교정했다. 기존 renderer/판정 코드와 역사적 JSON receipt는 변경하지 않았다.
  현재 변경사항은 아직 커밋·푸시하지 않았다.

## R3/R4/R4b 재현

| 단계 | GPU 반복 | 기존/새 실험 판정 | 검증 |
|---|---:|---|---|
| R3 seed 173 표기 | 2 프로세스 | PASS / PASS | 기존 metric·배열 SHA exact, 새 두 run.json 바이트 동일 |
| R4 seed 409 | 2 프로세스 + 독립 감사 1회 | FAIL / FAIL | 기존 장면별 지표·배열 SHA exact, 새 두 run.json 바이트 동일 |
| R4b seed 409 | 2 프로세스 + 독립 감사 1회 | FAIL / FAIL | 기존 장면별 지표·배열 SHA exact, 새 두 run.json 바이트 동일 |

R3의 seed 인자는 기록용이며 구현은 고정 fixture/고정 appearance를 쓴다.
이를 새로운 무작위 표본이나 seed 일반화 검정으로 부르지 않는다.
이전/현재 receipt는 source commit과 일부 추가 metadata가 다르므로 파일 전체가 같다는 주장은
**이번 캠페인의 동일 단계 두 run 사이**에만 적용한다. 전체 비교 필드는 summary에 열거돼 있다.

R4 C의 최대 R²는 0.5860849934, R4b는 0.5816185136으로 계속 0.5를 초과한다.
NumPy least-squares와 population std 기반 독립 재계산은 4개 arm × 8장면 전부 통과했다.
검사한 R²·평균·분산·bin std의 최대 절대 차이는 **4.533e-16**다.
비율도 고정 abs/rel 1e-10 허용 안에서 일치한다.

R4b material 유지율은 97.79–98.44%, 장면 6은 98.36%로 재현된다.
작게 작동한 재배정 하나의 FAIL로 모든 재배정 가능성이나 기준 설계의 무효를 입증할 수 없다.
픽셀 R²는 영상 기술통계로 유효하다. 독립 scene 수/불확실성으로 픽셀 수를 쓰면 안 되며,
같은 장면의 면들도 자동으로 독립 표본이 되지는 않는다. 평균색 기준 통일은 평균 휘도 일치가 아니다.

## R5 기존 방식 재실행과 메모리 결함 검증

기존 9+2셀을 모두 재실행했다. 128장면 단계별 평균 합산 비율은 **1.187986배**다.
그 비율은 전체 렌더 루프 직접 측정값도, 다른 메시에서의 비용 상한도 아니다.

| 480×270, batch 128 | Torch allocated peak | Torch reserved peak |
|---|---:|---:|
| 기존 runner의 64→128 연속 실행 | 1,830.528 MiB | 2,106 MiB |
| 같은 기존 runner의 128 단독 새 프로세스 | 1,601.134 MiB | 1,862 MiB |
| 차이 | **229.394531 MiB** | 244 MiB |

allocated 차이는 이전 64장면의 owned G-buffer 크기
`64 × 480 × 270 × (4+4+12+4+4+1) / 1024²`와 정확히 일치한다.
기존 runner는 `renderer`만 해제하고 `buffer`를 다음 셀 초기까지 유지한다.
따라서 반복 실행돼 같은 숫자가 나왔어도 그 메모리를 셀 단독 요구량으로 해석하면 틀린다.
**기존 runner는 역사적 재현용으로 보존**했으며 새 계측은 셀마다 새 프로세스를 사용해 이 문제를 피한다.

## R5 전체 루프 직접 계측 — 11셀 전부

arm별 3회 warm-up, 3반복 × 10표본. `shade(renderer.render())` 호출 앞뒤 CUDA 동기화를 포함한다.
각 셀은 새 프로세스다. 같은 셀 안 두 arm은 같은 geometry/material/lighting/pose 입력을 쓴다.
P95는 NumPy 선형 보간이다. 기존 runner의 order-statistic P95와 구분한다.
원시 30표본/arm, 반복별 통계, 입력, runtime, telemetry를 [r5_direct](r5_direct/)에 보존했다.

| 해상도 | batch | flat mean / P50 / P95 ms | Lambertian mean / P50 / P95 ms | 평균 비율 |
|---|---:|---:|---:|---:|
| 160×120 | 1 | 1.141 / 0.793 / 2.024 | 0.958 / 0.848 / 1.429 | 0.839 |
| 160×120 | 8 | 1.229 / 0.887 / 1.719 | 1.329 / 1.014 / 1.838 | 1.082 |
| 160×120 | 32 | 1.679 / 1.551 / 2.330 | 2.444 / 1.866 / 7.014 | 1.456 |
| 240×135 | 1 | 1.203 / 0.814 / 2.388 | 1.049 / 0.964 / 1.472 | 0.872 |
| 240×135 | 8 | 1.557 / 1.138 / 3.233 | 1.323 / 1.167 / 1.944 | 0.850 |
| 240×135 | 32 | 2.806 / 2.365 / 3.954 | 3.248 / 2.905 / 4.322 | 1.158 |
| 480×270 | 1 | 1.997 / 1.864 / 2.648 | 2.800 / 2.045 / 7.881 | 1.402 |
| 480×270 | 8 | 3.702 / 3.401 / 4.222 | 4.396 / 3.875 / 8.016 | 1.187 |
| 480×270 | 32 | 9.327 / 8.788 / 13.897 | 10.764 / 10.096 / 15.420 | 1.154 |
| 480×270 | 64 | 16.158 / 15.767 / 20.817 | 19.938 / 18.896 / 24.051 | 1.234 |
| 480×270 | 128 | 31.210 / 30.611 / 34.417 | 37.380 / 36.205 / 41.335 | 1.198 |

128장면 직접 측정: **31.210 → 37.380 ms, 1.197689배**, Lambertian 처리량 약 **3,424장/s**.
이는 batch 처리량이다. 128장면 한 반복의 속도는 약 26.75회/s이며 단일 실시간 카메라 FPS와 혼동하지 않는다.
이 조건에서 Torch allocated peak 1,601.1 MiB / reserved 1,862 MiB,
프로세스 VRAM **표본 최대 2,076 MiB**, 장치 전체 VRAM **표본 최대 2,748 MiB**였다.
장치 이용률 표본은 18–99%이고 warm-up/arm 사이 구간도 포함한다. 모든 표본과 시각은 receipt에 있다.

**계측 한계도 결과다.** 화면/원격접속 부하가 있는 GPU에서 주기적인 nvidia-smi 계측을 함께 돌렸다.
따라서 무부하 exclusive GPU 또는 계측 비용을 제거한 renderer-only latency를 주장하지 않는다.
작은 셀은 6–8 ms 이상치가 평균에 큰 영향을 주었고 세 셀에서 평균 비율이 1보다 작다.
이를 shading이 더 빠르다는 증거로 해석하지 않으며, 원인을 단독으로 확정하거나 표본을 제거하지 않았다.
작은 셀의 GPU 표본은 2–3개뿐이다. 표본 최대는 순간 VRAM peak나 arm별 GPU 이용률을 보장하지 않는다.
실제 environment/simulator step latency는 **N/A**다. 이 캠페인으로 일반적 느려짐 상한을 정하지 않는다.

## 테스트와 실행 중 문제

- 렌더러 전용 CPU 테스트: **67/67 통과** (기존 57 + 보완 10).
- 전체 Python unittest: **1,415개 실행, 1,411 통과, 4 skip, failure/error 0**.
  [첫 GPU-visible 전체 로그](full_tests_gpu_visible.log),
  [문서 교정 후 최종 전체 로그](full_tests_final.log). 두 번 모두 같은 개수와 결과다.
- 최초 전체 실행은 제가 `CUDA_VISIBLE_DEVICES=''`로 GPU를 숨겨 시작했다.
  일부 기존 import 계약이 CUDA를 요구해 1,401개 실행 / failure 1 / error 1 / skip 4였다.
  이 오류로 클래스의 테스트 4개가 실행되지 않았다. 이 시도는 신규 테스트 10개 추가 전이었다.
  GPU를 노출한 환경에서 코드를 수정하지 않고 전체 재실행해 위 결과를 얻었다.
- 로그의 의도된 예외/FAIL 문구는 실패 주입 테스트의 출력이다. 최종 unittest 결과와 혼동하지 않는다.

## 재현 방법

저장된 증거의 CPU 검증(파일을 쓰지 않음):

```bash
python3 -B results/renderer_reverification_2026-09-11_1951/recompute_summary.py
```

전체 회귀 테스트:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/fair/miniconda3/envs/aerialgym/bin/python -B -m unittest discover -s tests
```

GPU 실행 인자는 [commands.json](commands.json)에 모두 있다. **출력 경로는 반드시 새 경로로 변경한다.**
기존 runner는 tracked-clean 소스를 요구하므로 `7c62dbf`에 해당하는 소스를 사용하거나 검토된 변경을
커밋한 뒤 실행해야 한다. 기존 출력/실패 결과를 삭제하거나 guard를 우회하지 않는다.
보완 도구는 source manifest를 직접 기록한다. 수정된 source로 실행하면 새 계보로 취급한다.

다음 연구는 아직 미실행인 일반 배경/복잡 geometry에 대한 별도 설계다.
R4/R4b의 FAIL은 닫힌 기록으로 남기고, 결과를 보고 합격 기준이나 장면을 바꾸는 재시험은 하지 않는다.
