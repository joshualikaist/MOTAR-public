# Renderer 재검증 규약 — 2026-09-11

사용자가 전체 재실행을 요청했다. 기존 R3/R4/R4b 판정·seed·geometry·threshold는 변경하지 않는다.
이번 작업은 일반 상자 렌더러의 재현/계측 검증이며 학습·detector·tracking·policy를 실행하지 않는다.

## 역사적 실행의 반복

HEAD `7c62dbfd1d58a67aa9e3e1ce2362cdd833456312`의 tracked-clean 코드로 R3/R4/R4b 각각
두 독립 프로세스, R5 기존 9+2셀, 기존 R5 128장면 단독 프로세스를 실행한다. 결과 root는
`results/renderer_reverification_2026-09-11_1951/`이다. 이 절의 실행은 이미 시작/완료됐으며
새 사전등록이나 새로운 독립 가설 검정으로 포장하지 않는다. FAIL의 재현은 실행 무결성 실패가 아니다.

## 아직 관측하지 않은 보완 계측의 고정 규약

이 문서는 아래 direct benchmark와 NumPy 감사 실행 전에 고정한다.

- Direct R5: 동일 상자 2개/삼각형 24개, seed 0, 동일 카메라 자세, HFOV 60°, far 20 m.
- 셀: (160×120, 240×135, 480×270) × batch (1,8,32), 그리고 480×270 × (64,128), 총 11개.
- **셀마다 새 프로세스**. 두 arm 모두 `shade(renderer.render(), ...)` 전체 함수를 직접 측정한다.
- arm별 warm-up 3회, 반복 3회 × 10표본. 반복별 arm 순서 flat/Lambertian, 역순, 정순.
- 각 wall-clock 표본 앞뒤 CUDA 동기화. 모든 원시 시간, mean/P50/P95를 저장한다.
  P95는 NumPy 선형 보간 percentile이며 역사적 도구의 order-statistic P95와 구분한다.
- GPU 이용률/장치 VRAM/해당 프로세스 VRAM은 nvidia-smi 주기 표본으로 기록한다.
  표본 최대는 순간 peak를 보장하지 않으며, 장치 전체 수치에는 display/원격접속 부하가 포함된다.
- Torch allocated/reserved peak는 별도 기록한다. Warp·드라이버·runtime fingerprint와 입력을 보존한다.
- 속도에 새 PASS 임계를 만들지 않는다. 기존 합산 비율과 직접 비율을 병기하고 어느 쪽도 상한으로 부르지 않는다.
- Simulator/environment step latency는 독립 렌더러 범위 밖이므로 N/A다.

## 독립 수치 감사

R4와 R4b를 각 한 번 더 GPU 렌더한다. RGB/G-buffer SHA를 기존 결과와 대조하고, GPU 평가 함수를
호출하지 않는 NumPy least-squares로 R², stable rank 20-bin population std로 구간 내 퍼짐/비율을
다시 계산한다. float64 차이 허용은 abs=1e-10, rel=1e-10으로 고정한다. 상수 휘도 arm의 R²는 0으로
취급한다. 8개 장면 전부, 4개 arm 전부를 보고한다. material 유지율도 보이는 픽셀에서 독립 계산한다.
이 감사는 원인 분해나 새로운 가설 검정이 아니다.

## 출처와 보존

기존 runner와 renderer 수치 구현은 변경하지 않는다. 보완 도구는 별도 파일로 추가한다.
커밋하지 않은 보완 도구를 실행할 경우 commit만으로 출처를 주장하지 않고, 실제 소스 파일 SHA와
HEAD에서의 추적/일치 여부, 이 규약 SHA를 실행 전후 검사하여 receipt에 남긴다. 최종 커밋·푸시는
별도이며 이번 검증 완료 조건이 아니다. 예전 receipt는 덮어쓰지 않는다.

기존 전체 Python unittest와 보완 도구 단위 테스트를 실행한다. 테스트 개수, skip, failure/error를
구분한다. 결과와 다른 서술은 현행 설명에서 교정하되 동결 사전등록과 원자료는 보존한다.
