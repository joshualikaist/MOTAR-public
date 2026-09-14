# R4 background v1 — 구현 및 기술 검증 계약

기존 R4/R4b는 C FAIL로 보존한다. 이 작업은 그 실험의 세 번째 fixture 재시험이 아니라,
원래 요청에서 빠진 일반 background scene과 intervention 검사를 구현하는 별도 engineering 단계다.
R² < 0.5를 통과시키는 것이 목적이 아니며, seed/장면/임계 선택을 통해 과거 FAIL을 뒤집지 않는다.

## 입력과 장면

- 기존 독립 Warp G-buffer와 Lambertian shading을 재사용한다. simulator/task/model/policy import 금지.
- 절차적 floor, wall, 12각 column, 두 링크의 일반 hinged panel. UAV/실제 대상 자산 사용 없음.
- 길이 단위 m. camera +X right, +Y down, +Z forward; world quaternion xyzw.
- quad/material-patch ID와 part ID는 debug/evaluation metadata다. RGB 입력 채널에 추가하지 않는다.
- material 4종은 표면 patch 순서로 배정한다. 거리를 읽어 배정하지 않는다. 이것만으로 통계적 독립을
  주장하지 않는다. joint angle은 30도 고정, seed 619로 4개 appearance/camera view를 기술 점검한다.
- camera 240×135, HFOV 60°, far 20 m. 이것은 다양한 geometry seed의 본 실험이 아니다.

## 실행 전에 고정하는 기술 검사

1. 모든 view에서 4 object instance 각각 32 valid pixels 이상, material 0 각각 64 pixels 이상.
   부족하면 INSUFFICIENT_VISIBILITY이며 빈 mask의 vacuous PASS를 허용하지 않는다.
2. 같은 입력 2회 렌더의 range/depth/normal/face/instance/valid exact 일치.
3. material 0의 albedo만 0.5배: 선택 영역은 모든 픽셀 변화(채널 최대 차 >1e-7),
   비선택 영역/배경은 exact 불변. material을 공유하는 표면은 함께 바뀌는 것이 계약이다.
4. face→material을 `(id+1)%4`로 순환: **보이는 픽셀의 material ID 변경률=100%**,
   RGB 변경률도 전 view에서 100%. vertex/triangle/instance와 G-buffer는 그대로 사용한다.
5. light direction의 X 성분만 반전: material 불변, view별 valid RGB MAE >1e-6.
6. 모든 RGB finite [0,1], 정확히 3채널. instance debug ID +100에도 RGB exact 불변.

TECHNICAL_PASS는 위 구현 계약만의 통과다. 기존 R4 기준 C는 diagnostic에 그대로 병기한다.
R², raw within-bin std와 mean-normalized std를 모두 기술 통계로 기록한다. scene/view 내 픽셀이나
patch를 독립 반복으로 간주하지 않으며, p-value/일반화/shortcut 감소를 주장하지 않는다.
검사 실패 시 원인과 실패 receipt를 남기고, 기준이나 view를 결과에 맞춰 바꾸지 않는다.

## 검증/출처

CPU 단위 테스트, 새 프로세스 2회의 Warp/GPU 기술 smoke, 전체 regression을 수행한다.
출력은 새 디렉터리에만 생성한다. 실제 소스 SHA·source snapshot·runtime(Warp/driver 포함),
모든 geometry/appearance/camera 입력과 배열 SHA를 남긴다. 미커밋 소스는 명시한다.
원래 renderer/fixture/metric 코드와 역사적 JSON은 수정하지 않는다.

## 구현 다음 단계 — 이번 기술 smoke와 구분

1. 이 background 장면에 대한 새 R5 engineering benchmark를 고정한다. 작은 셀은 telemetry
   간섭을 분리하고, end-to-end 시간/VRAM을 측정한다. 상자 장면의 비용 수치를 전용하지 않는다.
2. 별도 R4 background 본 실험을 사전등록한다. 독립 geometry/camera 블록 안에서 material과
   lighting 조건을 교차하고, scene를 분석 단위로 고정하며 held-out scene을 분리한다.
3. 본 실험은 영상 통계까지만 보고한다. detector/tracker/PPO/실기 연결은 후속 단계에 포함하지 않는다.
