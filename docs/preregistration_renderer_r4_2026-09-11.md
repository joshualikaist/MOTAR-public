# R4 사전등록 — 배경·재질 외형 모델 비교 (2026-09-11)

결과를 보기 전에 커밋한다. R5 벤치마크는 이미 끝났으나 R4의 지표·기준과 무관하다.

## 무엇을 비교하나

**같은 기하**에 세 가지 외형 모델을 적용하고 영상 통계만 비교한다.

| arm | 정의 |
|---|---|
| `flat` | 모든 material이 동일 linear RGB. 현재 NavRL 표적 칠하기와 같은 부류 |
| `depth_gradient` | 휘도 = `base + gain × (1 − range/far)`. 현재 NavRL 배경과 같은 부류이나 **독립 구현**이다 |
| `lambertian` | 면 법선·면별 material·조명 방향 기반 음영 |

`depth_gradient`는 NavRL 렌더러의 재실행도, byte-identical 재현도 아니다. **일반적 외형 모델 사이의
비교**이며 NavRL 수치와 직접 비교하지 않는다.

## 고정 조건

- fixture: `box_fixture()` (상자 2개, material 4종, instance 2종)
- 카메라 480×270, 수평 FOV 60°, far 20 m
- seed 409, 장면 8개, 각 장면의 material·조명은 sequence 시작에 샘플링 후 고정
- 세 arm은 **동일한 G-buffer**에서 계산한다. 기하는 한 번만 캐스팅한다

## 기록할 지표

1. material별 linear 휘도 평균과 **공간 분산**
2. 같은 range 구간(20분위) 안에서 material·법선에 따른 휘도 차이
3. `range → 휘도` 단순 선형회귀의 R²
4. 포화(=1.0) 및 흑색(=0.0) 픽셀 비율, 유효 geometry coverage
5. 장면별 통계 분포

## 판정 — 사전 고정

| 항목 | 기준 |
|---|---|
| A. flat이 평평한가 | `flat`의 유효 픽셀 휘도 분산 ≤ `1e-12` |
| B. depth_gradient가 깊이의 함수인가 | `depth_gradient`의 R² ≥ `0.99` |
| C. lambertian이 깊이만의 함수가 아닌가 | `lambertian`의 R² < `0.5` |
| D. 같은 깊이에서 외형이 갈라지는가 | range 분위 안 휘도 표준편차의 중앙값이 `lambertian` > `depth_gradient` × 3 |
| E. 재현성 | 같은 seed 두 독립 프로세스에서 모든 지표·배열 SHA exact 일치 |

다섯 항목 전부 통과해야 `PASS`다. 하나라도 실패하면 실패한 항목을 명시한 `FAIL`이다.

**R²는 진단 지표로만 쓴다.** 낮다는 사실 하나로 "정상"을 선언하지 않으며, 통과시키려고 잡음을
주입하지 않는다. scene 배치에 따라 깊이와 법선이 상관될 수 있으므로 D를 함께 요구한다.

## 이 실험이 말하지 않는 것

학습 모델이 없다. shortcut 의존도가 줄었다는 결론은 내리지 않는다. 결론은 **영상 통계의 차이**까지다.
상자 fixture이며 UAV 자산도, 배경 질감 텍스처도 아직 아니다.
