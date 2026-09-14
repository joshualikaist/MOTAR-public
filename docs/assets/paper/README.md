# README 논문형 블록 다이어그램 · 2026-09-11

README 본문의 그림 9개 전체와 같은 파일입니다. 흰 배경·큰 블록·직교 화살표로 통일했습니다.
SVG와 PDF는 벡터 원본이며, PNG는 3840×2160입니다. PPT에는 PNG 또는 SVG, 논문에는 PDF를 사용할 수 있습니다.
수치·긴 설명은 본문 캡션으로 분리했습니다. 새 실험·알고리즘·매개변수 변경은 없습니다.

## Fig. 1. 전체 연구 흐름

세 연구 축의 입력 자료 → 분석 → 기록 산출물을 구분했습니다. 실제 비행의 end-to-end 연결을 뜻하지 않습니다.

[SVG](system-overview-block-diagram.svg) · [PNG](system-overview-block-diagram.png) · [PDF](system-overview-block-diagram.pdf)

## Fig. 2. 환경과 관측 생성

환경 계약에서 정적·동적 장면으로 분기하고, 센서 출력과 ego state가 관측으로 이어지는 구성도입니다. 공간 축척도는 별도 보존했습니다.

[SVG](arena-block-diagram.svg) · [PNG](arena-block-diagram.png) · [PDF](arena-block-diagram.pdf)

## Fig. 3. 플랫폼 구성

시뮬레이션 후보를 vehicle·sensor·actuation model로 분해한 구성도입니다. 부품 배선도나 조립 완료된 기체가 아닙니다.

[SVG](platform-block-diagram.svg) · [PNG](platform-block-diagram.png) · [PDF](platform-block-diagram.pdf)

## Fig. 4. 정책과 제어 체인

기존 8단계 제어 체인을 큰 블록으로 다시 배치했습니다. 윗줄은 왼쪽→오른쪽, 아랫줄은 오른쪽→왼쪽입니다. gain·수식은 본문에 남겼습니다.

[SVG](control-block-diagram.svg) · [PNG](control-block-diagram.png) · [PDF](control-block-diagram.pdf)

## Fig. 5. Safety filter

직선·arc-clearance는 선택 가능한 비교군이며 연속 필터가 아닙니다. geometry와 cap-law 축, 정책 명령의 분기·합류를 표시했습니다. 방향 재계획이나 안전 보장으로 해석하지 않습니다.

[SVG](safety-filter-block-diagram.svg) · [PNG](safety-filter-block-diagram.png) · [PDF](safety-filter-block-diagram.pdf)

## Fig. 6. 인지 처리 과정

검출·광류의 병렬 흐름 → 특징 결합 → 시간 이력 → Transformer → 후보 선택입니다. 출력은 프레임 안의 rank 또는 NO_LOCK이며 물리적 track ID나 거리 추정이 아닙니다.

[SVG](perception-block-diagram.svg) · [PNG](perception-block-diagram.png) · [PDF](perception-block-diagram.pdf)

## Fig. 7. 보관된 SAM 후보

점선 박스·화살표는 미구현 제안입니다. 실선은 기존 offline CPU instance adapter와 CC stub 입력뿐입니다. SAM worker나 제어 통합은 구현되지 않았습니다.

[SVG](sam-archive-block-diagram.svg) · [PNG](sam-archive-block-diagram.png) · [PDF](sam-archive-block-diagram.pdf)

## Fig. 8. 색 검출기 실패 경로

동색 물체가 같은 픽셀 분류를 통과해 하나의 mask와 중심점으로 합쳐지는 기존 실패를 도식화했습니다. 현재 채택한 실영상 detector가 아닙니다.

[SVG](color-baseline-block-diagram.svg) · [PNG](color-baseline-block-diagram.png) · [PDF](color-baseline-block-diagram.pdf)

## Fig. 9. E3 분석 절차

E3-S의 크기 기반 거리 평가와 E3-P의 자세 신뢰성 gate를 분리했습니다. 수치·GT 품질·시간 정합성 한계는 본문과 결과 보고서에 남겼고 새로운 원인 분해는 하지 않았습니다.

[SVG](e3-analysis-block-diagram.svg) · [PNG](e3-analysis-block-diagram.png) · [PDF](e3-analysis-block-diagram.pdf)

근거: [현재 README](https://github.com/joshualikaist/MOTAR/blob/main/README.md) · [streaming 명세](../../specs/perception_streaming_v1.md) · [E3 계획](../../plans/eth_ds5_e3_2026-09-10.md).

재생성: `python tools/render_paper_blocks.py` (Chrome, websocket-client 필요).
