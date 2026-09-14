# ref5in 추진계·배터리 1차 스크리닝 — 2026-08-23

이 문서는 구매 목록이 아니라 **공식 사양으로 질량·전류 예산을 먼저 거르는 화면(screen)**이다.
추력표, 실제 payload CAD, 완전한 BOM이 없으므로 이 조합으로 URDF나 frozen checkpoint를 바꾸지 않는다.

## 후보 조합

| 부품 | 후보 | 공식 수치 | 출처 |
|---|---|---|---|
| 모터 | iFlight XING2 2207 1855KV | 31.6 g/개, 6S(24 V), peak 35.08 A, 16×16 mm | [공식 제품 페이지](https://shop.iflight.com/xing2-2207-4s-6s-fpv-motor-unibell-pro1464) |
| 프로펠러 | iFlight Nazgul F5 Tri-blade (provisional) | 5.1 inch, pitch 3.5, 3 blades, 3.9 g/개, 5 mm hub | [공식 제품 페이지](https://shop.iflight.com/index.php?product_id=3196&route=product%2Fproduct) |
| ESC | iFlight BLITZ Mini E55 4-in-1 | 11.3 g, 2–6S, 55 A continuous / 65 A burst, 20×20 mm | [공식 제품 페이지](https://shop.iflight.com/BLITZ-Mini-E55-4-IN-1-2-6S-ESC-Pro1663) |
| 배터리 A | iFlight Fullsend 6S 1550 mAh 150C | 253±5 g, 34.41 Wh | [공식 제품 페이지](https://shop.iflight.com/iFlight-Fullsend-6S-1550mAh-150C-Battery-Pro2411) |
| 배터리 B | iFlight Fullsend 6S 1850 mAh 150C | 302±5 g, 41.07 Wh | [공식 제품 페이지](https://shop.iflight.com/iFlight-Fullsend-6S-1850mAh-150C-Battery-Pro2412) |
| 프레임 | iFlight AOS 5 V5.1 | 165±5 g, 228 mm | [공식 제품 페이지](https://shop.iflight.com/AOS5-V5-Pro3735) |

5.1-inch tri-blade는 iFlight Nazgul F5를 provisional 후보로 고정했다. 제품 페이지에는 직경·pitch·엽수·
질량·hub가 공개되어 있지만, XING2와 6S 조합의 **공식 propeller별 thrust/current 표는 확보하지 못했다**.
따라서 아래는 정적 질량 계산이며 “9.60 N 추력 달성”이 아니다.

## 질량 예산

계산에 포함한 값:

- 센서·연산·FCU 명시 단품 부분합: 404.2–411.8 g;
- AOS frame: 160–170 g;
- XING2 2207 ×4: 126.4 g;
- 5.1-inch prop ×4: 15.6 g;
- Mini E55 ESC: 11.3 g (제조사 tolerance 없음);
- 배터리: 제조사 표기 범위.

| 배터리 | 알려진 부품 합계 | 1,200 g까지 남는 질량 |
|---|---:|---:|
| 6S 1550 mAh | **965.5–993.1 g** | **206.9–234.5 g** |
| 6S 1850 mAh | **1,014.5–1,042.1 g** | **157.9–185.5 g** |

남은 질량에는 Orin carrier·냉각·storage·DC-DC, 배선·커넥터·마운트가 들어간다. Pixhawk는 이미
명시 단품 부분합에 포함되어 있다. 따라서 1550 mAh가 질량상 우선 후보지만, 체공시간과 전압강하는
실측하지 않았고 1850 mAh보다 짧을 수 있다.

## 전류 screen

모터 제조사 peak current 35.08 A × 4 = 140.32 A이고, ESC 제조사 continuous rating은 55 A × 4
= 220 A다. 수치상 ESC 연속 전류 정격은 모터 peak 합보다 79.68 A 높다. 1550 mAh 150C 배터리의
명목 C-rating 계산은 232.5 A다. 이것은 **정격 비교**일 뿐 실제 추진점의 전류, 전압 sag, 열 포화,
배터리의 실효 C-rating을 검증한 것이 아니다.

## 구매/승격 gate

다음 항목 전에는 어떤 부품도 exact BOM으로 승격하지 않는다.

1. XING2 2207 1855KV + 실제 선택 prop + 6S 전압에서 thrust/current/RPM 곡선을 제조사 자료 또는
   thrust stand로 확보하고, 9.60 N/모터와 hover point를 각각 확인한다.
2. 전체 부품 CAD에 Orin carrier/cooling/storage/DC-DC와 배선을 추가해 nominal/max AUW, CG, prop
   clearance, Mid-360 360° 시야, D435i 전방 시야를 계산한다.
3. 1550/1850 mAh 각각의 전압 sag·연속전류·열·예상 체공시간을 동일 조건으로 측정한다.
4. 결과가 1.20 kg 또는 collision envelope를 벗어나면 simulation contract 변경을 별도 실험으로
   등록한다. 기존 checkpoint를 조용히 덮어쓰지 않는다.

현재 판정: **1550 mAh 조합 = 질량 screen 통과, 추진 성능 미판정; 1850 mAh 조합 = 질량 screen 통과,
payload 여유가 더 작음. 둘 다 구매 확정 아님.** 실제 추력 측정의 필드와 판정 규칙은
[`navrl_ref5in_thrust_stand_protocol_2026-08-23.md`](navrl_ref5in_thrust_stand_protocol_2026-08-23.md)에
고정했다.
