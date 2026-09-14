# ref5in 기준 프레임 선정 — 2026-08-23

## 결론

현재 `navrl_ref5in_quad`의 형상을 가장 적게 바꾸는 **우선 설계 기준**은
**iFlight AOS 5 V5.1 Frame Kit**다. 아직 구매 확정이나 as-built 기체는 아니다. 아래 payload CAD와
질량 gate를 통과한 뒤에만 exact BOM으로 승격한다.

공식 제품 사양([iFlight product page](https://shop.iflight.com/AOS5-V5-Pro3735),
2026-08-23 확인)은 5-inch, wheelbase 228 mm, frame mass 165±5 g, arm thickness 6 mm,
flight-stack mounting 20×20/30.5×30.5 mm이다.

## 왜 이 후보인가

| 항목 | 현재 simulation design point | AOS 5 V5.1 | 차이/판정 |
|---|---:|---:|---|
| motor diagonal | 220.000 mm | 228 mm | +3.64%; 후보 중 가장 작은 계약 변경 |
| class | 5 inch | 5 inch | 일치 |
| frame mass | as-built 값 없음 | 165±5 g | exact BOM 입력 가능 |
| arm thickness | URDF가 실제 carbon arm을 규정하지 않음 | 6 mm | CAD/진동 검토 대상 |
| motor/stack mounting | 미동결 | 16/19 mm motor, 20/30.5 mm stack | 부품 선택 제약으로 사용 |

비교한 현행 공식 후보는 [AOS HS5](https://shop.iflight.com/AOS-HS5-Frame-Kit-Pro2258)
(233 mm, 226 g), [Nazgul Evoque F5 V3](https://shop.iflight.com/Nazgul-Evoque-F5-V3-Frame-Kit-Pro2407)
(236/239 mm, 256±10 g), [Nazgul DC5 ECO](https://shop.iflight.com/Nazgul-DC5-O4-Frame-Kit-Pro2287)
(240 mm, 239±5 g), [Nazgul XL5 ECO](https://shop.iflight.com/Nazgul-XL5-ECO-Frame-Kit-Pro2197)
(245 mm, 233 g)였다. 이들은 모두 AOS 5 V5.1보다 geometry/mass 변화가 크다. 공식 제품 페이지에
명시된 수치만 비교했으며 재고나 판매자 추정치는 쓰지 않았다.

## 지금 바로 구매하면 안 되는 이유

현재 공식 질량이 있는 센서/compute 단품의 **불완전 부분합**은 404.2–411.8 g이다. 프레임 허용범위
160–170 g을 더하면 564.2–581.8 g이고, simulation AUW 1,200 g에 남는 예산은
**618.2–635.8 g**뿐이다. 이 안에 다음이 전부 들어가야 한다.

- Orin carrier, cooling, storage, DC-DC;
- four motors, propellers, FC, ESC, receiver and power distribution;
- battery, wire/connectors, fasteners and every sensor mount.

또한 공식 frame dimension은 180×140×34 mm이고 max flight-stack height는 20 mm다. Mid-360,
D435i와 Orin assembly는 stock stack volume에 들어간다고 가정할 수 없으므로 custom two-level payload
deck의 scaled CAD가 필요하다.

## simulation에 미치는 최소 변경량

228 mm True-X에 5.0-inch prop을 쓴 단순 축방향 prop envelope는 약 **288.2 mm**다. 현재 collision
proxy 280 mm보다 약 8.2 mm(+2.94%) 크다. 하지만 실제 arm geometry, motor/prop model과 mount CAD가
고정되기 전에는 URDF collision box를 바꾸지 않는다. 후보 선정만으로 frozen checkpoint의 robot
contract를 수정하거나 재학습하지 않는다.

## 구매 전 packaging gate

다음 네 항목을 모두 통과해야 `purchase_authorized: true`와 exact BOM으로 승격한다.

1. 제조사 도면 또는 실측 frame CAD에 Mid-360, D435i, Orin carrier/cooling, Pixhawk, battery를 실제
   치수로 배치한다. 부품 상자만 겹치지 않는 것이 아니라 connector/cable bend와 정비 공간도 둔다.
2. prop swept disc와 payload/배선의 여유, Mid-360 360° 시야 가림, D435i 전방 FOV 가림을 수치로 낸다.
3. 모든 부품·mount·wire를 포함한 nominal/max AUW와 CG를 계산한다. 1.20 kg 초과 시 숫자를 숨기지
   않고 simulation mass/inertia/thrust contract 변경 대상으로 올린다.
4. 선택 motor/prop/ESC/battery의 공식 또는 thrust-stand 곡선으로 9.60 N/motor와 연속 열 한계를
   검증한다. 최대추력 한 점만으로 통과시키지 않는다.

이 gate 전까지 정확한 표현은 “AOS 5 V5.1 provisional packaging reference”이며
“실기 플랫폼 확정”이 아니다. 부품 envelope·FOV·connector·CG·power/thermal 입력 형식은
[`navrl_ref5in_payload_packaging_contract_2026-08-23.md`](navrl_ref5in_payload_packaging_contract_2026-08-23.md)에
고정했다.
