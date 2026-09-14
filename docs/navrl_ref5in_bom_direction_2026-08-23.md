# ref5in BOM 방향 결정 — 2026-08-23

이 문서는 사용자가 “필요한 것은 모두 한다”는 방향에 맞춘 **권장 통합 방향**이다. exact model·serial·실측
전에는 구매 확정이나 hardware contract가 아니다.

## 통합 목표

최종 기체에는 아래를 모두 탑재한다.

- Livox Mid-360: obstacle sensing;
- Intel RealSense D435i: target perception/depth;
- Jetson Orin NX: detector/tracker/policy compute;
- Pixhawk 6C Mini: flight-control/state interface.

센서별 bench/replay와 추진계 측정을 먼저 하는 이유는 원인 분리를 위한 것이지, 최종 시스템에서
센서를 영구히 제외하기 위해서가 아니다. 모든 센서를 동시에 올린 통합 기체에서 최종 latency, FOV,
power, thermal, CG를 다시 측정한다.

## 권장 BOM 방향

| 축 | 권장 방향 | 현재 판정 |
|---|---|---|
| frame | iFlight AOS 5 V5.1 provisional reference | 228 mm/165±5 g; packaging gate 전 구매 보류 |
| propulsion | XING2 2207 1855KV + **Nazgul F5 5.1-inch tri-blade provisional** | 3.9 g/개·pitch 3.5; thrust curve 미확인 |
| ESC | BLITZ Mini E55 4-in-1 | 2–6S, 55 A continuous; motor/prop current 실측 필요 |
| compute module | **Jetson Orin NX 16GB provisional** | 8GB 대비 메모리·CPU 여유 우선; 단품 견적/전력/열 확인 전 구매 보류 |
| compute carrier | **Auvidea JNX42-LC** 우선 / Connect Tech Hadron 2순위 + low-profile heatsink/fan | JNX42: 80×104.6 mm, 3×USB3/2×CSI, 12 V only; Hadron: 82.6×58.8 mm, 49 g, 9–60 V; full assembly pending |
| battery | 6S 1550 mAh first screen | 1.20 kg 질량 여유 우선; voltage sag/endurance 측정 후 1850 검토 |
| sensors | Mid-360 + D435i 동시 탑재 | target system requirement; FOV/prop interference pending |

## 왜 1550 mAh를 먼저 보는가

현재 알려진 부품만 합친 1550 mAh 조합은 965.5–993.1 g으로, 1,200 g 설계점에 206.9–234.5 g이
남는다. 1850 mAh는 남는 질량이 157.9–185.5 g으로 줄어 Orin carrier/cooling/storage/DC-DC,
배선과 mount를 넣을 여지가 작아진다. 반대로 1550 mAh의 체공시간이 충분하다는 뜻은 아니다.

1550에서 다음을 측정한다.

1. hover/요격 duty cycle에서 loaded voltage sag;
2. Orin·Mid-360·D435i·Pixhawk를 켠 steady/peak power;
3. thermal limit와 실제 usable flight time.

체공시간이 요구조건을 못 맞추고 complete AUW가 1.20 kg 아래로 유지될 때만 1850을 재검토한다.

## carrier 선택 규칙

carrier 모델은 다음을 만족하는 후보만 비교한다.

- Orin NX exact module/메모리 SKU 호환;
- 9–20 V 입력 또는 6S에서 검증된 DC-DC 경로;
- board + cooling + storage + DC-DC의 mass/bbox/connector keepout을 공식 또는 실측으로 제공;
- 100×80 mm 안팎의 탑재 envelope를 목표로 하되, 실제 AOS deck CAD에서 다시 검증;
- CSI/USB/Gigabit/serial 등 D435i·Mid-360·Pixhawk 인터페이스를 동시에 제공;
- 팬 배기와 LiDAR/camera FOV가 겹치지 않음.

이 조건을 만족하는 모델이 없으면 carrier/냉각을 custom payload deck으로 설계하고, simulation에는
임의 질량을 넣지 않고 실제 BOM이 닫힌 뒤 별도 ref5in contract를 만든다.

## 다음 승격 순서

1. compact carrier 후보 2개와 냉각 방식의 공식 mass/bbox 확보;
2. D435i 실제 unit/cable와 배터리 1550의 CAD 또는 실측 외곽 확보;
3. packaging contract의 prop/FOV/connector/CG gate 실행;
4. exact BOM이 닫히면 thrust-stand와 power/thermal 측정;
5. 그 결과에 맞춰서만 URDF mass/inertia/collision/actuator를 변경하고 smoke부터 재검증.

현재 carrier 후보의 공식 인터페이스/전원 제약은
[`navrl_ref5in_carrier_screen_2026-08-23.md`](navrl_ref5in_carrier_screen_2026-08-23.md)에 기록했다.

## Orin 메모리 SKU provisional 결정

최종 탑재 목표가 Mid-360 + D435i + Pixhawk + detector/tracker/policy 동시 실행이므로 Orin NX
16GB를 우선 SKU로 둔다. NVIDIA 공식 volume MSRP(1KU+)는 16GB $999, 8GB $649로 차이는
$350이지만, 단품·국내 유통가는 별도 견적이 필요하다. 8GB가 더 저렴하더라도 실제 시스템의
메모리 여유·동시 센서 처리·로그/추론 버퍼를 먼저 확인한 뒤 downgrade한다. 이 결정은 구매
승인이 아니며, exact module SKU·전력모드·냉각·carrier 조합이 닫힐 때까지 provisional이다.
