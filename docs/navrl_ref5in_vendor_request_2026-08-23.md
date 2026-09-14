# ref5in 구매·제조사 확인 요청서 — 2026-08-23

이 문서는 주문서가 아니라 **exact BOM과 packaging CAD를 닫기 위한 확인 목록**이다. 아래 값이
확인되기 전에는 구매 확정, URDF 변경, 재학습을 하지 않는다.

## 우선 견적 후보

| 항목 | 1차 후보 | 2차 후보 | 반드시 받을 값 |
|---|---|---|---|
| Orin module | Jetson Orin NX 16GB, P/N 900-13767-0000-001 | Orin NX 8GB | 단품 가격/납기/보증/실물 질량/전력모드 |
| carrier | Auvidea JNX42-LC | Connect Tech Hadron NGX012 | board mass, 2D/3D CAD, mounting hole, connector keepout, BSP/JetPack |
| carrier power | JNX42 12 V rail + 6S→12 V DC-DC | Hadron 9–60 V input | 입력 범위, peak current, converter/보호회로 질량·bbox, EMI 권장 |
| cooling | Orin NX 호환 low-profile fan heatsink | 제조사 active heatsink | complete assembly mass/bbox, 40 W thermal capability, fan voltage/current |
| storage | M.2 NVMe 2242/2280 | Hadron 2242 | model, mass, bbox, vibration/temperature rating |
| battery | 6S 1550 mAh | 6S 1850 mAh | 실물 mass/bbox, connector, loaded sag/current/thermal data |
| prop | iFlight Nazgul F5 5.1×3.5×3 | 동일 규격 여분 | exact revision, CW/CCW balance, mass, purchase lot |

## 제조사/판매처에 물을 질문

1. Orin NX 16GB와 carrier가 **동일 JetPack/L4T 버전**에서 부팅되는가?
2. board/module/heatsink/fan/NVMe를 합친 실제 질량과 외곽 CAD(STEP)가 있는가?
3. carrier mounting hole 위치와 board bottom keepout, connector가 튀어나오는 방향은 무엇인가?
4. 6S 배터리 연결 시 startup/inrush와 peak current를 허용하는가? 별도 fuse/TVS/DC-DC가 필요한가?
5. UART/GbE/USB3 연결을 잠금식 cable로 구성할 때 cable·breakout·strain relief의 질량과 최소 굽힘
   반경은 얼마인가?
6. 팬을 장착한 상태에서 40 W Orin NX power mode를 연속 운용할 수 있는가? 열저항 또는 시험 조건은?
7. D435i USB3, Mid-360 GbE/9–27 V, Pixhawk UART를 동시에 연결할 수 있는 wiring diagram을 제공할 수 있는가?

## 내부 승격 gate

- `measured_mass` 또는 제조사 문서값과 문서/도면 SHA가 기록됨;
- frame·prop swept volume, Mid-360 360°×59° FOV, D435i 전방 FOV, connector bend, CG를 CAD에서 PASS;
- complete AUW가 1.20 kg 이하이고, battery를 포함한 power/thermal budget이 닫힘;
- bench boot + sensor timestamp + Pixhawk link + 30분 thermal/power smoke를 통과함.

현재 기본 방향은 **Orin NX 16GB + JNX42-LC + 6S1550 + Nazgul F5**이며, Hadron은 질량·전원
조건이 더 유리할 때 승격하는 2순위다. 이 문서의 빈칸을 채우기 전에는 어느 후보도 최종 실기
플랫폼으로 부르지 않는다.
