# Orin NX carrier 후보 screen — 2026-08-23

## 1차 권장 후보: Auvidea JNX42-LC

공식 [JNX42 product page](https://auvidea.eu/product/jnx42/)와 제조사
[technical reference](https://auvidea.eu/download/D70784-JNX42-v1.8.5.pdf)를 대조했다.

| 항목 | 공식 확인값 | MOTAR 적합성 |
|---|---|---|
| module | Jetson Orin Nano/NX 계열 | Orin NX 후보와 일치 |
| size | 80×104.6 mm; side wings 사용 시 80×112.6 mm | AOS 180×140 mm deck 안에 배치 가능성 있음; custom deck 필수 |
| USB | Orin NX에서 native USB 3.0 ×3 | D435i + 여분 포트에 유리 |
| CSI-2 | 2× 4-lane CSI-2 | camera 직접 연결 경로 |
| storage | M.2 Key M NVMe, 2242/2280 | 별도 storage 질량을 줄일 가능성 |
| cooling | original NVIDIA heatsink/fan용 5 V fan connector | fan/방열 외곽과 배기 방향은 별도 CAD 필요 |
| power | base board 12 V; optional module은 12–48 V 입력을 12 V로 변환 | 6S(최대 25.2 V) 직결 금지, DC-DC 필수 |
| board mass | 공식 technical reference에서 확인하지 못함 | exact AUW 계산 보류 |

JNX42-LC를 우선 보는 이유는 LTE/추가 side option을 빼고 필요한 USB/CSI만 남길 수 있기 때문이다.
JNX42-M2/LM은 기능은 늘지만 현재 센서 계약에 필요하지 않아 1차 후보에서 제외한다.

## 인터페이스 배치

- D435i: USB 3.0 한 포트. USB cable strain relief와 camera FOV keepout을 CAD에 포함한다.
- Mid-360: Ethernet과 9–27 V 전원. carrier의 USB 숫자에 포함하지 않고 별도 Ethernet/power path로
  기록한다. startup peak 18 W와 steady 6.5 W를 DC-DC budget에 넣는다.
- Pixhawk: UART/USB 중 실제 연결을 하나로 고정하고, serial level/ground/시간 동기 방식을 기록한다.
- Orin cooling: fan airflow가 Mid-360의 광학부나 D435i FOV를 가리지 않도록 배기 방향을 위·측면으로
  정한다.

## 현재 판정

`JNX42-LC = INTERFACE_AND_SIZE_CANDIDATE`, exact BOM은 아니다.

다음 세 항목이 없으면 구매/packaging 승격을 하지 않는다.

1. JNX42-LC 실물 또는 제조사 확인 질량;
2. 선택 Orin NX SKU, heatsink/fan, NVMe, DC-DC의 질량·bbox·connector keepout;
3. AOS deck와 Mid-360/D435i/Pixhawk/battery를 모두 넣은 CAD에서 prop/FOV/CG gate 통과.

JNX42의 12 V 제약 때문에 6S battery → regulated 12 V carrier rail → 5 V/3.3 V peripheral rail의
전력 구조를 사용한다. 이 DC-DC를 빼고 “carrier가 6S를 지원한다”고 쓰지 않는다.

## 2순위 후보: Connect Tech Hadron (NGX012)

제조사 [Hadron product page](https://connecttech.com/product/hadron-carrier-for-nvidia-jetson-orin-nx/)의
공개 사양을 확인했다.

| 항목 | 공식 확인값 | MOTAR 적합성 |
|---|---|---|
| module | Jetson Orin NX/Nano 호환 | Orin NX 후보와 일치 |
| size / mass | 82.6×58.8 mm; 49 g | JNX42-LC보다 작고 가벼움 |
| USB / network | USB 3.1 ×2, 1 GbE ×1 | D435i 1포트와 여분 1포트; Mid-360 Ethernet 경로 가능 |
| camera | 4-lane MIPI CSI-2 ×1 | 현재 센서 구성에는 충분할 가능성, 이중 CSI는 불가 |
| storage / cooling | M.2 2242 NVMe, 5 V 4-pin fan | NVMe와 저 profile fan 별도 확인 필요 |
| power | +9–60 V DC (nominal +12–48 V) | 6S 최대 25.2 V 범위에 들어가나 전원 커넥터/노이즈 검증 필요 |
| connector | rugged locking IO connector | 일반 USB/Ethernet cable보다 harness/strain relief 설계가 필요 |

현재 인터페이스만 보면 Hadron은 D435i(USB3), Mid-360(Ethernet), Pixhawk(UART) 조합을 수용할 수
있다. 다만 공개 사양만으로는 선택 Orin NX SKU의 냉각 온도, 케이블/브레이크아웃의 실제 질량과
외곽, 6S 전원 transient가 닫히지 않는다.

## 후보 간 선택 규칙

| 후보 | 장점 | 차단 위험 | 현재 순위 |
|---|---|---|---|
| JNX42-LC | USB3×3, CSI-2×2, 넉넉한 I/O | 12 V only라 regulated DC-DC 필수; board mass 미확인 | 1차 |
| Hadron NGX012 | 49 g, 82.6×58.8 mm, 9–60 V 입력 | USB3×2/CSI×1; locking harness와 breakout 질량 필요 | 2차 |

두 후보 모두 exact Orin NX module/heatsink/fan/NVMe/cable/DC-DC(or direct-6S protection)의
질량·bbox·keepout을 확보하고, packaging contract의 prop/FOV/CG/thermal gate를 통과해야
구매 후보로 승격한다. 현재는 어느 쪽도 exact BOM이나 URDF 변경 근거가 아니다.
