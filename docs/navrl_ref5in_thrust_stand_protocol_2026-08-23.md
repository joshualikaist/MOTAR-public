# ref5in thrust-stand 측정 계약

상태: **장비·실기체 미보유, 측정 전**. 이 문서는 수치를 만들어내는 시뮬레이션이 아니라, 실제
선택 부품의 추력 계약을 닫기 위한 원자료 형식이다.

## 고정 조건

측정마다 아래 조건을 바꾸지 않거나, 바꾼 경우 별도 `condition_id`로 분리한다.

```text
motor: exact model / KV / serial
prop: exact manufacturer, diameter, pitch, blade count, revision
ESC: exact model, firmware, timing/PWM/DShot settings
battery: exact model, cell count, starting SOC, resting voltage, serial
test stand: load-cell model/range/calibration date, tachometer method
ambient: temperature, pressure, humidity
sample: >=100 Hz synchronized force/current/voltage/RPM/temperature
```

단일 모터를 stand에 장착하고, 프로펠러·모터 회전방향·공기 흐름·guard 유무를 기록한다. 실제
기체에 쓰는 조합과 다른 prop/ESC/배터리 결과는 후보 참고자료일 뿐 ref5in 계약값이 아니다.

## 한 run의 순서

1. load cell zero와 tare를 확인하고 calibration ID를 기록한다.
2. 모터를 장착하지 않은 상태에서 10 s 센서 baseline을 기록한다.
3. 0%에서 10% 간격으로 20–100% command를 올리고 각 지점에서 5 s 안정화 후 10 s 기록한다.
4. 목표 부근을 2.943 N, 9.60 N 주변에서 각각 3회 반복한다. 목표를 넘기기 위해 정격을
   무리하게 초과하지 않는다.
5. hover proxy(2.943 N)와 candidate maximum(9.60 N)을 30 s 유지하는 별도 run을 한다.
6. 모든 run에서 force, current, voltage, RPM, ESC temperature, motor temperature, timestamp를
   같은 clock으로 저장한다. 배터리 전압은 시작·종료 resting voltage와 load voltage를 모두 남긴다.
7. 이상 진동, runaway, sensor saturation, thermal limit, voltage cutoff가 발생하면 즉시 중지하고
   `termination_reason`을 기록한다. 실패 run을 삭제하거나 평균에서 숨기지 않는다.

## 원자료 CSV 계약

파일은 UTF-8 CSV이며, 다음 필드를 반드시 포함한다.

```text
trial_id,condition_id,utc_start_s,t_s,command_0_1,thrust_n,current_a,voltage_v,rpm,
motor_temp_c,esc_temp_c,ambient_temp_c,load_cell_zero_n,calibration_id,termination_reason
```

`trial_id`는 독립 측정 단위다. 같은 run의 샘플을 독립 표본으로 세지 않는다. `termination_reason`
은 정상 완료도 `complete`로 명시하며, 모든 결측·saturation·timestamp 역행은 원자료 품질 오류다.

## 사전 판정 규칙

아래를 모두 만족해야 `THRUST_CONTRACT_PASS`다.

- calibration ID와 exact BOM(모터·prop·ESC·배터리)이 manifest와 일치한다.
- force/current/voltage/RPM timestamp가 단조 증가하고 결측·saturation이 없다.
- hover 반복 3회의 평균 추력이 2.943 N ±5% 안이다.
- 9.60 N 목표 반복 3회가 모두 안전하게 완료되고, 각 반복의 전류·전압 sag·온도가 기록됐다.
- 30 s 유지 중 thermal limit·battery cutoff·load-cell saturation이 없다.
- 정적 최대추력은 단순 한 점이 아니라 command→thrust/current/RPM 곡선으로 보존된다.

하나라도 실패하면 `THRUST_CONTRACT_INCONCLUSIVE`이며, 시뮬레이터의 9.60 N을 자동으로
수정하지 않는다. 측정된 곡선에서 새 actuator contract를 별도 제안하고, 그 후 smoke→platform
gate→학습 순서로 다시 검증한다.

## 기록해야 할 파생값

- hover thrust/current/voltage/RPM의 trial 평균과 p10/p50/p90;
- 9.60 N 근처의 current, loaded voltage, sag, motor/ESC temperature rise;
- command→thrust monotonicity와 fit residual;
- motor-up/down step response에서 10–90% rise/fall time;
- 네 모터를 합친 nominal total thrust와 1.20 kg AUW 기준 T/W;
- 측정 장비·firmware·환경·원자료 SHA-256.

측정 전까지 현재 `motor_time_constant_s=0.04`, `max_thrust_per_motor_n=9.60`은 합성 설계점으로만
표시한다.
