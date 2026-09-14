# ref5in payload packaging 계약 — 2026-08-23

상태: **부품은 미조립, carrier/cooling/storage/DC-DC와 배터리 모델 미동결**.
따라서 아래는 CAD를 시작하기 위한 envelope 계약이며, “현재 프레임에 실제로 들어간다”는 판정이 아니다.

## 기준과 좌표

- frame 기준점: AOS 5 V5.1의 geometric center, `x` 전방, `y` 좌측, `z` 위쪽.
- 최종 기준점은 조립 후 측정한 CG로 바꾼다. 지금의 frame center를 simulation CG로 승격하지 않는다.
- 모든 치수는 외곽 envelope로 기록한다. PCB nominal size만 적고 connector·케이블 굽힘 공간을 빼먹지 않는다.
- 각 부품은 `part_id`, 모델/revision, source URL 또는 실측 사진, mass, bbox, mount holes, datum,
  FOV/keepout, power/thermal을 함께 기록한다.

## 현재 확보된 envelope

| 부품 | 질량 | 외곽/시야로 사용할 값 | 상태 |
|---|---:|---|---|
| Livox Mid-360 | 265 g | 65×65×60 mm; 360° horizontal × 59° vertical FOV | 공식값; [3D model/사양](https://www.livoxtech.com/de/mid-360/specs) |
| Intel RealSense D435i | 72 g nominal | 제조사 외곽·mount hole·케이블 방향을 CAD/실물로 확인해야 함 | 질량만 공식값; bbox pending |
| Jetson Orin NX SOM | 28 g | SOM 외곽만 확정; carrier/heatsink/fan/storage/DC-DC 제외 | incomplete |
| Pixhawk 6C Mini | 39.2–46.8 g | revision별 board/connector envelope pending | revision 미동결 |
| AOS 5 V5.1 frame | 165±5 g | 228 mm wheelbase, 180×140×34 mm frame dimension | provisional reference |
| battery | 미정 | exact model의 bbox, connector exit, strap keepout 필요 | pending |

Mid-360는 FOV를 가리는 상부 plate, standoff, cable, heat spreader를 “부품 밖”으로 취급할 수 없다.
D435i도 카메라 전방과 stereo baseline 앞을 가리는 mount를 허용하지 않는다.

## 필수 CAD gate

1. **prop swept volume**: 선택한 motor/prop의 회전 disc와 모든 payload/mount/wire가 겹치지 않는다.
   5.1-inch prop를 쓸 경우 단순 frame 계산만으로 통과시키지 말고 실제 motor hole·blade geometry를 쓴다.
2. **LiDAR gate**: Mid-360 중심에서 360° 수평 ray와 59° 수직 cone에 frame arm, battery, mast, cable,
   D435i, Orin cooling이 들어오면 각 가림 각도와 거리별 가림을 기록한다.
3. **camera gate**: D435i의 depth/RGB FOV와 baseline 앞에 frame/센서/배선이 들어오지 않는다. 가림이
   있으면 각도·비율을 수치로 기록한다.
4. **connector/maintenance gate**: 모든 전원·통신 connector의 삽입 방향과 최소 bend radius, 탈착
   공간을 표시한다. “bbox가 안 겹침”만으로 통과시키지 않는다.
5. **CG gate**: nominal/max BOM의 CG `(x,y,z)`와 frame 기준 차이를 계산한다. 축별 허용치를 사전에
   선언하고, 초과하면 mount 위치를 바꾸거나 simulation mass/CG contract를 별도 변경한다.
6. **thermal/power gate**: Orin cooling exhaust가 LiDAR/camera를 가리지 않고, Mid-360 startup 18 W
   peak와 compute/FCU/DC-DC 전원을 포함한 peak/steady budget을 따로 계산한다.

## 입력 파일 형식

최종 CAD 전에 아래 표를 CSV/YAML로 채운다.

```text
part_id,model,revision,mass_g,bbox_x_m,bbox_y_m,bbox_z_m,
mount_datum,connector_keepout_m,fov_keepout,nominal_power_w,peak_power_w,
cg_x_m,cg_y_m,cg_z_m,source_or_measurement_sha256,status
```

`status`는 `official`, `measured`, `estimated`, `pending` 중 하나다. `estimated` 값은 AUW/CG/관성
합산에 사용하지 않고 sensitivity bound로만 표시한다.

## 승격 판정

- 모든 부품이 `official` 또는 `measured`이고 exact revision/serial이 있다.
- CAD에서 prop/LiDAR/camera/connector/thermal gate가 PASS다.
- nominal/max AUW, CG, inertia 입력과 사진/도면 SHA가 일치한다.
- 질량이 1.20 kg을 넘거나 prop envelope가 0.28 m proxy를 넘으면 **실패가 아니라 contract 변경
  후보**로 별도 등록한다. 기존 checkpoint에 조용히 적용하지 않는다.

이 조건 전까지 사이트·논문에서의 표현은 “provisional packaging reference”이며, 실제 플랫폼이나
sim-to-real 검증 완료로 표현하지 않는다.
