# R1 — 독립 일반 물체 렌더러를 위한 현재 경로 감사

작성일: 2026-09-11. 기준 소스 커밋: `17e4a82f185787c080ef8fea52f3c4cc4870a580`.

| 항목 | 상태 |
|---|---|
| R1 정적 소스 조사 | `STATIC_AUDIT_COMPLETE` |
| 카메라/GPU smoke | `NOT_RUN` |
| R2 shading prototype | `NOT_STARTED` |
| R3–R5 검증·background·benchmark | `NOT_STARTED` |

R1의 완료는 **조사 결과와 근거 파일의 기록 완료**다. 렌더러 구현이나 실행 검증 완료가 아니다.
이번 변경은 문서와 소스 SHA 목록뿐이다. 원래의 appearance overhaul 계획·실험 결과는 변경하지 않는다.

## 1. 범위와 해석 제한

일반 geometry의 flat-color/normal/material/lighting 표현에 관한 정적 감사다.
`navrl_task`, detector, tracking/association, target selection, PPO, controller,
distractor envelope를 수정하거나 실행하지 않는다. 기존 pseudo-RGB 위치는 경계 확인용으로만 기록한다.
학습 모델을 사용하지 않으므로 후속 실험도 영상 통계·geometry 일관성·재현성·계산 비용까지만 해석한다.
모델이 shortcut을 덜 사용한다거나 특정 autonomous task 성능이 개선됐다는 결론은 내리지 않는다.

## 2. R1 요청 항목과 소스 위치

아래 링크의 줄 번호는 기준 커밋의 위치다. 소스가 바뀌면 절 9의 SHA 검사로 구분한다.

| 요청 항목 | 파일 / class / function | 확인 내용 |
|---|---|---|
| 현재 Warp camera | [warp_cam.py](../aerial_gym/sensors/warp/warp_cam.py#L10), `WarpCam.create_render_graph_depth_range()` | pinhole intrinsics·pose·mesh ID를 사용해 depth/range kernel을 호출 |
| normal/face camera | [warp_normal_faceID_cam.py](../aerial_gym/sensors/warp/warp_normal_faceID_cam.py#L10), `WarpNormalFaceIDCam.create_render_graph()` | normal과 face index만 출력. depth 출력은 없음 |
| normal 생성 | [warp_camera_kernels.py](../aerial_gym/sensors/warp/warp_kernels/warp_camera_kernels.py#L70), `DepthCameraWarpKernels.draw_optimized_kernel_normal_faceID()` | `wp.mesh_query_ray()`의 `n`을 world 또는 별도 camera-frame 표현으로 저장 |
| face ID 생성 | 같은 kernel의 ray query와 `face_pixels[...] = f` | 현재 환경의 합쳐진 mesh 내부 triangle index. material/instance ID가 아님 |
| depth 생성 | 같은 파일의 `draw_optimized_kernel_depth_range()` / `_segmentation()` | ray hit 거리 `t`; `calculate_depth`이면 광축 투영 거리로 변환 |
| 현재 RGB/pseudo-RGB | 절 5 | native RGB, debug visualization, task-owned pseudo-RGB를 구분 |
| material 구조 | 절 4 | URDF/Isaac Gym 색은 있지만 확인한 Warp GPU 경로에는 명시적 material table 전달이 없음 |
| GPU batch pipeline | 절 3 | 환경별 `wp.Mesh`; `[env, camera, x, y]` kernel launch와 graph capture |
| benchmark/camera example | 절 6 | 기존 예제는 환경·controller를 포함. 전용 renderer unit test는 검색에서 발견하지 못함 |
| 일반 자산의 최소 경로 | 절 7 | 기존 asset loader 경로와 독립 실험에서 허용할 경로를 구분 |

## 3. Geometry에서 buffer까지

```text
URDF visual geometry + link transform
    ↓ WarpAsset.load_from_file
asset별 triangle mesh + vertex segmentation metadata
    ↓ WarpEnv.add_asset_to_env / prepare_for_simulation
환경별 wp.Mesh(points, indices, velocities-as-segmentation)
    ↓ camera intrinsics + pose → pixel ray → mesh_query_ray
    ├─ depth/range kernel → depth/range, optional segmentation
    │                     normal은 출력하지 않음
    └─ normal/face kernel → normal, triangle index
                          hit 거리 t는 출력하지 않음
```

현재 Warp camera의 geometry 처리는 **triangle ray casting**이며 rasterization이라고 부르지 않는다.

- [WarpAsset.load_from_file()](../aerial_gym/assets/warp_asset.py#L19)는 `URDF.load()`와
  `visual_trimesh_fk()`로 visual geometry를 가져온다. 옵션에 따라 collision geometry를 쓸 수도 있다.
  link transform을 vertices에 적용하고 `tm.util.concatenate()`로 합친다.
- [WarpEnv.prepare_for_simulation()](../aerial_gym/env_manager/warp_env_manager.py#L97)는 환경의 mesh를
  합쳐 `wp.Mesh`를 만든다. `velocities[:, 0]`에는 실제 속도 대신 segmentation metadata를 넣는다.
  `reset_idx()`는 asset root pose로 vertices를 갱신하고 `refit()`한다. 이것만으로 관절별 동적 자세가
  camera mesh에 반영된다고 가정하지 않는다.
- [WarpSensor](../aerial_gym/sensors/warp/warp_sensor.py#L27)는 설정에 따라 camera class를 고르고,
  `set_pose_tensor()`/`set_image_tensors()`로 Torch tensor를 연결한다. `wp.from_torch()`를 사용한다.
- camera는 `(num_envs, num_sensors, width, height)`로 launch하고 pixel을 `[env, cam, y, x]`에 쓴다.
  normal은 vec3다. wrapper의 `capture()`는 초기 graph capture 이후 graph를 재실행한다.
  첫 호출의 준비·컴파일·capture 비용을 정상 프레임 비용으로 간주하지 않는다.

### 이름과 실제 의미

[RobotManagerIGE.prepare_for_sim()](../aerial_gym/robots/robot_manager.py#L196)와
[BaseSensor.init_tensors()](../aerial_gym/sensors/base_sensor.py#L16)에서 같은 사전 key를 재사용한다.

| 경로 | `depth_range_pixels` | `segmentation_pixels` |
|---|---|---|
| 일반 depth/range camera | float depth 또는 range | 설정에 따른 semantic/link label |
| normal/face camera | float normal 3채널 | triangle face index |

key 이름만으로 단위·의미를 추정하면 안 된다. normal camera는 scalar depth를 동시에 제공하지 않는다.

### 수치 계약에서 아직 검증되지 않은 부분

- depth kernel은 `calculate_depth=True`일 때 광축 방향 깊이를 사용한다. normal kernel의 far plane은
  ray 거리 기준이므로 두 camera에 같은 max-range 숫자를 주는 것만으로 clipping이 같아지지 않는다.
- depth 경로의 no-hit 초기값은 `1000.0`, segmentation은 `-2`다. normal 경로는 normal을 0,
  face index를 `-1`로 초기화하고 ray-query 성공 여부를 분기하지 않는다. 실제 miss 출력은 runtime test
  대상으로 남긴다. face index **0은 유효한 첫 triangle**이다.
- [normal camera config](../aerial_gym/config/sensor_config/camera_config/base_normal_faceID_camera_config.py#L7)는
  world-frame normal을 기본으로 한다. camera-frame 분기의 기저·부호·단위 길이는 아직 실측하지 않았다.
- [depth config](../aerial_gym/config/sensor_config/camera_config/base_depth_camera_config.py#L7)는
  후처리 정규화도 포함한다. raw metric depth와 normalized observation을 혼동하지 않는다.

## 4. Material·lighting·ID의 실제 경계

[AssetLoader.load_selected_file_from_config()](../aerial_gym/env_manager/asset_loader.py#L99)는 asset의
`color`, `semantic_id`, `per_link_semantic` 등을 전달한다.
[IsaacGymEnv.add_asset_to_env()](../aerial_gym/env_manager/IGE_env_manager.py#L248)는 색이 없으면
임의 색을 선택하고 `set_rigid_body_color()`로 적용한다.
`_apply_scene_lighting_for_viewer()`의 `set_light_parameters()`도 Isaac Gym viewer/native RGB용이다.

반면 확인한 Warp GPU mesh 생성 경로는 vertices·indices·segmentation metadata만 전달한다.
URDF/trimesh가 material 정보를 보유할 가능성과, 그 material이 GPU ray-shading에 사용된다는 것은 다르다.
현재 경로에서 명시적 face→material table과 material 기반 RGB shading은 확인되지 않았다.

face index는 material ID가 아니고, semantic/link label도 고유 instance ID를 보장하지 않는다.
독립 renderer에서 이들을 지원하려면 scene metadata의 별도 대응표와 테스트가 필요하다.
이는 R2의 미구현 항목이지 기존 기능으로 간주할 사항이 아니다.

## 5. 최종 영상이 만들어지는 세 경로

| 경로 | 소스 | 결과와 이번 범위 |
|---|---|---|
| Native RGB | [IsaacGymCameraSensor.add_sensor_to_env()/capture()](../aerial_gym/sensors/isaacgym_camera_sensor.py#L73) | `IMAGE_COLOR`를 읽어 RGB(A) tensor에 복사. class docstring의 color 미구현 문구와 달리 실제 코드에는 color 취득·복사가 있음. runtime은 미검증 |
| Normal debug image | [save_camera_stream_normal_faceID.py](../aerial_gym/examples/save_camera_stream_normal_faceID.py#L1) | normal과 고정 방향의 내적 절댓값을 grayscale로 만들고 face ID를 colormap으로 시각화. material rendering은 아님 |
| Task-owned pseudo-RGB | [navrl_detector.py](../aerial_gym/task/navrl_task/navrl_detector.py#L866), `render_raw_rgbd()` | depth 기반 배경과 mask flat-fill. 위치 확인용으로만 읽었으며 독립 모듈의 재사용·수정·실행 대상에서 제외 |

normal 예제의 colormap과 side-by-side RGBA 저장물은 debug 산출물이다. 일반 RGB renderer 출력 계약으로
재사용하지 않는다. 이 예제는 face index 0도 검정으로 표시하므로 시각화만 보고 miss라고 판정할 수 없다.

## 6. 기존 benchmark와 테스트

- [examples/benchmark.py](../aerial_gym/examples/benchmark.py#L1)는 `SimBuilder` 환경을 step한다.
  소스 기본값은 `rendering_benchmark=False`인데 안내 문구는 rendering이 기본이라고 설명한다.
  렌더링 분기는 robot/controller를 포함하므로 이번 독립 실험에서 실행하지 않는다.
- normal 저장 예제 역시 `SimBuilder`, robot/controller, environment step을 포함한다. 실행하지 않는다.
- `tests/`에서 `WarpNormalFaceIDCam`, `draw_optimized_kernel_normal_faceID`,
  `warp_camera_kernels`, `WarpCam`을 직접 참조하는 테스트는 검색 시 발견하지 못했다.
  따라서 기존 테스트가 이 renderer 계약을 검증했다고 주장하지 않는다.
- R1에서는 GPU나 시뮬레이터를 초기화하지 않았다. throughput, peak VRAM, output 재현성 수치는 없다.

## 7. 독립 scene 경로와 import 경계

기존 일반 경로는 [SimBuilder.build_env()](../aerial_gym/sim/sim_builder.py#L22) →
[EnvManager](../aerial_gym/env_manager/env_manager.py#L24) → asset/robot/controller 구성이다.
정책 체크포인트를 주지 않아도 순수 renderer만 구성되는 경로는 아니다.

또한 [aerial_gym/__init__.py](../aerial_gym/__init__.py#L1)는 task·control 등을 import하고,
[task/__init__.py](../aerial_gym/task/__init__.py#L122)는 NavRL task도 import/register한다.
따라서 독립 harness에서 기존 package를 일반 import하면 안 된다는 경계가 필요하다.

R2 후보 경로는 다음과 같다. **아직 구현·smoke 검증하지 않았다.**

1. 일반 mesh/URDF를 task registry 없이 읽고 scene pose·camera·ID 대응표를 독립적으로 보유한다.
2. 현재 `warp`만 import하는 범용 `warp_camera_kernels.py`를 소스 파일 단위로 제한적으로 재사용한다.
   package 초기화를 우회하는 로더가 해당 Warp 버전에서 동작하는지 먼저 검사한다.
3. 기존 normal/face 및 depth/range kernel을 독립 buffer에 실행한다. 기존 camera wrapper나
   task·controller를 수정하지 않는다. 실패하면 공유 package를 임의 개편하지 않고 보고한다.

초기 fixture 후보는 [small_cube.urdf](../resources/models/environment_assets/objects/small_cube.urdf),
[cuboidal_rod.urdf](../resources/models/environment_assets/objects/cuboidal_rod.urdf),
[bottom_wall.urdf](../resources/models/environment_assets/walls/bottom_wall.urdf)다.
UAV 자산이나 특정 task semantics는 필요하지 않다. 일반 articulated 예제가 필요해도 초기 범위는
고정 관절 자세의 조립 geometry이며 physics/controller 실행을 의미하지 않는다.

## 8. R2로 넘기는 확인 사항 — 구현 완료 아님

- 소스 로더가 task/controller를 import하지 않는지 검증.
- normal/face와 scalar depth를 별도 buffer로 유지하고 clipping·단위·valid-hit 계약을 맞출 것.
- face→material/instance 대응표가 mesh 합치기 이후에도 맞는지 검증.
- world normal·조명 좌표계, face 0·miss·경계 사례를 검사.
- flat/shaded 모두 RGB는 정확히 3채널. debug buffer는 별도 출력.
- 공간 variance > 0 검사는 여러 방향의 면이 보이는 fixture에 한정.
  균일한 단일 평면의 정상 shading도 공간 variance가 0일 수 있음.
- depth→luminance R² < 0.5는 diagnostic이며 일반적인 성공 gate가 아님.
- 독립 scene에는 physics step이 없으므로 environment step latency는 `N/A`.
  scene update/ray casting/shading 시간과 images/s만 명확히 구분해서 측정.
- 기존 task-owned renderer의 역사적 성능을 독립 flat reference의 성능으로 대체하지 않음.

현재 판정은 **정적 소스상 재사용 후보 있음, 독립 runtime 작동 여부는 미검증**이다.
R2–R5를 승인·착수·완료한 것으로 해석하지 않는다.

## 9. 근거 파일과 재확인

[소스 SHA-256 목록](renderer_r1_audit_2026-09-11.sha256)은 감사한 코드·자산 23개를 고정한다.
프로젝트 루트에서 다음 명령은 파일을 읽기만 하고 Python/GPU/task를 실행하지 않는다.

```bash
sha256sum -c docs/renderer_r1_audit_2026-09-11.sha256
```

불일치는 **현재 파일이 감사 당시와 다르다**는 뜻이지, 렌더러의 기능 실패라는 뜻이 아니다.
향후 정상적인 소스 변경을 막는 회귀 테스트나 runtime fingerprint로 이 목록을 사용하지 않는다.
기준 커밋에서 해당 경로를 읽으면 감사 당시 소스를 확인할 수 있다.

작성 시 검증 결과: 현재 파일 23개와 기준 커밋의 blob 모두 SHA 일치, 보고서 링크 24개 존재 확인,
범용 kernel의 AST signature/import 검사 통과, 문서 상태값·색인 연결·공백 검사 통과.
AST 검사는 파일을 텍스트로 파싱했으며 해당 모듈을 import하거나 kernel을 실행하지 않았다.
