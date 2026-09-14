# R2 — 독립 일반 물체 shading prototype

2026-09-11. 선행 근거: [R1 정적 감사](renderer_r1_audit_2026-09-11.md).

| 구분 | 상태 |
|---|---|
| 독립 prototype 구현 | `IMPLEMENTED`; R3의 제한된 fixture에서 runtime 검증 |
| CPU 계약/모의 backend 단위 테스트 | 초기 39개; 현재 렌더러 전용 67개 통과 |
| 실제 Warp kernel 실행·GPU smoke | `TECHNICAL_SMOKE_PASS` |
| R3 renderer validation | [`PASS`](../results/renderer_r3_seed173_comparison_2026-09-11/README.md) |
| R4 / R4b | 상자 fixture 실행 완료, 둘 다 C `FAIL`; 확장 background scene은 미실행 |
| R5 throughput | 기존 11셀 재현 + 셀별 전체 루프 직접 측정 완료; 일반적 상한 아님 |
| 학습·detector·association·policy | 범위 밖, 실행 경로 없음 |

> 후속 상태(2026-09-11): 실제 Warp/GPU 기술 smoke 2회와 사전등록된 R3 검증을 실행했다.
> smoke의 모든 배열 hash가 두 프로세스에서 일치했고, 정식 R3도 최종 PASS했다.
> 절 6의 명령과 `미실행` 표현은 R2 구현 직후의 인수인계 기록이다.
> 전체 후속 상태: [재검증 보고서](../results/renderer_reverification_2026-09-11_1951/README.md).

아래 본문은 구현만 요청받았던 시점의 설계/인수인계 기록이다. 이후 사용자의 실행 승인으로 이루어진
검증 상태는 위 표와 후속 보고서에서 관리한다. 일반화된 렌더링 품질이나 성능 개선 주장은 하지 않는다.
R1 기록은 작성 당시 정적 감사로 보존하며, 후속 구현 상태는 이 문서에서 관리한다.

## 1. 구현된 파일과 경계

| 파일 | 역할 |
|---|---|
| [scene.py](../tools/renderer_validation/scene.py) | triangle mesh, camera, material/light 계약; 일반 상자 fixture |
| [gbuffer.py](../tools/renderer_validation/gbuffer.py) | 기존 범용 kernel 파일만 재사용하는 독립 두-pass ray casting adapter |
| [shading.py](../tools/renderer_validation/shading.py) | 동일 G-buffer의 flat-fill / Lambertian RGB 생성 |
| [run_renderer_validation.py](../tools/run_renderer_validation.py) | 작은 frame 묶음과 실행 receipt를 저장하는 독립 CLI |
| [test_renderer_validation.py](../tests/test_renderer_validation.py) | CPU tensor, AST, fake Warp backend 기반 계약 검사 |

기존 `aerial_gym/`, URDF, 실험 결과, appearance overhaul 계획은 수정하지 않았다.
기존 package import는 task 등록을 동반하므로 사용하지 않는다. 재사용 대상은 R1에서 확인한
`warp_camera_kernels.py` 단일 파일이며 SHA가 달라지면 로더가 거부한다.
로더는 원래 filename을 Warp codegen에 제공하되 package 초기화나 소스 디렉터리 pyc 생성을 하지 않는다.
Warp 자체의 JIT cache는 실제 실행 시 만들어질 수 있으며 receipt에 위치를 기록한다.

`renderer_validation` import만으로 Warp 초기화나 렌더링은 시작되지 않는다.
`WarpGBufferRenderer(...)` 구성 시점부터 실제 Warp 초기화·mesh 생성이 시작된다.
`--help`는 Torch/Warp 없이 실행된다. CLI는 이미 `aerial_gym`을 import한 프로세스 안에서 실행되는 것도 거부한다.

## 2. Geometry·camera·appearance 계약

- 입력 geometry: `MeshScene(vertices[V,3], triangles[F,3], face_material[F], face_instance[F])`.
  단위는 metre이며 degenerate triangle·범위 밖 인덱스·비유한 입력을 거부한다.
- 초기 CLI scene은 두 개의 일반 상자 조립체다. UAV 자산·task semantics·physics는 없다.
  범용 triangle arrays API는 있지만 URDF material importer나 관절 동역학은 이번에 구현하지 않았다.
- 같은 정적 geometry를 N개 scene에 배치하며 각 scene에 카메라 하나를 사용한다.
  카메라는 `+X right, +Y down, +Z forward`, world pose의 quaternion 순서는 `xyzw`다.
  기본 pose는 원점/identity이며 `set_camera_poses()`로 명시적으로 바꿀 수 있다.
- intrinsics는 기존 kernel의 pinhole 규칙을 따른다. 픽셀 좌표는 정수 `(x,y)`, 주점은 `(W/2,H/2)`다.
  임의의 다른 intrinsic matrix, distortion, AA, motion blur, near-plane 재추적은 지원하지 않는다.
- `far_range_m`은 **ray 거리 cutoff**이며 기존 miss sentinel과 충돌하지 않도록 `0 < far < 1000`이다.
- material color는 linear RGB `[0,1]`, `kd`는 `[0,1]`, light direction은 world surface-to-light 방향이다.
  ambient·directional gain은 유한한 비음수다. 재질·조명 배열은 독립 복사한 읽기 전용 NumPy 배열이다.
- scene별 `SeedSequence([seed, scene_index])`를 사용한다. scene 수를 늘려도 기존 prefix의 샘플은 유지된다.
  샘플링은 sequence 시작에 한 번만 수행하며 `render()`/`shade()` 안에서는 RNG를 사용하지 않는다.

## 3. 구현 방식

기존 normal/face kernel을 world-normal 모드로, depth/range kernel을 `calculate_depth=False`로 호출한다.
둘 다 같은 ray-distance cutoff를 사용한다. 후처리에서 camera ray의 광축 투영 계수로 optical depth를 만든다.
따라서 원래 두 camera의 far-plane 의미 차이를 공유 kernel 수정 없이 분리했다.

face hit과 range hit이 다르면 결과를 받아들이지 않고 오류로 중단한다. face 0은 유효한 triangle이다.
miss는 공개 출력에서 `range=depth=0`, `normal=0`, `face=instance=-1`, `valid=false`로 통일한다.
zero-normal hit과 비유한 raw buffer도 거부한다. 유효 normal은 단위 길이로 정규화한다.
반환 tensor는 재사용 raw buffer와 분리되어 다음 프레임에 덮어써지지 않는다.

Shading은 다음 두 모드뿐이다.

```text
flat:       RGB = material_color
lambertian: RGB = material_color × (ambient + kd × directional × max(0, normal · light_direction))
```

RGB는 `[0,1]`로 clamp하고 빈 공간은 검정이다. depth나 instance ID를 shading 계산에 사용하지 않는다.
face ID는 material table 조회에만 사용하며 debug colormap이나 hidden channel을 넣지 않는다.
PBR, shadow ray, texture, learned model은 없다. flat 모드는 역사적 task renderer의 byte 재현이 아니다.

Torch/Warp stream 사이에는 명시적 동기화를 둔다. 아직 graph capture/fusion/throughput 최적화는 하지 않았다.
매 호출의 테이블 전송·수치 검사·동기화 비용도 존재하므로 이 prototype을 최적화된 renderer로 부르지 않는다.

## 4. 출력·재현 기록

CLI는 새 output directory에만 쓸 수 있으며 기존 directory·frame·receipt를 덮어쓰지 않는다.
전체 raw frame 산출물은 256 MiB 이하, 1–16 frames로 제한한다. benchmark나 dataset generator가 아니다.

| 파일 | 내용 |
|---|---|
| `request.json` | 실행 전 소스·scene·camera·seed·appearance 요청 기록, 상태 `INCOMPLETE` |
| `frame_0000.npz` 등 | 아래 별도 배열들. `allow_pickle=False`로 읽을 수 있음 |
| `receipt.json` | 정상 생성 시 `RENDERED_UNASSESSED`, 실제 runtime, 파일/배열 hash |
| `failure.json` | 실패 시 `FAILED_INCOMPLETE`; 성공 receipt는 생성하지 않음 |

배열 shape에서 N은 scene batch다.

- `rgb_flat`, `rgb_lambertian`: `[N,H,W,3]`, float32 linear RGB.
- `depth_m`, `range_m`: `[N,H,W]`, float32.
- `normal_world`: `[N,H,W,3]`, float32 debug normal.
- `face_id`, `instance_id`: `[N,H,W]`, int32 debug metadata.
- `valid`: `[N,H,W]`, bool.

receipt는 source commit·실행 전 dirty 상태·신규 구현 및 재사용 kernel SHA, 실행 프로세스의
`runtime_fingerprint`, Warp 버전, camera/material/lighting/scene 전체 입력, pose, seed를 기록한다.
CUDA 실행에서는 GPU·driver 정보도 수집한다. 실행 중 소스 SHA나 HEAD가 바뀌면 실패로 처리한다.
출력 hash는 파일뿐 아니라 배열의 dtype·shape·C-order bytes 기준으로도 기록한다.
다른 실행 간 재현성 비교에는 배열 hash를 사용하며 압축 container의 byte 일치만으로 판정하지 않는다.

`experiment_verdict=NOT_EVALUATED`, `benchmark=NOT_RUN`, `training=NOT_SUPPORTED`는 고정이다.
물리 환경을 실행하지 않으므로 `environment_step_latency=null`이다.
runtime receipt는 실행 시 생성하는 것이며 이번 구현 작업에서 실제 실행 receipt를 만든 것은 아니다.

## 5. 이번에 수행한 검사

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  /home/fair/miniconda3/envs/aerialgym/bin/python -B \
  -m unittest discover -s tests -p 'test_renderer_validation.py'
sha256sum -c docs/renderer_r1_audit_2026-09-11.sha256
```

CPU 테스트 39개: 입력 계약, face 0/miss, depth 투영, 두-pass 불일치 거부, RGB 3채널,
조명/재질 수치 계산, ID 재번호화 불변성, output 수명, RNG 재현, import 경계, dump 제한,
덮어쓰기 거부, 모의 실행의 receipt/failure 기록과 실행 중 source drift 거부를 검사했다.
backend 호출 검사는 **fake Warp**이며 실제 ray intersection·GPU 성능·영상 품질 실험이 아니다.
신선한 subprocess에서 import가 Warp/Isaac Gym/task를 로드하거나 CUDA를 초기화하지 않는지도 검사했다.

## 6. Sol 인수인계 — 아직 실행하지 않은 작업

사용자가 후속 실행을 지시하면, 먼저 소스를 검토·커밋한 뒤 독립 R2 smoke를 수행한다.
다음은 **미실행 명령 예시**다. 기존 output이 있으면 새 이름을 사용한다.

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  /home/fair/miniconda3/envs/aerialgym/bin/python -B tools/run_renderer_validation.py \
  --output results/renderer_r2_smoke_2026-09-11 \
  --device cuda:0 --seed 0 --num-scenes 1 --width 160 --height 120 --frames 2
```

첫 실제 실행에서 확인할 것:

1. Warp 1.0.0의 source loader/codegen, mesh 생성, 두 camera kernel 실행 가능 여부.
2. ray hit·normal 부호·camera pose·far cutoff·triangle 경계의 실제 일관성.
3. NPZ의 RGB와 debug buffer를 분리해 검수. 기본 장면에서도 잘 보이는지는 아직 미검증.
4. 별도의 새 output에서 같은 seed/runtime으로 실행하여 배열 hash 비교.
5. 실패 시 `failure.json`과 오류를 남기고 공유 renderer/task를 임의 수정하지 않음.

R3의 평가 기준과 R4/R5 실험은 별도 작업이다. 이 후속 작업에도 학습·detector·association·policy는 포함하지 않는다.
