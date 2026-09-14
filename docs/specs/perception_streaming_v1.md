# Streaming perception v1

`tools/perception_streaming.py`의 공개 객체:

- `FrozenDetector(weights, expected_sha256, yolov5, device)` — 기존 P4와 동일한
  tile640/overlap128, batch8, confidence floor .001, NMS .45, FP32, Top-5.
- `StreamingSelector(checkpoint, expected_sha256, device)` — P7c v2의 T16 buffer와 logits.
- `PerceptionPipeline(detector, selector).process(frame_bgr, sequence, timestamp_ns)` —
  RGB 배열부터 후보·motion·최종 rank 및 단계별 시간 반환. rank=None이면 NO_LOCK.

기본 pipeline은 이전/현재 grayscale의 CPU Farnebäck flow를 worker에서 계산하는 동안 GPU detector를
실행한다. detector 뒤에는 후보 의존 GMC와 motion feature를 계산한다. context manager 또는
`close()`로 worker를 종료한다. 검증기의 `--serial-flow`는 동일 연산을 직렬로 실행하는 기준선 옵션이다.

호출은 단일 스트림에서 시간순으로 직렬화해야 한다. source sequence 변경 또는 explicit
`selector.reset()`으로 이력을 초기화한다. 동일 sequence의 비단조 timestamp와 해상도 변경은
오류다. frame은 uint8 BGR이며 현재 frozen detector는 두 변이 모두 640 px 이상이어야 한다.
GT annotation은 API 입력에 없으며 평가 도구만 이를 읽는다. 결과 rank는 해당 호출의
`candidates` 배열을 가리키며 물리적인 target identity/track ID를 의미하지 않는다.

기본 grayscale은 `cv2.cvtColor(BGR, COLOR_BGR2GRAY)`다. 과거 motion v2는 JPEG를
grayscale로 직접 decode했으므로 byte parity 평가에는 optional `gray_frame` 인자로
동일 JPEG의 grayscale decode를 전달한다. JPEG grayscale decode와 BGR→gray의
반올림 차이가 있을 수 있다. 실제 카메라 입력의 정확한 parity는 이번 검증 범위 밖이다.

`verify_perception_streaming.py --mode replay`는 frozen candidate+motion을 streaming
buffer에 공급하고 전체 offline window와 logits 및 선택 rank를 비교한다.
`--mode rgb`는 validation 이미지 전체를 다시 검출하고 motion을 계산한다.
**RGB 실행 환경은 `/home/fair/workspaces/aerial_gym_ws/detector_runs/venv/bin/python`이다.**
frozen candidate cache가 이 환경(Python 3.10 / torch 2.10.0+cu128 / CUDA 12.8 / cuDNN 9.10.2)에서
생성됐고, **다른 환경에서 돌리면 재현되지 않는다.** 2026-09-08에 `datasets/detenv`
(3.8 / 2.4.1+cu121 / 12.1 / 9.1.0)로 돌려 rank parity가 FAIL했는데, 원인은 비결정성이 아니라
cuDNN 9.1과 9.10.2가 서로 다른 컨볼루션 커널을 고르는 것이었다. 같은 60프레임에서 `venv`는 전부
비트 일치하고 `detenv`는 confidence가 최대 45 % 어긋난다. TF32·benchmark·deterministic 플래그로는
메울 수 없다. aerialgym 환경은 pandas가 없어 쓰지 않으며, `venv`에는 pandas가 있다.

실행 예 (저장소 루트):

```bash
/home/fair/workspaces/aerial_gym_ws/detector_runs/venv/bin/python tools/verify_perception_streaming.py \
  --data-root /home/fair/workspaces/aerial_gym_ws/detector_runs/results/nps_detfly_joint_final \
  --run /home/fair/workspaces/aerial_gym_ws/detector_runs/runs/perception_temporal/candidate_motion_transformer_v2 \
  --mode rgb \
  --weights /home/fair/workspaces/aerial_gym_ws/detector_runs/runs/nps_detfly_joint/yolov5s_ms_b8_e30_s0/weights/best.pt \
  --yolov5 /home/fair/workspaces/aerial_gym_ws/datasets/yolov5 \
  --output /tmp/motar_stream_rgb_new.json
```

출력 경로는 미존재 경로여야 한다. 이미지 디코딩은 모델 처리 밖에 별도 측정하고
with_decode에 합산한다. FPS는 1000/mean(ms)이며 카메라 전송·ROS·PPO·비행 제어 비용은
포함하지 않는다. 이 버전은 직렬 처리 API이며 frame drop/queue/backpressure 정책을
실장한 카메라 서비스가 아니다.

P7e crop verifier는 별도 실험 branch로 보존했다. 64D learned feature를 histogram에
대입하려면 temporal selector도 그 feature로 재학습해야 하며 현재 checkpoint와 호환되지 않는다.

## 검증 상태

P7e utility 0.52352는 P7c v2 0.68554보다 낮아 채택하지 않았다.
현재 파이프라인의 기본 선택기는 P7c v2다. 캐시 replay는 validation 2,296프레임의
logits와 rank가 모두 일치했다.

**RGB parity는 2026-09-09에 해소됐다.** 위 실행 환경(`detector_runs/venv`)으로 전체 2,296프레임을
다시 돌린 `stream_rgb_venv_v1.json`이 `rank_mismatches 0`, `rank_parity_pass true`,
`candidate_frame_mismatches 0`이며 selected 2122 / no_lock 174 / false_lock 274 /
center error 1.941796 / loss 106이 cached replay와 전부 일치한다. 이전의 "rank 5개 불일치"는
**환경 불일치의 증상**이었고 모델·threshold는 무관했다. 원인 조사는
[`docs/plans/perception_streaming_findings_2026-09-09.md`](../plans/perception_streaming_findings_2026-09-09.md).

parity 실패를 재현하려면 `datasets/detenv`로 돌리면 된다. gate는 그때 report를 보존한 뒤 exit 1을 반환한다.

이는 여전히 **카메라 배포 승인이 아니다.** offline/online parity는 저장된 JPEG 경로에 한해 PASS이며,
실제 카메라 전송·ROS·PPO는 연결돼 있지 않다.
상세 측정은 [결과](../../results/perception_p7e_streaming_2026-09-08/README.md)를 본다.

2026-09-09 S1에서는 validation 2,296프레임 전체를 직렬·겹침 각 3회 실행했다.
candidate/rank/motion bytes/하위 지표가 모두 정확히 일치했고 decode 포함 mean은
67.91 → 44.49 ms, 평균 처리율은 14.72 → 22.48 FPS였다.
[S1 결과와 실행 artifact hash](../../results/perception_streaming_overlap_s1_2026-09-09/README.md).
