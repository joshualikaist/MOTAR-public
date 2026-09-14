# P3 Det-Fly zero-shot 실행 계약

작성: 2026-09-07

상태: **COMPLETE — 13,271장 두 arm 전수 평가 및 receipt 검증 완료**

상위 정본: `docs/plans/perception_final_implementation_plan_2026-09-07.md`

## 질문과 금지 사항

NPS-Drones로만 학습한 detector가 가중치나 threshold 재조정 없이 Det-Fly에 얼마나 전이되는지 측정한다.
이 실행 전·중에는 Det-Fly로 재학습, fine-tuning, threshold 선택, size bin 변경을 하지 않는다. 결과가 낮다는
이유만으로 사후 실패선을 만들지 않고 전체 분포와 오류 유형을 먼저 보고한다.

## 동결 입력

| 입력 | 경로·revision | SHA-256 |
|---|---|---|
| NPS YOLOv5s checkpoint | `detector_runs/runs/nps_det/s_tiles640_b8_e40/weights/best.pt` | `ccd65dc37ec2fce765e0860232922287323c49179a4ea0e12cb6ad1b4f28f580` |
| NPS split receipt | `datasets/nps_yolo/receipt.json` | `8887d35752804a0a873e30e2c3e5acb24f1cda86cbce9c5d4efefcd7024366de` |
| NPS tile receipt | `datasets/nps_yolo_tiles640/receipt.json` | `d87df74fcc739853a9e1c0ebd1a2aa3a46faf06ed92add8dd0e65413fa754dba` |
| YOLOv5 source | `datasets/yolov5` | git `35b48237aef6d71ca9de2c5dea345d7536eb7fa7` |
| Det-Fly official remote manifest | 13,271 XML + 13,271 JPEG, 18,849,340,854 bytes | `306866af38d80374d86253b1163089f4619ea0b98378ddbad7cee88371a13088` |

Det-Fly는 공식 GitHub README가 연결하는 두 공개 OneDrive 폴더에서 익명 read-only API로 받는다. 원격
파일명·크기·QuickXOR manifest와 로컬 각 파일의 SHA-256을 receipt에 남긴다. 공식 문서의 9.34 GB는
압축 배포 크기로 보이지만, OneDrive가 보고한 풀린 파일 합계는 18.849 GB다.

## 데이터 계약

배포 XML은 Pascal VOC 형식이며 3840×2160 JPEG와 `UAV` box, `difficult`, `truncated`를 포함한다.
`tools/prepare_detfly_dataset.py`는 이미지를 복사하거나 재압축하지 않고 다음을 fail-closed로 검사한다.

- XML–JPEG 1:1 stem pairing과 두 파일 수
- XML 크기와 JPEG SOF 크기 일치
- 단일 `UAV` class, 유효한 frame 내부 box, binary flags
- 각 XML/JPEG SHA-256과 안정적인 content-manifest SHA-256

공식 README는 sky·urban·field·mountain 네 배경을 설명하지만 배포 XML에는 배경 field가 없고 `010/020`
두 폴더 각각에도 여러 배경과 viewing angle이 섞여 있다. 근거 없는 픽셀 분류로 배경 정답을 만들지 않는다.
P3에서는 `background=UNAVAILABLE_IN_DISTRIBUTED_METADATA`를 기록하고 `source_group=010/020`을 대체
분해로 보고한다. 공식 매핑을 확보하면 raw prediction을 재추론하지 않고 그 매핑으로 재집계한다.

## 두 arm — scale-matched 조건 추가 (2026-09-07, **예측 열람 전** 동결)

**추가 사유(어노테이션만 보고 결정. 예측은 한 건도 보지 않았다).** 배포된 Det-Fly 어노테이션 13,271개를
전수 집계한 결과, 표적 크기가 NPS 학습 분포와 거의 겹치지 않는다.

| 640 px 창 기준 box 대각 | p25 | 중앙 | p75 | p95 |
|---|---:|---:|---:|---:|
| NPS 학습 타일 (14,608 box) | 19 | 25 | 36 | 62 |
| Det-Fly native 4K (13,270 box) | 74 | **109** | 165 | 298 |

**Det-Fly 표적의 75 %가 NPS 학습 표적의 95 %보다 크다.** 중앙값 배율은 4.3배다. 따라서 native scale
단독 평가는 "도메인이 달라서 못 찾는가"와 "작은 물체 검출기가 큰 물체를 못 찾는가"를 분리하지 못한다.
낮은 mAP가 나와도 원인을 귀속할 수 없으므로, 결과를 보기 전에 두 번째 arm을 동결한다.

| arm | 전처리 | 답하는 질문 |
|---|---|---|
| **A. native scale** (`--downscale 1.0`) | 3840×2160 그대로 640 창 sliding | 이 detector를 4K 공대공 영상에 **그대로 배치**하면 어떤가 |
| **B. scale-matched** (`--downscale 4.0`) | INTER_AREA로 960×540 축소 후 640 창 | 실패가 **도메인 때문인가 크기 때문인가** |

arm B의 배율 4.0은 관측된 중앙값 비 4.36에서 골랐고(축소 후 Det-Fly 대각 중앙 ≈ 27 px, NPS 25 px),
960×540은 NPS의 원본 1920×1080 → 640 타일 구조와도 맞는다. **두 arm 모두 box를 원본 4K 픽셀로 되돌린 뒤**
NMS·IoU·size bin을 적용하므로 지표의 의미가 같다. threshold·bin·checkpoint는 arm 사이에서 동일하다.

**판정 규칙(동결)**: arm A와 B의 mAP50 차이가 arm A 값보다 크면 실패의 지배적 원인은 **크기**이고,
두 arm이 비슷하게 낮으면 **도메인**이다. 어느 쪽이든 "NPS 단독 학습 detector는 Det-Fly에 그대로 쓸 수 없다"는
결론은 같지만, 처방이 다르다 — 전자는 다중 스케일 학습·앵커 재계산, 후자는 두 데이터셋 합동 학습이다.

### arm B 배관 검증에서 잡은 결함 (결과 판정 아님)

arm B를 추가한 뒤 이미 받은 12장으로 **배관만** 점검했다. 첫 구현이 각 축을 타일 크기로 따로 clamp해
**종횡비를 깨뜨렸다** — 2160/4 = 540이 640으로 올라가 y는 3.375배, x는 4.0배로 줄었고, 되돌릴 때 둘 다
4.0을 곱해 모든 예측이 엉뚱한 자리에 놓였다. 균일 축소 후 **bottom-right 패딩**으로 고쳤다(패딩이 원점을
바꾸지 않으므로 역변환은 그대로 ×4). raw record의 `width`/`height`도 작업 해상도가 아니라 원본 4K를
기록하도록 고쳤다. **threshold·size bin·checkpoint는 건드리지 않았다.**

수정 후 같은 이미지에서 최상위 예측이 GT와 같은 자리에 온다(GT `[1146,760,1333,808]`,
예측 `[1285,762,1312,798]`). 이 12장은 표본이 아니라 좌표 검증용이며 P3 결과가 아니다.

## 추론·평가 조건

NPS 모델은 표적 native scale을 보존한 640 px tile로 학습됐다. Det-Fly 3840×2160 전체를 640으로 축소하면
표적이 1/3 이하가 되므로, 같은 native-scale 조건의 sliding window를 사용한다.

| 항목 | 고정값 |
|---|---:|
| tile / overlap | 640 / 128 px |
| frame당 tiles | 32 |
| batch | 32, FP32 |
| inference confidence floor | 0.001 |
| 보고 operating confidence | 0.25 |
| tile NMS / image NMS IoU | 0.45 / 0.45 |
| max detections | tile당 100, image당 top 300 |
| match IoU | 0.3과 0.5 모두 보고 |
| equivalent-side size bins | `[0,8)`, `[8,12)`, `[12,20)`, `[20,32)`, `[32,64)`, `[64,128)`, `[128,∞)` px |

`difficult`를 포함한 전체 annotation을 primary로 보고하고 non-difficult/difficult를 별도 분해한다. 전체와
source group별 P/R/F1/AP, GT pixel-size별 P/R/F1/AP를 쓴다. size slice에서는 다른 size GT에 매칭된
detection을 ignore하고, 어느 GT에도 매칭되지 않은 detection은 각 slice의 FP로 센다. AP는 confidence
floor와 image top-300 이후 후보에서 계산하므로 그 아래 점수가 존재하면 lower bound다.

## 결과 (2026-09-07)

모든 수치는 frozen NPS checkpoint와 사전 등록한 threshold에서 13,271장 전체를
평가한 결과다. GT는 13,270 box이며 1장은 배경 negative frame이다.

| arm | tiles | IoU | P | R | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|
| A. native 4K | 424,672 | 0.3 | 0.0097 | 0.1668 | 0.0183 | **0.0035** |
| A. native 4K | 424,672 | 0.5 | 0.0051 | 0.0873 | 0.0096 | **0.0010** |
| B. scale-matched ÷4 | 26,542 | 0.3 | 0.2416 | 0.3503 | 0.2860 | **0.2382** |
| B. scale-matched ÷4 | 26,542 | 0.5 | 0.1612 | 0.2337 | 0.1908 | **0.1126** |

IoU 0.3 AP 차이는 `0.2382 - 0.0035 = 0.2347`로 arm A AP 자체보다 크다. 따라서
사전 등록 판정은 **SIZE-DOMINANT ZERO-SHOT FAILURE**다. NPS-only detector를 4K Det-Fly에
그대로 배치하는 경로는 기각한다. ÷4에서 AP가 크게 회복되지만 IoU 0.3 recall은
0.3503에 머물고 operating point의 FP도 14,587개이므로, scale matching은 완성 detector가
아니다. 다음 detector는 NPS+Det-Fly 합동 학습과 multi-scale/anchor 재설계를 하되, Det-Fly
test split은 모델·threshold 선택에서 봉인해야 한다.

공식 배포 metadata에 sky·urban·field·mountain mapping이 없어 배경별 수치는 만들지
않았다. 대신 source group `010`/`020`, GT pixel-size, difficult/truncated 분해는
report에 모두 남겼다.

## 산출물과 완료 조건

| 산출물 | 경로 |
|---|---|
| 다운로드 receipt | `datasets/detfly/download_receipt.json` |
| 검증 index와 receipt | `datasets/detfly_index/{index.jsonl,receipt.json}` |
| 전체 raw predictions | `detector_runs/results/nps_to_detfly_zeroshot/predictions.jsonl.gz` |
| 지표·runtime report | `detector_runs/results/nps_to_detfly_zeroshot/{report.json,receipt.json}` |
| scale-matched raw/report | `detector_runs/p3_detfly_scale_matched/{predictions.jsonl.gz,report.json,receipt.json}` |

P3 완료 조건은 모두 충족했다. native raw/report SHA-256은 각각
`a98e952d…` / `ca45d1f8…`, scale-matched는 `4b0277a6…` / `421aeed8…`이며
각 receipt와 독립 재계산값이 일치한다.


## 결과와 판정 (13,271장 전수, 두 arm)

| IoU 0.5 | arm A native | arm B scale-matched ÷4 |
|---|---:|---:|
| AP | **0.0010** | **0.1126** |
| 정밀도 | 0.0051 | 0.1612 |
| 재현율 | 0.0873 | 0.2337 |
| 타일 | 424,672 | 26,542 |

**동결 규칙 적용**: 두 arm의 차이는 0.1116이고 arm A 값은 0.0010이다. 차이가 arm A의 **111배**이므로
판정은 **크기(SIZE)** 다. 처방은 다중 스케일 학습·앵커 재계산이며, 합동 학습은 그다음이다.

**크기별 재현율이 근거다.**

| 원본 등가변 | GT | arm A | arm B |
|---|---:|---:|---:|
| 12–20 px | 243 | 0.177 | 0.062 |
| 20–32 px | 1,245 | 0.257 | 0.177 |
| 32–64 px | 4,668 | 0.170 | **0.320** |
| 64–128 px | 4,961 | **0.0008** | 0.246 |
| 128 px 이상 | 2,151 | **0.000** | 0.071 |

native arm은 64 px를 넘는 순간 **완전히 실패**한다(4,961개 중 4개 검출, 2,151개 중 0개). 그런데 Det-Fly
박스의 **54 %가 64 px를 넘는다.** 축소해서 학습 분포로 들여보내면 그 구간 재현율이 0.246으로 돌아온다.
즉 이 detector가 Det-Fly에서 실패하는 지배적 이유는 배경·카메라가 아니라 **표적이 학습 때보다 크다는 것**이다.

**도메인 성분도 남는다.** arm B의 0.1126은 NPS 자체 검증 0.579의 **19 %**이고 사전등록 문턱 0.33에 못 미친다.
크기를 맞춰도 5분의 1로 떨어지므로, 크기가 지배적이되 유일한 원인은 아니다.

**부차 관측**: 어려움 표시 990개는 arm B 재현율 0.010으로 사실상 전부 놓친다. 그룹별은 010 0.222 /
020 0.244로 차이가 작다. arm A 정밀도 0.0051은 오검출 398만 건에서 나온 값으로, 학습 분포 크기의
배경 텍스처를 확신하는 증상이다.

**결론**: NPS 단독 학습 detector를 Det-Fly에 그대로 쓸 수 없다. 다음 실험은 다중 스케일 학습이며,
그것이 크기 성분을 제거한 뒤에 남는 격차가 진짜 도메인 격차다.
