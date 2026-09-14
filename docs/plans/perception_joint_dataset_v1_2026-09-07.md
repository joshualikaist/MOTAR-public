# NPS+Det-Fly detector dataset v1 실행 계약

작성: 2026-09-07

상태: **COMPLETE — split 봉인·통합 학습 데이터 구성·전수 검증 완료**

상위 정본: `docs/plans/perception_final_implementation_plan_2026-09-07.md`

## 목적과 제약

P3에서 NPS-only detector의 Det-Fly native 4K AP30이 0.0035였고 ÷4 scale-matched AP30이
0.2382로 회복됐다. 다음 detector는 NPS+Det-Fly 합동 학습과 native-scale 표적을
포함해야 한다. 이 문서는 그 전제인 (1) 누수 방지 split과 (2) 이미지를 전체
복제하지 않는 통합 학습 dataset을 고정한다.

Det-Fly는 video ID를 배포하지 않고 `010`/`020` source group과 숫자 frame ID만
남긴다. 따라서 frame 단위 무작위 split은 금지한다. 또한 이 split은 P3 결과가
존재한 뒤에 만들어졌다. 실행 코드는 P3 prediction/report를 읽지 않고 source group,
frame ID, 개수만 사용하지만, 이 순서상 한계를 없던 일로 소급하지 않는다.

## 1. 누수 방지 split

split 규칙은 성능·confidence·box·배경 pixel을 보지 않는 구조 함수로 구현했다.

1. 더 큰 공식 source group을 전체 test로 선택한다.
2. 남은 group은 인접 frame ID 사이의 가장 큰 자연 단절에서 나눈다.
3. 더 큰 쪽을 train, 작은 쪽을 validation으로 사용한다.

| split | Det-Fly 구간 | images | boxes | 비고 |
|---|---|---:|---:|---|
| train | `010`, ID ≤ 4333 | 4,200 | 4,200 | development lower side |
| val | `010`, ID ≥ 4752 | 2,158 | 2,158 | full-width 비정상 box 1개는 index에 유지 |
| **sealed test** | `020` 전체 | **6,913** | **6,912** | negative frame 1장 포함 |

train/val 경계는 `4333 → 4752`로 ID 차이 419이며 중간 418개 ID가 배포본에
없다. test source group은 development와 완전히 분리됐다.

검사 결과:

- 13,271 JPEG SHA-256은 모두 고유하고 split 간 exact duplicate는 0이다.
- 64×36 RGB thumbnail exact hash도 split을 가로지 않았다.
- dHash는 하늘 처럼 저정보량인 frame에서 다수 충돌했다(Hamming ≤2: 2,963). 이를
  근접 중복 근거로 오인하지 않고 split 판정에 사용하지 않았다.
- 봉인 test manifest에는 6,913장이 있고 SHA-256은
  `3ae8bfc06e4df09d27c9881e481170b745dd9abb0559a0b579586550b24f086f`이다.
- 학습 YAML은 `test:` key를 두지 않았다. model/threshold/anchor 선택은 train/val로만
  하고, 봉인 test는 최종 model에서 한 번만 열어야 한다.

## 2. 통합 학습 dataset

NPS는 기존 clip-level split을 그대로 유지한 640 tile을 `train.txt`/`val.txt`에 절대
경로로 참조한다. NPS test clip은 기존 계약대로 tile 학습 dataset에 들어가지 않는다.

Det-Fly train/val은 P3와 같은 native 640 px, overlap 128 px 기하를 쓴다. GT box의 60%
이상을 보존하고 보이는 변이 4 px 이상인 모든 positive tile과, GT와 교차가 0인
결정론적 negative 1장/frame을 만든다. 4K 원본은 복제·재압축·수정하지 않고,
표현에 필요한 640 crop만 JPEG quality 90으로 생성했다.

| split | NPS references | Det-Fly tiles | total samples | boxes | positive / negative samples |
|---|---:|---:|---:|---:|---:|
| train | 19,659 | 10,165 | **29,824** | 20,573 | 18,889 / 10,935 |
| val | 4,096 | 5,211 | **9,307** | 6,921 | 6,002 / 3,305 |
| total | 23,755 | 15,376 | **39,131** | 27,494 | 24,891 / 14,240 |

Det-Fly tile은 train positive/negative `5,965/4,200`, val `3,054/2,157`이다. `0106343`의
3840×17 full-width annotation이 있는 val frame 1장은 잘못된 negative로 바꾸지 않고 tile
생성에서 frame 전체를 제외했다. 원 annotation은 index와 sealed provenance에 남아 있다.

## 산출물·검증

| 산출물 | 경로 / SHA-256 |
|---|---|
| YOLOv5 training YAML | `datasets/nps_detfly_joint_v1/joint_detector.yaml` / `08025f79…` |
| train / val file list | `datasets/nps_detfly_joint_v1/{train.txt,val.txt}` |
| split / sealed test | `datasets/nps_detfly_joint_v1/manifests/{detfly_split.jsonl,detfly_test_sealed.jsonl}` |
| joint sample manifest | `datasets/nps_detfly_joint_v1/manifests/joint_samples.jsonl` / `b6d8282e…` |
| split receipt | `datasets/nps_detfly_joint_v1/receipts/detfly_split_receipt.json` / `f4be5d56…` |
| dataset receipt | `datasets/nps_detfly_joint_v1/receipt.json` / `ef9babda…` |

`tools/verify_nps_detfly_joint_dataset.py`가 39,131 sample 전체에 대해 파일·SHA-256,
640×640 크기, YOLO label domain, box 개수, source-unit split, exact image split, sealed test 유입,
YAML test key를 fail-closed로 검사했고 **PASS**했다. YOLOv5 `check_dataset` 결과도
train/val을 정상 해석하고 `test=None`이다. 생성물은 917 MB, 남은 디스크는 약 14 GB다.

다음 단계인 통합 detector 학습은 이 실행에 포함하지 않았다. 학습 전에 model
초기화, multi-scale 범위, anchor 산출, epoch/batch, validation 선택 규칙을 별도로 동결해야 한다.
