# Track C 사람 검토 인계 — 30분이면 끝납니다

**왜 필요한가.** E3-S의 6.2%가 지금은 전부 AI가 표적을 식별한 것 위에 얹혀 있습니다. 논문 방법 절에
"AI가 표적을 식별했다"고 쓸 수는 없습니다. 사람이 확인하면 이 구멍이 닫힙니다.

**전부 볼 필요는 없습니다.** 53장 중 아래 10장만 확인해도 충분합니다. 근접·중거리·원거리와 실패 사례를
고르게 담았습니다.

## 1. 이 10장을 보세요

| 패널 파일 | GT 거리 | 무엇을 확인하나 |
|---|---:|---|
| `review_material_offset0/review_000901.jpg` | 17.9 m | 굵은 팔 4개 + 가는 착륙 다리 2개 + 안테나 마스트. **자작기 양식** |
| `review_material_offset0/review_001051.jpg` | 30.7 m | 901과 같은 기체가 계속 보입니다 |
| `review_material_offset0/review_005251.jpg` | 61.4 m | 비행 후반에도 같은 기체입니다 |
| `alignment_review/review_004937.jpg` | 61.5 m | 회전익 형상이 분해됩니다 |
| `alignment_review/review_001605.jpg` | 82.3 m | 하늘 배경, 모션 블러가 있는 전형적 프레임 |
| `alignment_review/review_003909.jpg` | 97.0 m | 최원거리에 가깝습니다. 후보 5번이 표적입니다 |
| `review_material_offset0/review_001651.jpg` | 87.2 m | **제외한 프레임**입니다. 나무와 겹칩니다 |
| `alignment_review/review_001778.jpg` | 87.5 m | **제외한 프레임**입니다. 나무 꼭대기와 겹칩니다 |
| `review_material_offset0/review_000001.jpg` | 5.8 m | 표적이 화면 밖입니다. "안 보임"이 정답입니다 |
| `review_material_offset0/review_003451.jpg` | 54.9 m | 후보가 나무입니다. 드론이 아닙니다 |

경로 앞에 `/home/fair/workspaces/aerial_gym_ws/datasets/eth_ds5_cam0_extracted/`를 붙이세요.
패널 아래쪽에 확대 crop이 붙어 있으니 그걸 보시면 됩니다.

## 2. 무엇을 보고 판단하나

> **2026-09-11 정정.** 이전 기준이던 "로터 팔이 6개인가"는 **무효**입니다. drone0도 쿼드로터입니다.
> 프레임 901을 14배로 다시 보면 로터가 달린 굵은 팔 4개, 가는 착륙 다리 2개, 안테나 마스트입니다.
> 제가 다리 둘을 팔로 잘못 셌습니다. 세 대 모두 팔이 4개라 개수로는 구분되지 않습니다.
> → `CORRECTION_2026-09-11_rotor_count.md`

셋 다 쿼드로터이므로 볼 것은 두 가지입니다.

**기체 양식 (근거리).** drone0은 Pixhawk 자작기라 노출된 착륙 다리와 안테나 마스트가 보입니다.
DJI Phantom·Mavic은 사출 성형 소비자 기체이고 Mavic은 팔이 접히며 노출된 다리가 없습니다.

**예측 위치·크기와 연속성 (원거리).** 헤더의 GT 거리·방위와 부합하는지, 앞뒤 프레임에서 이어지는지
봅니다. 다만 이것은 "예측한 자리에 예측한 크기의 항공기가 있다"까지만 확인해 줍니다. 그 점이
drone1·drone2가 아니라는 것까지는 말해주지 못합니다.

확신이 없으면 `unsure`가 정답입니다. 억지로 `yes`로 적지 마세요.

## 3. 서식 채우기

두 파일에 `drone0_visible`, `confidence`, `reviewer`, `notes` 네 칸만 채우면 됩니다.

- `results/eth_ds5_intake_2026-09-10/review_offset0/review_template.csv` (36행)
- `results/eth_ds5_intake_2026-09-10/review_alignment/review_template.csv` (17행)

박스 좌표는 **다시 그리실 필요 없습니다.** AI가 찍은 중심이
`results/eth_ds5_intake_2026-09-10/provisional_review_2026-09-10/alignment_review_claude.csv`에 있고,
사람이 할 일은 "그 중심이 실제 기체 위에 있는가"를 확인하는 것입니다. 어긋난 프레임만 `notes`에
적어 주시면 됩니다.

## 4. 확인이 끝나면

10장 중 하나라도 식별이 틀렸다면 E3-S를 다시 돌려야 합니다. 전부 맞으면 방법 절에
"53프레임 중 10프레임을 사람이 확인했고 불일치 0건"이라고 쓸 수 있고, 그것으로 충분합니다.

현재 배포 상태(2026-09-14): 검수 JPEG/ZIP은 재배포하지 않습니다. 당시 파일명·해시는
historical provenance로 유지합니다. [외부 데이터 취득 및 로컬 검수 안내](../../docs/external_data/ETH_DS5.md).
과거 대비표 경로는 `review_offset0/contact_sheet.jpg`, `review_alignment/contact_sheet.jpg`였습니다.
