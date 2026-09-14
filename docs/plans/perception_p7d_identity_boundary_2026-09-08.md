# P7d learned appearance / ReID 착수 조건

상태: DATA REQUIRED — 학습 미실행

2026-09-08 실제 다운로드 시도 결과 공식 Google Drive/Baidu 링크가 모두 실패했다.
[접근 확인·재개 조건](perception_p7d_access_check_2026-09-08.md)을 참조한다.

P7b/P7c의 출력은 현재 frame의 UAV 후보 선택이다. 동일 표적 association을 학습하려면
원본 영상과 `(sequence_id, frame_index, timestamp, track_id, xyxy, visibility)`가 필요하다.
데이터셋 이름이나 파일명만으로 track identity를 추정하지 않는다.

데이터를 확보하면 먼저 다운로드 출처·이용 조건·원본 hash·annotation schema·이미지와
annotation 대응·FPS와 sequence 경계·track ID의 sequence 내 일관성을 검사한다.
분할은 최소한 source sequence 단위로 분리한다. 개체의 전역 ID가 없다면 물리적으로
같은 UAV가 split 사이에 없는지 검증할 수 없으므로 identity-disjoint라고 주장하지 않는다.

학습 계약은 데이터 감사 후 별도로 고정한다. 같은 track의 시간상 분리된 crop은 positive,
다른 track crop은 negative로 사용하며 가림·작은 crop과 ID 오류 처리 규칙을 고정한다.
기존 parameter-free appearance와 learned appearance를 동일 detector 후보에서 비교한다.
IDF1/ID switch/fragmentation은 annotation의 identity 의미를 확인한 뒤 계산한다.

MOT-FLY는 후보 데이터셋이며 이번 구현에서는 다운로드·라이선스·세부 사양을 검증하지 않았다.
현재 NPS 변환본으로 ReID를 학습하거나 P7b/P7c proxy를 실제 ID-switch로 보고하지 않는다.
Memory bank, KF→PPO 연결과 실기 latency 검증은 이 단계의 완료 산출물이 아니다.
