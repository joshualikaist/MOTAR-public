# P7d 데이터 접근 확인

2026-09-08 확인. 상태: BLOCKED — 공식 MOT-FLY 다운로드 링크 불가.

사용자가 요청한 다음 두 단계는 (1) MOT-FLY 확보·identity 검증,
(2) learned appearance/ReID 구현·학습이다. 1번의 다운로드가 실패해 실제
annotation 감사 및 2번의 데이터 기반 구현·학습은 완료되지 않았다.

## 확인한 경로

- 공식 저장소: https://github.com/CZC-123/MOT-FLY
- Google Drive file ID `1GiWLF8B18FGDcCSuSuvGokczCkP_NEgo`:
  view 페이지는 페이지 없음, uc download 및 usercontent download는 HTTP 404.
- Baidu `1eS84Ooz0URojz1tAJNZ5Eg`, 공개 코드 `pe53`:
  HTTP 200이지만 HTML title은 `百度网盘-链接不存在` (링크 없음).
- 공식 기존 이슈 https://github.com/CZC-123/MOT-FLY/issues/1 :
  2024-12-27 링크 만료 신고; 확인 당시 댓글에도 다운로드 실패 보고만 있고 대체 링크 없음.
- 저장소 contents에는 README, 설명 이미지 폴더, 소개 mp4만 존재.
- 웹 대체 배포처 검색과 Hugging Face dataset API의 `MOT-FLY`, `motfly`
  검색에서 사용 가능한 원본 배포처를 찾지 못함. 이는 모든 사본의 부재를 증명하지 않는다.

다운로드 파일, 추정 track ID, 학습 checkpoint를 만들지 않았다.
NPS refined ID를 검증된 실제 identity로 대신 사용하지 않았고 test는 열지 않았다.

## 재개 입력

작동하는 MOT-FLY 배포 링크 또는 로컬 원본 `img1/`, `gt/gt.txt`, `seqinfo.ini`가 필요하다.
데이터 확보 후 크기·hash·이용 조건·image/GT 대응·sequence와 track ID를 검사한다.
공식 README는 일부 sequence 이름을 train/test 양쪽에 표시하므로 공식 분할만으로
source-video 독립성을 가정하지 않는다. 독립된 development split을 감사한 후
crop 크기, pair 구성, appearance loss, checkpoint 선택 규칙을 데이터 기반으로 고정한다.

저자에게 메일이나 이슈를 보내지는 않았다. 새 링크를 요청하는 외부 연락은 별도 사용자 지시가 필요하다.
