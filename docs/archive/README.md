# docs/archive — 역사 기록

2026-08-20 문서 통합으로 검증 단계에 불필요한 planning/review/PPT/handoff/prereg 문서를
여기로 옮겼다. **현재 실행 authority는 저장소 루트 [`VERIFICATION.md`](../../VERIFICATION.md)**.

| 파일 | 내용 |
|---|---|
| `RESEARCH_PLAN_v2_history.md` | 구 RESEARCH_PLAN §8.1–8.22 (v2 205-bar, TTC, riskcap) |
| `ref5in_audit_and_next_steps_2026-08-13.md` | ref5in 감사 요약 (VERIFICATION.md로 흡수) |
| `prereg_2026-08-13_detector_coupling.md` | detector coupling probe 사전등록 |
| `prereg_2026-08-14_detector_coupling_binbias.md` | bin-bias coupling 사전등록 |
| `codex_review_2026-08-*.md` | 독립 검수 기록 |
| `review_brief_2026-08-*.md` | 검증 브리프 |
| `GENSPARK_PPT_BRIEF_*`, `CLAUDE_PPT_*` | 발표 자료 |
| `NEXT_WEEK_HANDOFF_*`, `development_directions_*` | 구 실행 handoff |
| `sim_vs_hardware_gap_*`, `reference_platform_proposal_*` | 하드웨어 격차·플랫폼 제안 |

삭제하지 않았다. 필요하면 grep으로 찾되, 판정·다음 단계는 archive가 아니라 VERIFICATION/WORKLOG를 본다.

## ⚠️ 아카이브인데 소스가 여전히 참조하는 파일

아래는 `docs/archive/`에 있지만 **`.py`/`.sh`가 경로를 인용한다.** 이동·삭제 금지.

| 파일 | 인용처 |
|---|---|
| `prereg_2026-08-13_detector_coupling.md` | 소스 5곳이 **미archive 경로**를 인용 → `docs/prereg_2026-08-13_detector_coupling.md` 스텁으로 복구(2026-09-05) |
| `prereg_2026-08-14_detector_coupling_binbias.md` | `eval_navrl_v2_detector_coupling_binbias.sh:3` |
| `development_directions_2026-08.md` | `env_object_config.py:843` |
| `sim_vs_hardware_gap_2026-08.md` | `tools/generate_platform_spec.py:37` |

## 색인에서 빠져 있던 파일

아래 둘은 위 표에 없었다(2026-09-05 추가).

| 파일 | 내용 |
|---|---|
| `midterm_summary_2026-08.md` | 중간 요약. **riskcap 관련 오래된 수치를 포함하므로 인용 금지** |
| `presentation_followup_2026-08-14.md` | 발표 후속 메모 |
