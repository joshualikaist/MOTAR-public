"""Guard current prose against resurrection of withdrawn claims; no experiment runs."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


class RepositoryClaimsTest(unittest.TestCase):
    def test_current_verification_has_completed_and_blocked_outcomes(self):
        page = read("VERIFICATION.md").split("## 역사 기록:")[0]
        for word in ("P3–P10", "S4", "R-C", "INCONCLUSIVE", "WITHDRAWN",
                     "ATTITUDE_NOT_RELIABLE", "BLOCKED", "FAIL 유지"):
            self.assertIn(word, page)

    def test_completed_runs_are_not_current_authority(self):
        page = read("OPERATIONS.md")
        self.assertIn("완료된 실행 기록 — P10", page)
        self.assertNotIn("## 허가된 실행 — P10", page)
        self.assertIn("소비된 실행 계약", page)

    def test_current_r4_prose_does_not_blame_criterion(self):
        for name in ("README.md", "docs/status/index.html"):
            page = read(name)
            for withdrawn in ("What failed was the criterion:", "C가 실패한 진짜 이유",
                              "<strong>기준 설계 결함</strong>"):
                self.assertNotIn(withdrawn, page)

    def test_c3_summary_is_withdrawn(self):
        page = read("docs/plans/confirmation_phase_plan_2026-09-06.md")
        summary = page.split("## 0. 한 문장")[1].split("## 1.")[0]
        self.assertIn("철회", summary)
        self.assertNotIn("정책이 무엇과 함께 학습됐는지가 정하며", summary)

    def test_e3p_latest_test_is_not_reported_unrun(self):
        page = read("docs/plans/eth_ds5_e3_2026-09-10.md")
        self.assertIn("ATTITUDE_NOT_RELIABLE", page)
        self.assertIn("third reliability test has already run and failed", page)
        self.assertNotIn("where the six\narms are resolved", read(
            "results/eth_ds5_e3p_reliability_2026-09-10/README.md"))

    def test_smoothness_is_not_accuracy(self):
        page = read("results/eth_ds5_e3s_2026-09-10/README.md")
        self.assertIn("trajectory-smoothness residual", page)
        self.assertIn("not a bound on measurement", page)
        self.assertIn("no temporal embargo", page)

    def test_test_access_ledger_is_experiment_specific(self):
        page = read("VERIFICATION.md").split("## 역사 기록:")[0]
        self.assertIn("Det-Fly 020", page)
        self.assertIn("P4/P5", page)
        self.assertIn("never-observed holdout 아님", page)

    def test_memory_measure_has_scope(self):
        page = read("docs/status/archive-2026-09-13.html")
        self.assertIn("183.9 MiB Torch peak allocated", page)
        self.assertIn("전체 프로세스 VRAM은 아닙니다", page)

    def test_prototype_area_is_not_active_detector_area(self):
        for name in ("docs/results_overview_2026-09-12.md", "docs/status/archive-2026-09-13.html"):
            page = read(name)
            self.assertIn("0.587", page)
            self.assertIn("1.000", page)
            self.assertIn("3,840", page)
            self.assertIn("target_appearance_in_sim_2026-09-12/AUDIT.md", page)
        self.assertNotIn("Making the target drone-shaped costs 41%", read("README.md"))
        self.assertNotIn("색만으로 표적을 찾을 수 있었습니다", read("docs/status/index.html"))

    def test_current_v1_resolution_does_not_erase_failure_history(self):
        for name in ("VERIFICATION.md", "docs/v1_shared_airframe_contract.md",
                     "docs/status/archive-2026-09-13.html"):
            page = read(name)
            self.assertIn("TEST_CONTRACT_RECONCILED", page)
            self.assertIn("5cea0e4", page)
            self.assertIn("CONTRACT_MISMATCH", page)
            self.assertIn("repository_followup_2026-09-12.md", page)
        for stale in ("contract mismatch unresolved", "still-recorded one-link test contract",
                      "current 13-link asset conflicts"):
            self.assertNotIn(stale, read("README.md"))

    def test_followup_preserves_the_failed_install_time_summary(self):
        folder = ROOT / "results/renderer_cpu_install_2026-09-12"
        followup = json.loads((folder / "followup.json").read_text())
        payload = (folder / "summary.json").read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(),
                         followup["historical_summary_sha256"])
        historical = json.loads(payload)["full_repository_regression"]
        self.assertEqual(historical["failures"], 1)
        current = followup["full_repository_regression"]
        self.assertEqual((current["failures"], current["errors"], current["skipped"]), (0, 0, 4))
        self.assertEqual(current["tests_run"], current["passed"] + current["skipped"])
        self.assertFalse(followup["source_dirty_at_test_start"])
        self.assertEqual(followup["source_commit"], "ef598328c5a2edb54080e13124a9208d7ed9c9c7")

    def test_new_path_cost_is_not_inferred_from_archived_renderer_timings(self):
        page = read("docs/repository_followup_2026-09-12.md")
        self.assertIn("INTEGRATED_RENDER_COST_UNMEASURED", page)
        self.assertIn("PARTIAL_EVIDENCE", page)
        self.assertIn("기존 원자료를 변경하거나 새 성능 추정치를 만들어 넣지 않았다", page)


if __name__ == "__main__":
    unittest.main()
