"""CPU-only contract tests for the closed ref5in P2 orchestration."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

from history_helpers import require_local_evidence
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "attest_navrl_ref5in_p2", ROOT / "tools/attest_navrl_ref5in_p2.py"
)
P2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(P2)


class TestP2Environment(unittest.TestCase):
    def test_hostile_environment_is_replaced_by_canonical_contract(self):
        hostile = {
            "NAVRL_SEED": "999", "NAVRL_V2_FORCE": "1",
            "NAVRL_DETECTOR_CHECKPOINT": "/tmp/hostile.pth",
            "NAVRL_V2_FIXED_TARGET_SPEED": "0.3", "NAVRL_SPEED_GOVERNOR": "riskcap",
            "NAVRL_V2_DENSITIES": "300", "NAVRL_V2_ACTION_MODE": "stochastic",
            "NAVRL_EVAL_REFLECTION_MODE": "conjugate", "GPU4GB": "1", "NUM_ENVS": "2",
            "PYTHONPATH": "/tmp/inject", "PYTHONHOME": "/tmp/inject", "SAFE_UNRELATED": "kept",
        }
        with mock.patch.dict(os.environ, hostile, clear=True):
            env = P2.canonical_env(Path("/tmp/result"))
        self.assertEqual(env["SAFE_UNRELATED"], "kept")
        self.assertEqual(env["NAVRL_SEED"], "313")
        self.assertEqual(env["NAVRL_V2_DENSITIES"], "70")
        self.assertEqual(env["NAVRL_V2_ACTION_MODE"], "deterministic")
        self.assertEqual(env["NAVRL_EVAL_REFLECTION_MODE"], "original")
        self.assertEqual(env["NAVRL_SPEED_GOVERNOR"], "off")
        self.assertEqual(env["GPU4GB"], "0")
        for forbidden in (
            "NAVRL_V2_FORCE", "NAVRL_DETECTOR_CHECKPOINT", "NAVRL_V2_FIXED_TARGET_SPEED",
            "PYTHONPATH", "PYTHONHOME", "NUM_ENVS",
        ):
            self.assertNotIn(forbidden, env)

    def test_preflight_flag_is_only_added_when_requested(self):
        self.assertNotIn("NAVRL_PREFLIGHT_ONLY", P2.canonical_env(Path("/tmp/a")))
        self.assertEqual(P2.canonical_env(Path("/tmp/a"), True)["NAVRL_PREFLIGHT_ONLY"], "1")


class TestP2Decision(unittest.TestCase):
    @staticmethod
    def evidence(captured, crash, timeout):
        actual = captured + crash + timeout
        result = {
            "condition": {"seed": 313},
            "outcome": {
                "captured": captured, "crash": crash, "timeout": timeout,
                "capture_rate": captured / actual, "crash_rate": crash / actual,
                "timeout_rate": timeout / actual,
            },
        }
        return {
            "counts": [captured, crash, timeout], "actual": actual, "result": result,
            "manifest_sha256": "a" * 64, "runtime_git_commit": "b" * 40,
            "result_sha256": "c" * 64, "receipt_sha256": "d" * 64,
            "log_sha256": "e" * 64,
        }

    def test_integer_threshold_boundary_passes(self):
        payload = P2.make_attestation(self.evidence(650, 300, 50), self.evidence(1, 0, 0))
        self.assertEqual(payload["verdict"], "PASS")
        self.assertEqual(payload["unlocks"], "manual_p3_seed211_only")

    def test_each_one_count_threshold_failure_is_closed(self):
        cases = ((649, 301, 50), (650, 331, 19), (650, 299, 51))
        for captured, crash, timeout in cases:
            with self.subTest(counts=(captured, crash, timeout)):
                payload = P2.make_attestation(
                    self.evidence(captured, crash, timeout), self.evidence(1, 0, 0)
                )
                self.assertEqual(payload["verdict"], "FAIL")
                self.assertEqual(payload["unlocks"], "none")

    def test_anchor_outcome_never_changes_primary_verdict(self):
        decision = self.evidence(650, 300, 50)
        good = P2.make_attestation(decision, self.evidence(1, 0, 0))["verdict"]
        bad = P2.make_attestation(decision, self.evidence(0, 1, 0))["verdict"]
        self.assertEqual(good, bad)


class TestP2Provenance(unittest.TestCase):
    def test_frozen_p1c_snapshot_retains_exact_runtime_digest(self):
        require_local_evidence(ROOT / "aerial_gym/rl_training/rl_games/train_source_receipts")
        training, _ = P2.manifest_map(P2.P1_MANIFEST, 1, require_original=False)
        self.assertEqual(P2.map_digest(training), P2.P1_RUNTIME_MAP_SHA)

    def test_historical_verify_does_not_require_current_runtime(self):

        require_local_evidence(ROOT / "aerial_gym/rl_training/rl_games/train_source_receipts")
        with mock.patch.object(
            P2, "current_runtime_map", side_effect=AssertionError("must not be called")
        ):
            payload = P2.verify_attestation()
        self.assertEqual(payload["verdict"], "FAIL")

    def test_map_digest_changes_on_added_path(self):
        mapping = {"aerial_gym/a.py": ("0" * 64, 1)}
        first = P2.map_digest(mapping)
        mapping["aerial_gym/new.py"] = ("1" * 64, 2)
        self.assertNotEqual(first, P2.map_digest(mapping))

    def test_duplicate_manifest_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "aerial_gym/x.py"
            source.parent.mkdir(parents=True)
            source.write_text("x\n", encoding="utf-8")
            snapshot = root / "snap/aerial_gym/x.py"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("x\n", encoding="utf-8")
            environment = root / "python_environment.txt"
            environment.write_text("env\n", encoding="utf-8")
            entry = {
                "path": "aerial_gym/x.py", "sha256": P2.sha256_file(source),
                "size_bytes": source.stat().st_size, "snapshot": "snap/aerial_gym/x.py",
            }
            manifest = {
                "schema_version": 2, "repository_root": str(root), "runtime_file_count": 2,
                "runtime_files": [entry, dict(entry)], "python_environment": environment.name,
                "python_environment_sha256": P2.sha256_file(environment),
            }
            path = root / "source_manifest.json"
            path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(P2.ContractError, "duplicate"):
                P2.manifest_map(path, 2)


if __name__ == "__main__":
    unittest.main()
