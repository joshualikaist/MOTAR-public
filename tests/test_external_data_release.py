"""External-data release gates use metadata and synthetic files, never real dataset images."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import audit_external_data_release as audit
import build_result_manifest as manifest
import check_eth_ds5_local_data as local
from check_public_docs import local_link_errors


class ExternalDataReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads(audit.INVENTORY.read_text())
        cls.contract = json.loads((ROOT / "docs/external_data_manifest.json").read_text())

    def test_exact_removed_set_and_no_tracked_eth_images(self):
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        self.assertEqual(len(self.inventory["assets"]), 14)
        self.assertEqual(sum(a["type"] == "JPG" for a in self.inventory["assets"]), 13)
        for a in self.inventory["assets"]:
            self.assertNotIn(a["path"], tracked)
            self.assertFalse((ROOT / a["path"]).exists())
        self.assertFalse([p for p in tracked if "eth_ds5" in p and Path(p).suffix.lower()
                          in {".jpg", ".jpeg", ".zip", ".png", ".mp4"}])

    def test_historical_evidence_bytes_are_preserved(self):
        for path, row in self.inventory["preserved_evidence"].items():
            if row["editorial_change_allowed"]:
                self.assertEqual(path, "results/eth_ds5_intake_2026-09-10/HUMAN_REVIEW_HANDOFF.md")
                continue
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), row["sha256"], path)

    def test_all_eth_results_and_receipts_are_covered(self):
        pinned = self.inventory["preserved_evidence"]
        for directory in (ROOT / "results").glob("eth_ds5_*"):
            for p in directory.rglob("*.json"):
                relative = p.relative_to(ROOT).as_posix()
                tracked = subprocess.run(["git", "ls-files", "--error-unmatch", relative],
                                         cwd=ROOT, capture_output=True).returncode == 0
                if tracked:
                    self.assertIn(relative, pinned)

    def test_release_contract_exact_hashes_no_broad_exception(self):
        self.assertFalse(self.contract["redistributed"])
        self.assertEqual(self.contract["current_release_availability"], "EXTERNAL_DATA_REQUIRED")
        self.assertEqual({a["path"]: a["sha256"] for a in self.contract["excluded_assets"]},
                         {a["path"]: a["sha256"] for a in self.inventory["assets"]})

    def test_external_reference_requires_path_and_hash(self):
        a = self.inventory["assets"][0]
        self.assertEqual(manifest.reference_availability(a["path"], a["sha256"]), "EXTERNAL_DATA_NOT_REDISTRIBUTED")
        self.assertEqual(manifest.reference_availability(a["path"], "0" * 64), "MISSING_REQUIRED_REPOSITORY_ARTIFACT")
        self.assertEqual(manifest.reference_availability("results/eth_ds5_other/missing.jpg", a["sha256"]),
                         "MISSING_REQUIRED_REPOSITORY_ARTIFACT")

    def test_present_bad_bytes_never_receive_external_exception(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "x.jpg").write_bytes(b"synthetic-corrupted")
            with patch.object(manifest, "ROOT", root):
                self.assertEqual(manifest.reference_availability("x.jpg", "0" * 64), "HASH_MISMATCH")

    def test_source_manifest_retains_missing_and_drift_errors(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            a = self.inventory["assets"][0]
            (folder / "source_manifest.json").write_text(json.dumps({"source_sha256": {
                a["path"]: a["sha256"], "tools/missing_required.py": "0" * 64,
                "README.md": "0" * 64}}))
            state = manifest.source_manifest_state(folder)
            self.assertEqual(state["missing"], ["tools/missing_required.py"])
            self.assertEqual(state["drifted"], ["README.md"])
            self.assertEqual(state["matching"], 0)
            self.assertEqual(state["external_data_not_redistributed"][0]["reference_kind"], "HISTORICAL_INPUT_REFERENCE")

    def test_external_docs_and_public_links(self):
        for name in ("README.md", "docs/REPRODUCIBILITY.md", "THIRD_PARTY_LICENSES.md"):
            self.assertIn("external_data/ETH_DS5.md", (ROOT / name).read_text())
        self.assertIn("NOT_REDISTRIBUTED in current tree", (ROOT / "THIRD_PARTY_LICENSES.md").read_text())
        for name in ("docs/external_data/ETH_DS5.md", "docs/public_release_status_2026-09-14.md",
                     "results/eth_ds5_intake_2026-09-10/HUMAN_REVIEW_HANDOFF.md"):
            path = ROOT / name
            self.assertEqual(local_link_errors(path.read_text(), path), [], name)

    def test_private_output_and_removed_paths_are_ignored(self):
        for name in [a["path"] for a in self.inventory["assets"]] + ["artifacts/private/eth_ds5/new/frame.png"]:
            self.assertEqual(subprocess.run(["git", "check-ignore", "--quiet", name], cwd=ROOT).returncode, 0, name)
        self.assertNotEqual(subprocess.run(["git", "check-ignore", "--quiet", "results/new_summary.json"], cwd=ROOT).returncode, 0)

    def test_missing_dataset_fails_with_acquisition_guidance(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "ETH ds5 is not distributed with MOTAR"):
                local.check(td, Path(td) / "absent.mp4")
            result = subprocess.run([sys.executable, str(ROOT / "tools/check_eth_ds5_local_data.py"),
                                     "--dataset", td, "--video", td + "/absent.mp4"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--dataset and --video", result.stderr)
            self.assertFalse(list(Path(td).iterdir()))

    def test_synthetic_local_presence_is_not_scientific_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in local.REQUIRED + ("cam0.mp4",):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic-not-a-real-dataset")
            self.assertEqual(local.check(root, root / "cam0.mp4")["scientific_validation"], "NOT_RUN")
            (root / "cam0.mp4").write_bytes(b"")
            with self.assertRaises(ValueError):
                local.check(root, root / "cam0.mp4")

    def test_no_current_public_embed_or_code_dependency_on_removed_paths(self):
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        for name in tracked:
            if (name.startswith(("aerial_gym/", "tools/")) and name.endswith((".py", ".sh"))) or name.endswith(".html"):
                text = (ROOT / name).read_text(errors="replace")
                for a in self.inventory["assets"]:
                    self.assertNotIn(a["path"], text, name)

    def test_committed_result_manifest_has_external_availability(self):
        rows = json.loads((ROOT / "results/MANIFEST.json").read_text())["results"]
        by_id = {r["result_id"]: r for r in rows}
        for key in self.contract["required_for"]:
            record = by_id[key]["external_data"]
            self.assertEqual(record["current_release_availability"], "EXTERNAL_DATA_REQUIRED")
            self.assertEqual(record["contract_sha256"], hashlib.sha256((ROOT / record["contract_path"]).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
