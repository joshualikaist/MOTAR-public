"""Public maintenance/graphics contracts; no simulator execution or network."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import motar_doctor as doctor
import audit_public_history as history
import attach_archive_notices as notices
from benchmark_renderer_pipeline import stats


class DoctorTest(unittest.TestCase):
    def data(self):
        p = {name: {"import": "AVAILABLE", "path": "/tmp/example/env/site", "version": version}
             for name, version in (("torch", "2.4.1+cpu"), ("warp-lang", "1.0.0"),
                                    ("trimesh", "4.11.5"), ("numpy", "1.23.0"), ("urdfpy", "0.0.22"))}
        p["torch"]["cuda_build"] = None
        p["urdfpy"]["source"] = {"url": "https://github.com/mmatl/urdfpy.git",
            "vcs_info": {"commit_id": doctor.URDF_COMMIT}}
        return {"python_minor": [3, 8], "platform": "Linux", "machine": "x86_64",
                "isolated_venv": True, "user_site_enabled": False, "prefix": "/tmp/example/env",
                "packages": p, "paths": {"kernel": True}, "isaacgym_discoverable_not_imported": False}

    def test_cpu_pass_without_isaac(self):
        self.assertEqual(doctor.assess(self.data(), "renderer-cpu")["status"], "PASS")

    def test_missing_import_wrong_source_and_user_site_fail(self):
        for field in ("missing", "source", "site", "outside", "cuda"):
            d = self.data()
            if field == "missing": d["packages"]["torch"].pop("import")
            if field == "source": d["packages"]["urdfpy"]["source"] = {}
            if field == "site": d["user_site_enabled"] = True
            if field == "outside": d["packages"]["numpy"]["path"] = "/usr/lib"
            if field == "cuda": d["packages"]["torch"]["cuda_build"] = "12.1"
            self.assertEqual(doctor.assess(d, "renderer-cpu")["status"], "FAIL", field)

    def test_simulator_inventory_never_certifies_execution(self):
        d = self.data()
        self.assertEqual(doctor.assess(d, "simulator")["status"], "UNAVAILABLE")
        d["isaacgym_discoverable_not_imported"] = True
        self.assertEqual(doctor.assess(d, "simulator")["status"], "INVENTORY_ONLY")


class HistoryTest(unittest.TestCase):
    def test_redacted_detection(self):
        candidate = b"gh" + b"p_" + b"a" * 36
        found = history.findings(candidate + b" /home/" + b"example/file")
        self.assertEqual(found, {"github_token": 1, "personal_absolute_path": 1})
        self.assertNotIn(candidate.decode(), json.dumps(found))

    def test_current_and_removed_blobs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            def git(*args):
                return subprocess.check_output(["git", "-C", td, *args], stderr=subprocess.DEVNULL)
            git("init")
            git("config", "user.email", "fixture@example.invalid")
            git("config", "user.name", "Fixture")
            (root / "old.png").write_bytes(b"old-image-fixture")
            git("add", ".")
            git("commit", "-m", "old")
            (root / "old.png").unlink()
            (root / "current.png").write_bytes(b"new-image-fixture")
            git("add", "-A")
            git("commit", "-m", "new")
            rows = {r["example_path"]: r for r in history.audit(root)["binary_archive_image_weight_inventory"]}
            self.assertFalse(rows["old.png"]["in_current_index"])
            self.assertTrue(rows["current.png"]["in_current_index"])


class ArchiveNoticeTest(unittest.TestCase):
    def test_preserves_original_members_and_rejects_repeat(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            archive = root / "review.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("image.jpg", b"unchanged-image-fixture")
            original = hashlib.sha256(archive.read_bytes()).hexdigest()
            notice = root / "NOTICE.txt"
            notice.write_text("fixture attribution")
            result = notices.attach(archive, original, [notice], root / "receipt.json")
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read("image.jpg"), b"unchanged-image-fixture")
                self.assertEqual(z.read("NOTICE.txt"), notice.read_bytes())
            self.assertNotEqual(result["before_sha256"], result["after_sha256"])
            with self.assertRaises(ValueError):
                notices.attach(archive, original, [notice], root / "other.json")
            with self.assertRaises(FileExistsError):
                notices.attach(archive, result["after_sha256"], [notice], root / "receipt.json")

    def test_external_eth_inventory_matches_historical_receipt_and_notices(self):
        folder = ROOT / "results/eth_ds5_intake_2026-09-10"
        r = json.loads((folder / "archive_notice_receipt.json").read_text())
        archive = folder / r["archive"]
        self.assertFalse(archive.exists(), "ETH review ZIP must not be bundled")
        inventory = json.loads((ROOT / "docs/eth_ds5_public_release_inventory_2026-09-14.json").read_text())
        entry = next(a for a in inventory["assets"] if a["path"] == archive.relative_to(ROOT).as_posix())
        self.assertEqual(entry["sha256"], r["after_sha256"])
        expected = {**r["original_members_sha256"], **r["added_members_sha256"]}
        self.assertEqual({m["path"]: m["sha256"] for m in entry["members"]}, expected)
        self.assertEqual(len(entry["members"]), len(expected))
        for name, digest in r["added_members_sha256"].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), digest)


class GenericPipelineTest(unittest.TestCase):
    def test_timing_distribution_not_sum_of_stage_estimates(self):
        r = stats([.001, .002, .003, .004], 2)
        self.assertAlmostEqual(r["mean_ms"], 2.5)
        self.assertAlmostEqual(r["p95_ms"], 4)
        self.assertAlmostEqual(r["images_per_second"], 800)

    def test_bad_timing_rejected(self):
        for values in ([], [0], [float("nan")], [-1]):
            with self.assertRaises(ValueError): stats(values, 1)

    def test_geometry_and_output_budget(self):
        from renderer_validation.public_pipeline import budget, scene_for
        self.assertGreater(budget(160, 90, 1), 0)
        self.assertGreater(len(scene_for("background").triangles), len(scene_for("boxes").triangles))
        with self.assertRaises(ValueError): scene_for("target")
        with self.assertRaises(ValueError): budget(2048, 2048, 128, 16)

    def test_material_and_light_sampling_are_decoupled(self):
        import numpy as np
        from renderer_validation.public_pipeline import separate_appearance
        a = separate_appearance(0, 0, 2, 4)
        light = separate_appearance(0, 2, 2, 4)
        material = separate_appearance(2, 0, 2, 4)
        np.testing.assert_array_equal(a.base_color, light.base_color)
        np.testing.assert_array_equal(a.light_direction, material.light_direction)
        self.assertFalse(np.array_equal(a.light_direction, light.light_direction))
        self.assertFalse(np.array_equal(a.base_color, material.base_color))

    def test_incomplete_export_is_rejected(self):
        from renderer_validation.public_pipeline import verify_export
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "receipt.json").write_text('{"status":"INCOMPLETE","files":[]}')
            with self.assertRaises(ValueError): verify_export(td)


if __name__ == "__main__":
    unittest.main()
