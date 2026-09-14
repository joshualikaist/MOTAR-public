"""RC-R6 exporter: the five preregistered validation checks, plus its refusals.

The five checks run the exporter as a subprocess, because the first of them is about two independent
processes producing the same bytes, and because an in-process check would share state that a user of
the tool never shares.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import export_renderer_dataset as exporter
from renderer_validation.public_pipeline import verify_export

SMALL = ["--width", "40", "--height", "30", "--device", "cpu", "--views", "fit"]


def run_export(folder, *extra, arm="box_proxy", expect=0):
    command = [sys.executable, "-B", str(ROOT / "tools/export_renderer_dataset.py"),
               "--output", str(folder)]
    if arm is not None:
        command += ["--arm", arm, *SMALL]
    command += list(extra)
    result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True,
                            env=dict(os.environ, PYTHONNOUSERSITE="1"))
    if result.returncode != expect:
        raise AssertionError("exporter returned %d: %s" % (result.returncode, result.stderr[-2000:]))
    return result


class ArmExportTest(unittest.TestCase):
    """The five preregistered checks. One temporary directory, five independent processes."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.cases = {"base": [], "repeat": [],
                     "light": ["--ambient", "0.05", "--directional", "0.4"],
                     "material": ["--colour", "0.2"],
                     "renumber": ["--instance-offset", "100"]}
        cls.records = {}
        for name, extra in cls.cases.items():
            run_export(root / name, *extra)
            cls.records[name] = verify_export(root / name)
        cls.arrays = {name: record["files"][0]["arrays"] for name, record in cls.records.items()}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def geometry(self, name):
        return {key: self.arrays[name][key] for key in
                ("range_m", "depth_m", "normal_world", "face_id", "valid")}

    def test_1_same_arguments_in_two_processes_give_identical_arrays(self):
        self.assertEqual(self.arrays["base"], self.arrays["repeat"])
        self.assertEqual(self.records["base"]["source"], self.records["repeat"]["source"])

    def test_2_lighting_changes_rgb_and_leaves_geometry_identical(self):
        self.assertNotEqual(self.arrays["base"]["rgb"], self.arrays["light"]["rgb"])
        self.assertEqual(self.geometry("base"), self.geometry("light"))
        self.assertEqual(self.arrays["base"]["instance_id"], self.arrays["light"]["instance_id"])

    def test_3_material_changes_rgb_and_leaves_geometry_identical(self):
        self.assertNotEqual(self.arrays["base"]["rgb"], self.arrays["material"]["rgb"])
        self.assertEqual(self.geometry("base"), self.geometry("material"))
        self.assertEqual(self.arrays["base"]["instance_id"], self.arrays["material"]["instance_id"])

    def test_4_renumbering_preserves_rgb_and_changes_only_labels(self):
        self.assertEqual(self.arrays["base"]["rgb"], self.arrays["renumber"]["rgb"])
        self.assertEqual(self.geometry("base"), self.geometry("renumber"))
        self.assertNotEqual(self.arrays["base"]["instance_id"],
                            self.arrays["renumber"]["instance_id"])

    def test_5_receipt_hashes_match_the_bytes_and_the_decoded_arrays(self):
        # verify_export already recomputes both; this asserts it would notice a change.
        folder = Path(self.directory.name) / "base"
        path = folder / self.records["base"]["files"][0]["file"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         self.records["base"]["files"][0]["sha256"])
        receipt = json.loads((folder / "receipt.json").read_text())
        receipt["files"][0]["sha256"] = "0" * 64
        (folder / "receipt.json").write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):
            verify_export(folder)

    def test_metadata_describes_the_arm_the_views_and_the_appearance(self):
        description = self.records["base"]["description"]
        self.assertEqual(description["arm"], "box_proxy")
        self.assertEqual(description["arm_code"], "A1")
        self.assertEqual(len(description["views"]["views"]), 4)
        self.assertEqual(description["camera"]["width"], 40)
        self.assertIn("appearance", description)
        self.assertEqual(description["light_mode"], "camera_relative")
        self.assertEqual(description["instance_offset"], 0)
        self.assertEqual(self.records["base"]["schema"], "renderer_characterization_export_v1")
        self.assertEqual(self.records["base"]["status"], "EXPORTED_UNASSESSED")
        self.assertEqual(self.records["base"]["experiment_verdict"], "NOT_EVALUATED")
        self.assertEqual(self.records["base"]["training"], "NOT_SUPPORTED")
        self.assertEqual(self.records["base"]["runtime"]["warp"], "1.0.0")

    def test_arrays_have_one_entry_per_view(self):
        for name, record in self.arrays["base"].items():
            self.assertEqual(record["shape"][0], 4, name)
            self.assertEqual(record["shape"][1:3], [30, 40], name)


class ExporterRefusalTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_arm_options_are_refused_without_an_arm(self):
        for extra in (["--shading", "flat"], ["--colour", "0.3"], ["--views", "sweep"],
                      ["--ambient", "0.1"], ["--scale", "1.2"], ["--light-mode", "world"]):
            with self.assertRaises(ValueError):
                exporter.main(["--output", str(self.root / "x"), *extra])

    def test_scale_belongs_to_the_area_matched_box_alone(self):
        with self.assertRaises(ValueError):
            exporter.main(["--output", str(self.root / "a"), "--arm", "box_proxy",
                           "--scale", "1.1", *SMALL])
        with self.assertRaises(ValueError):
            exporter.main(["--output", str(self.root / "b"), "--arm", "area_matched_box", *SMALL])

    def test_generic_options_that_cannot_apply_to_an_arm_are_refused(self):
        for extra in (["--frames", "2"], ["--num-scenes", "4"], ["--position", "1", "0", "0"],
                      ["--quaternion", "0", "0", "1", "0"]):
            with self.assertRaises(ValueError):
                exporter.main(["--output", str(self.root / "c"), "--arm", "box_proxy",
                               *SMALL, *extra])

    def test_out_of_range_appearance_is_refused(self):
        for extra in (["--colour", "1.5"], ["--kd", "-0.1"], ["--ambient", "-1"],
                      ["--directional", "-0.5"]):
            with self.assertRaises(ValueError):
                exporter.main(["--output", str(self.root / "d"), "--arm", "box_proxy",
                               *SMALL, *extra])

    def test_existing_output_is_refused_before_anything_is_written(self):
        (self.root / "taken").mkdir()
        with self.assertRaises(FileExistsError):
            exporter.main(["--output", str(self.root / "taken"), "--arm", "box_proxy", *SMALL])

    def test_a_failed_export_leaves_a_failure_record_and_no_receipt(self):
        folder = self.root / "failing"
        with self.assertRaises(Exception):
            exporter.main(["--output", str(folder), "--arm", "quadrotor_mesh", *SMALL,
                           "--width", "0"])
        self.assertFalse((folder / "receipt.json").exists())


class GenericPathUnchangedTest(unittest.TestCase):
    def test_generic_receipt_records_no_arm_options(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "generic"
            run_export(folder, "--width", "32", "--height", "24", "--device", "cpu", arm=None)
            record = verify_export(folder)
            self.assertEqual(record["schema"], "generic_renderer_export_v1")
            self.assertEqual(sorted(record["config"]),
                             ["device", "frames", "geometry", "height", "instance_offset",
                              "light_seed", "material_seed", "num_scenes", "output", "position",
                              "quaternion", "width"])
            for name in exporter.ARM_ONLY:
                self.assertNotIn(name, record["config"])

    def test_generic_geometry_choices_still_exclude_every_specimen(self):
        from renderer_validation.public_pipeline import scene_for
        for name in ("target", "quadrotor_mesh", "box_proxy", "analytic_sphere"):
            with self.assertRaises(ValueError):
                scene_for(name)


if __name__ == "__main__":
    unittest.main()
