"""Read-only CPU installation evidence checks; no actual rendering or simulator import."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import unittest

from history_helpers import require_research_history

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/renderer_cpu_install_2026-09-12"
ARRAYS = {"rgb_flat", "rgb_lambertian", "depth_m", "range_m", "normal_world",
          "face_id", "instance_id", "valid"}


class CpuInstallEvidenceTest(unittest.TestCase):
    def test_environment_records_verified_profile_and_noneditable_source(self):
        environment = json.loads((RESULT / "environment.json").read_text())
        self.assertEqual(environment["profile_sha256"], hashlib.sha256(
            (ROOT / "requirements-renderer-cpu.txt").read_bytes()).hexdigest())
        self.assertFalse(environment["user_site_enabled"])
        self.assertIsNone(environment["torch_cuda"])
        self.assertEqual(environment["pip_check"]["exit_code"], 0)
        packages = environment["packages"]
        self.assertEqual(packages["torch"]["version"], "2.4.1+cpu")
        source = packages["urdfpy"]["direct_url"]
        self.assertEqual(source["url"], "https://github.com/mmatl/urdfpy.git")
        self.assertEqual(source["vcs_info"]["commit_id"],
                         "5466842899b33bd549e8f9e2a9a987bd5e37373b")
        self.assertFalse(source.get("dir_info", {}).get("editable", False))
        prefix = PurePosixPath(environment["prefix"])
        for package in packages.values():
            location = PurePosixPath(package["path"])
            self.assertTrue(location == prefix or prefix in location.parents)

    def test_frames_have_correct_file_and_decoded_array_hashes(self):
        import numpy as np
        for name in ("smoke1", "smoke2"):
            folder = RESULT / name
            receipt = json.loads((folder / "receipt.json").read_text())
            self.assertEqual([frame["file"] for frame in receipt["files"]],
                             ["frame_0000.npz", "frame_0001.npz"])
            for frame in receipt["files"]:
                path = folder / frame["file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), frame["file_sha256"])
                self.assertEqual(set(frame["arrays"]), ARRAYS)
                with np.load(path, allow_pickle=False) as arrays:
                    self.assertEqual(set(arrays.files), ARRAYS)
                    for key, expected in frame["arrays"].items():
                        array = arrays[key]
                        self.assertEqual(list(array.shape), expected["shape"])
                        self.assertEqual(str(array.dtype), expected["dtype"])
                        self.assertEqual(hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                                         expected["sha256"])

    def test_two_runs_match_without_creating_an_experiment_verdict(self):

        require_research_history(ROOT)
        runs = [json.loads((RESULT / name / "receipt.json").read_text())
                for name in ("smoke1", "smoke2")]
        for run in runs:
            for key, value in {"device": "cpu", "status": "RENDERED_UNASSESSED",
                               "experiment_verdict": "NOT_EVALUATED", "benchmark": "NOT_RUN",
                               "training": "NOT_SUPPORTED", "seed": 0, "num_scenes": 1,
                               "requested_frames": 2, "scene_kind": "static_generic_box_fixture"}.items():
                self.assertEqual(run[key], value)
            self.assertEqual((run["camera"]["width"], run["camera"]["height"]), (160, 120))
            self.assertFalse(run["source"]["git_dirty_before_output"])
            self.assertEqual(run["source"]["git_commit"],
                             "cdc8113652008b0b856f056601c13ca4e25e6632")
        self.assertEqual(runs[0]["source"], runs[1]["source"])
        self.assertTrue(runs[0]["source"]["source_sha256"])
        for name, expected in runs[0]["source"]["source_sha256"].items():
            payload = subprocess.check_output(["git", "show", "{}:{}".format(
                runs[0]["source"]["git_commit"], name)], cwd=str(ROOT))
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)
        for left, right in zip(runs[0]["files"], runs[1]["files"]):
            self.assertEqual(left["arrays"], right["arrays"])


if __name__ == "__main__":
    unittest.main()
