"""The new documentation contract checks decoded old-exporter output, without editing that source."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from renderer_validation.gbuffer import GBuffer
from renderer_validation.scene import box_fixture
from renderer_validation.characterization_pipeline import appearance_for
from renderer_validation.shading import shade
from renderer_validation.public_pipeline import verify_export


class ExportBufferContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "export"
        subprocess.run([sys.executable, "-B", str(ROOT / "tools/export_renderer_dataset.py"),
                        "--output", str(cls.output), "--arm", "box_proxy", "--views", "fit",
                        "--width", "80", "--height", "60", "--device", "cpu"],
                       cwd=ROOT, env=dict(os.environ, PYTHONNOUSERSITE="1"),
                       capture_output=True, text=True, check=True)
        cls.receipt = verify_export(cls.output)
        with np.load(cls.output / "frame_0000.npz", allow_pickle=False) as archive:
            cls.arrays = {k: archive[k] for k in archive.files}
        cls.contract = json.loads((ROOT / "docs/renderer_buffer_contract_v1.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_seven_buffers_obey_machine_readable_dtypes_and_layout(self):
        self.assertEqual(set(self.arrays), set(self.contract["buffers"]))
        for key, spec in self.contract["buffers"].items():
            self.assertEqual(str(self.arrays[key].dtype), spec["dtype"])
            self.assertEqual(self.arrays[key].shape, (4, 60, 80) + ((3,) if spec["layout"] == "NHWC" else ()))
            self.assertTrue(np.isfinite(self.arrays[key]).all())

    def test_miss_sentinels_and_background_are_explicit(self):
        miss = ~self.arrays["valid"]
        self.assertTrue(miss.any())
        self.assertTrue((~miss).any())
        for key, spec in self.contract["buffers"].items():
            if "miss" in spec:
                self.assertTrue((self.arrays[key][miss] == spec["miss"]).all(), key)
        np.testing.assert_allclose(self.arrays["rgb"][miss], 0, rtol=0, atol=0)

    def test_optical_depth_is_not_ray_range(self):
        arrays = self.arrays
        v, u = np.indices((60, 80))
        f = 80/(2*np.tan(np.deg2rad(60)/2))
        factor = (1/np.sqrt(1+((u-40)/f)**2+((v-30)/f)**2)).astype(np.float32)
        np.testing.assert_allclose(arrays["depth_m"], arrays["range_m"]*factor, rtol=1e-6, atol=1e-7)
        self.assertTrue((arrays["range_m"]-arrays["depth_m"] > .001).any())

    def test_valid_normals_are_world_space_unit_vectors(self):
        arrays = self.arrays
        normals = arrays["normal_world"][arrays["valid"]]
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, rtol=0, atol=1e-6)
        # Axis-aligned world box: camera rotation must not rotate the exported world normals.
        np.testing.assert_allclose(np.abs(normals).max(axis=1), 1, rtol=0, atol=1e-6)

    def test_labels_match_hits_without_certifying_an_experiment(self):
        arrays = self.arrays
        np.testing.assert_array_equal(arrays["face_id"] >= 0, arrays["valid"])
        np.testing.assert_array_equal(arrays["instance_id"] >= 0, arrays["valid"])
        for key in ("status", "experiment_verdict", "training"):
            self.assertEqual(self.receipt[key], self.contract[key])
        self.assertEqual(self.contract["causality_vs_d8b"], "NOT_TESTED")

    def test_receipt_pins_decoded_array_bytes_and_camera_metadata(self):
        for key, array in self.arrays.items():
            self.assertEqual(hashlib.sha256(array.tobytes()).hexdigest(),
                             self.receipt["files"][0]["arrays"][key]["sha256"])
        self.assertEqual(self.receipt["description"]["camera"]["horizontal_fov_deg"], 60.)
        self.assertEqual(len(self.receipt["description"]["views"]["quaternions_xyzw"]), 4)


class ShaderBoundaryContract(unittest.TestCase):
    def fixture(self):
        scene = box_fixture()
        scalar = torch.ones((1, 1, 1), dtype=torch.float32)
        normal = torch.tensor([[[[0., 0., -1.]]]])
        face = torch.zeros((1, 1, 1), dtype=torch.int32)
        buffer = GBuffer(scalar, scalar, normal, face, face.clone(), scalar.bool())
        appearance = appearance_for(1, scene.material_count, colour=[.1, .4, .7])
        return scene, buffer, appearance

    def test_instance_identity_never_changes_shading(self):
        scene, buffer, appearance = self.fixture()
        changed = replace(buffer, instance_id=buffer.instance_id + 999)
        self.assertTrue(torch.equal(shade(buffer, scene, appearance), shade(changed, scene, appearance)))

    def test_face_id_really_does_address_material_not_an_identity_cue(self):
        scene, buffer, _ = self.fixture()
        appearance = appearance_for(1, scene.material_count,
                                    colour=[[.1, .1, .1], [.8, .8, .8], [.3, .3, .3], [.5, .5, .5]])
        changed = replace(buffer, face_id=buffer.face_id + 6)
        self.assertFalse(torch.equal(shade(buffer, scene, appearance, "flat"),
                                     shade(changed, scene, appearance, "flat")))
        self.assertTrue(torch.equal(buffer.normal_world, changed.normal_world))


class FigureManifestTest(unittest.TestCase):
    """The figure manifest is only useful if it cannot quietly disagree with the files.

    A figure page that keeps claiming a verdict its data no longer supports is worse than no page,
    so these checks re-hash the assets, the generator and the result summaries the figures were
    drawn from, and fail on any drift in either direction.
    """

    ASSETS = ROOT / "docs/assets/paper/renderer-contract-v1-2026-09-14"
    PAGE = ROOT / "docs/status/renderer-contract-v1.html"
    STEMS = ("fig-rc-1-resolution-convergence", "fig-rc-2-area-control-validation",
             "fig-rc-3-normal-entropy-vs-shading", "fig-rc-4-timing-breakdown",
             "fig-rc-5-contract-diagram")

    def manifest(self):
        return json.loads((self.ASSETS / "manifest.json").read_text())

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def test_every_figure_ships_as_svg_png_and_vector_pdf(self):
        for stem in self.STEMS:
            for suffix in ("svg", "png", "pdf"):
                path = self.ASSETS / ("%s.%s" % (stem, suffix))
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 2048, path)

    def test_manifest_hashes_match_the_assets_on_disk(self):
        manifest = self.manifest()
        self.assertEqual(sorted(manifest["assets"]),
                         sorted("%s.%s" % (stem, suffix) for stem in self.STEMS
                                for suffix in ("svg", "png", "pdf")))
        for name, record in manifest["assets"].items():
            path = self.ASSETS / name
            self.assertEqual(record["sha256"], self.digest(path), name)
            self.assertEqual(record["bytes"], path.stat().st_size, name)

    def test_manifest_pins_the_result_summaries_and_the_generator(self):
        manifest = self.manifest()
        self.assertEqual(sorted(manifest["generated_from"]), ["r1b", "r2b", "r3b", "r5b"])
        for stage, record in manifest["generated_from"].items():
            path = ROOT / record["path"]
            self.assertTrue(path.is_file(), stage)
            self.assertEqual(record["sha256"], self.digest(path), stage)
        generator = manifest["generator"]
        self.assertEqual(generator["sha256"], self.digest(ROOT / generator["path"]))
        self.assertEqual(manifest["causality_vs_d8b"], "NOT_TESTED")

    def test_the_page_links_resolve_and_state_the_untested_boundary(self):
        text = self.PAGE.read_text()
        for target in set(re.findall(r'(?:href|src)="([^"#]+)"', text)):
            if target.startswith(("http://", "https://")):
                continue
            resolved = (self.PAGE.parent / target.split("?")[0]).resolve()
            self.assertTrue(resolved.is_file(), target)
            self.assertIn(ROOT.resolve(), resolved.parents)
        self.assertIn("NOT_TESTED", text)
        # The page must keep naming the preserved negative verdicts rather than only the passes.
        for verdict in ("GEOMETRY_DEFECT", "AREA_MATCH_FAILED", "SHADING_GATE_FAILED"):
            self.assertIn(verdict, text)

    def test_the_page_quotes_the_verdicts_its_results_actually_carry(self):
        text = self.PAGE.read_text()
        for stage in ("r1b", "r2b", "r3b", "r5b"):
            summary = json.loads((ROOT / ("results/renderer_characterization_%s_2026-09-14"
                                          "/summary.json" % stage)).read_text())
            self.assertIn(summary["verdict"], text, stage)


if __name__ == "__main__":
    unittest.main()
