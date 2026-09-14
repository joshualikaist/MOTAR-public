"""Offline contracts for ETH intake: no network, GPU or dataset required."""
import hashlib
import importlib.util
import math
import io
from pathlib import Path
import tempfile
import unittest
import sys
import json
import contextlib
from unittest import mock

SPEC = importlib.util.spec_from_file_location("eth_ds5", Path(__file__).resolve().parents[1] / "tools/prepare_eth_ds5.py")
ETH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ETH)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_eth_ds5_frames import validate_video
import prepare_eth_ds5_frames as FRAMES
from extract_eth_ds5 import video_member
import verify_eth_ds5_video as VERIFY
import prepare_eth_ds5_review as REVIEW
import check_eth_ds5_reprojection as REPROJ
import select_eth_ds5_alignment_frames as SELECT
import fetch_eth_ds5_calibration_images as FETCH
import calibrate_eth_ds5_cam0 as CALIB
import track_eth_ds5_drone as TRACK
import audit_eth_ds5_camera_model as AUDIT
import measure_eth_ds5_size_range as E3S
import check_eth_ds5_attitude_gate as GATE
import check_eth_ds5_attitude_reliability as RELY


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


class IntakeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = b"abcdef"
        self.entry = {"path": "file.txt", "size": 6,
                      "sha": hashlib.sha1(b"blob 6\0abcdef").hexdigest()}

    def test_download_and_verified_skip(self):
        sha = ETH.download(self.root, self.entry, lambda *a, **k: Response(self.data))
        self.assertEqual(sha, hashlib.sha256(self.data).hexdigest())
        self.assertEqual(ETH.download(self.root, self.entry, None), sha)

    def test_corrupt_existing_is_preserved(self):
        (self.root / "file.txt").write_bytes(b"xxxxxx")
        with self.assertRaises(ValueError): ETH.download(self.root, self.entry, None)
        self.assertEqual((self.root / "file.txt").read_bytes(), b"xxxxxx")

    def test_valid_resume(self):
        (self.root / "file.txt.partial").write_bytes(b"abc")
        ETH.download(self.root, self.entry, lambda *a, **k: Response(b"def", 206, {"Content-Range": "bytes 3-5/6"}))
        self.assertEqual((self.root / "file.txt").read_bytes(), self.data)

    def test_ignored_range_preserves_partial(self):
        (self.root / "file.txt.partial").write_bytes(b"abc")
        with self.assertRaises(ValueError):
            ETH.download(self.root, self.entry, lambda *a, **k: Response(self.data))
        self.assertEqual((self.root / "file.txt.partial").read_bytes(), b"abc")

    def test_wrong_range(self):
        with self.assertRaises(ValueError):
            ETH.download(self.root, self.entry, lambda *a, **k: Response(self.data, 206, {"Content-Range": "bytes 1-6/7"}))

    def test_wrong_blob_not_promoted(self):
        with self.assertRaises(ValueError):
            ETH.download(self.root, self.entry, lambda *a, **k: Response(b"xxxxxx"))
        self.assertFalse((self.root / "file.txt").exists())

    def test_incomplete_resumable(self):
        with self.assertRaises(ValueError):
            ETH.download(self.root, self.entry, lambda *a, **k: Response(b"abc"))
        self.assertEqual((self.root / "file.txt.partial").read_bytes(), b"abc")

    def test_traversal(self):
        for p in ("../x", "/x", "x/../../a", "x\\a"):
            with self.assertRaises(ValueError): ETH.safe_path(self.root, p)

    def test_symlink_escape(self):
        (self.root / "escape").symlink_to(self.root.parent)
        with self.assertRaises(ValueError): ETH.safe_path(self.root, "escape/x")

    def test_transport_retry(self):
        with mock.patch.object(ETH, "download", side_effect=[TimeoutError("interrupted"), "sha"]), mock.patch.object(ETH.time, "sleep"):
            self.assertEqual(ETH.fetch_verified(self.root, self.entry), "sha")

    def test_hash_failure_not_retried(self):
        with mock.patch.object(ETH, "download", side_effect=ValueError("Git blob mismatch")) as fetch:
            with self.assertRaises(ValueError): ETH.fetch_verified(self.root, self.entry)
            self.assertEqual(fetch.call_count, 1)


class MetadataTest(unittest.TestCase):
    def setUp(self):
        self.pose = [[t, 3., 4., 0., 0., 0., 0., .01, .01, .01, 0.] for t in (1., 2., 3.)]
        self.frames = [[float(i+1), t] for i, t in enumerate((0., 1., 1.5, 2., 3., 4.))]

    def test_overlap_not_full_video(self):
        r = ETH.audit(self.pose, self.frames, [0., 0., 0.])
        self.assertEqual(r["temporal_overlap_frames"], 4)
        self.assertEqual(r["raw_slant_range_m_not_quality_filtered"], [5., 5.])
        self.assertFalse(r["range_model_fit"])
        self.assertTrue(r["blockers"])

    def test_duplicate_timestamp_rejected(self):
        self.pose[1][0] = 1.
        with self.assertRaises(ValueError): ETH.audit(self.pose, self.frames, [0., 0., 0.])

    def test_invalid_frame_id(self):
        self.frames[0][0] = .5
        with self.assertRaises(ValueError): ETH.audit(self.pose, self.frames, [0., 0., 0.])

    def test_queue_never_fabricates_gt(self):
        queue = ETH.annotation_queue(self.pose, self.frames, 1)
        self.assertEqual(len(queue), 4)
        for r in queue:
            self.assertIsNone(r["box_xyxy"])
            self.assertFalse(r["measurement_eligible"])
            self.assertFalse(r["identity_verified"])
            self.assertEqual(r["opencv_index"], r["frame_id"]-1)

    def test_invalid_stride(self):
        with self.assertRaises(ValueError): ETH.annotation_queue(self.pose, self.frames, 0)

    def test_nonfinite_table(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "table.txt"
            p.write_text("header\n1 nan\n")
            with self.assertRaises(ValueError): ETH.numeric_table(p, 2)


class VideoTest(unittest.TestCase):
    def test_matching_metadata_is_not_calibration_proof(self):
        result = validate_video({"codec_type": "video", "nb_frames": "10", "width": 1920, "height": 1080, "avg_frame_rate": "30000/1001"},
                                10, {"resolution": [1920, 1080], "fps": 29.970030})
        self.assertTrue(result["dimensions_match_calibration_candidate"])
        self.assertFalse(result["calibration_validated"])

    def test_mismatched_count_rejected(self):
        with self.assertRaises(ValueError):
            validate_video({"codec_type": "video", "nb_frames": "9"}, 10, {})

    def test_mismatched_count_marks_alignment_unresolved_when_allowed(self):
        r = validate_video({"codec_type": "video", "nb_frames": "11", "width": 1, "height": 1, "avg_frame_rate": "30/1"},
                           10, {"resolution": [1, 1], "fps": 30.}, allow_count_mismatch=True)
        self.assertEqual(r["frame_index_alignment"], "UNRESOLVED_count_mismatch")
        self.assertFalse(r["calibration_validated"])

    def test_unknown_count_rejected(self):
        with self.assertRaises(ValueError): validate_video({"codec_type": "video"}, 10, {})

    def test_audio_rejected(self):
        with self.assertRaises(ValueError): validate_video({"codec_type": "audio"}, 10, {})

    def test_fps_mismatch(self):
        with self.assertRaises(ValueError):
            validate_video({"codec_type": "video", "nb_frames": "10", "avg_frame_rate": "60/1"}, 10, {"fps": 30.})

    def test_real_decode_pilot_stays_unannotated(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/numpy unavailable")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            video = root / "synthetic.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30., (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(4): writer.write(np.full((48, 64, 3), i * 50, np.uint8))
            writer.release()
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps({"resolution": [64, 48], "fps": 30.}))
            receipt = root / "intake.json"
            receipt.write_text(json.dumps({"source_commit": ETH.COMMIT, "video_timestamp_rows": 4,
                "files": [{"path": "calibration/sony5100/sony5100.json", "sha256": ETH.digests(calibration)[1]}]}))
            queue = root / "queue.json"
            queue.write_text(json.dumps({"source_commit": ETH.COMMIT, "frames": [
                {"frame_id": 2, "opencv_index": 1, "box_xyxy": None,
                 "measurement_eligible": False, "identity_verified": False}]}))
            stream = {"streams": [{"codec_type": "video", "nb_frames": "4", "width": 64, "height": 48,
                                   "avg_frame_rate": "30/1"}]}

            def run(output, *extra):
                args = ["prepare", "--video", str(video), "--queue", str(queue), "--intake-receipt", str(receipt),
                        "--calibration", str(calibration), "--output", str(root / output), "--ffprobe", "mock"] + list(extra)
                with mock.patch.object(sys, "argv", args), mock.patch.object(
                        FRAMES.subprocess, "check_output", return_value=json.dumps(stream)), contextlib.redirect_stdout(io.StringIO()):
                    FRAMES.main()
                return json.loads((root / output / "receipt.json").read_text())

            result = run("output")
            self.assertFalse(result["calibration_validated"])
            self.assertIsNone(result["frames"][0]["box_xyxy"])
            self.assertEqual(result["container_index_offset"], -1)
            self.assertEqual(result["frames"][0]["container_index"], 1)
            image = cv2.imread(str(root / "output/frame_000002.png"))
            self.assertAlmostEqual(float(image.mean()), 50., delta=3.)

            # The offset is a hypothesis, not a constant: offset 0 must render the NEXT container frame.
            shifted = run("output_offset0", "--index-offset", "0")
            self.assertEqual(shifted["container_index_offset"], 0)
            self.assertEqual(shifted["container_index_rule"], "container_index = frame_id + 0")
            self.assertEqual(shifted["frames"][0]["container_index"], 2)
            self.assertEqual(shifted["frames"][0]["queue_declared_opencv_index"], 1)
            image = cv2.imread(str(root / "output_offset0/frame_000002.png"))
            self.assertAlmostEqual(float(image.mean()), 100., delta=3.)

    def test_index_offset_outside_the_video_is_refused(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/numpy unavailable")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            video = root / "synthetic.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30., (64, 48))
            for i in range(4):
                writer.write(np.full((48, 64, 3), i * 50, np.uint8))
            writer.release()
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps({"resolution": [64, 48], "fps": 30.}))
            receipt = root / "intake.json"
            receipt.write_text(json.dumps({"source_commit": ETH.COMMIT, "video_timestamp_rows": 4,
                "files": [{"path": "calibration/sony5100/sony5100.json", "sha256": ETH.digests(calibration)[1]}]}))
            queue = root / "queue.json"
            queue.write_text(json.dumps({"source_commit": ETH.COMMIT, "frames": [
                {"frame_id": 4, "opencv_index": 3, "box_xyxy": None,
                 "measurement_eligible": False, "identity_verified": False}]}))
            stream = {"streams": [{"codec_type": "video", "nb_frames": "4", "width": 64, "height": 48,
                                   "avg_frame_rate": "30/1"}]}
            args = ["prepare", "--video", str(video), "--queue", str(queue), "--intake-receipt", str(receipt),
                    "--calibration", str(calibration), "--output", str(root / "out"), "--ffprobe", "mock",
                    "--index-offset", "0"]
            with mock.patch.object(sys, "argv", args), mock.patch.object(
                    FRAMES.subprocess, "check_output", return_value=json.dumps(stream)), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    FRAMES.main()


class AlignmentSelectionTest(unittest.TestCase):
    def setUp(self):
        self.values = {100: 6.0, 110: 5.5, 400: 4.5, 405: 4.4, 900: 1.0}
        self.ranges = {f: 60.0 for f in self.values}
        self.ranges[110] = 10.0

    def test_prefers_motion_keeps_separation_and_respects_range(self):
        chosen = SELECT.select(self.values, self.ranges, 4.0, 25.0, 30, 10)
        self.assertEqual(chosen, [100, 400])          # 110 too close AND too near, 405 too close, 900 too slow
        self.assertEqual(SELECT.select(self.values, self.ranges, 4.0, 25.0, 1, 10), [100, 400, 405])

    def test_out_of_view_frames_are_excluded(self):
        in_frame = {100: None, 400: [10., 10.], 405: [20., 20.], 110: [1., 1.], 900: [1., 1.]}
        self.assertEqual(SELECT.select(self.values, self.ranges, 4.0, 25.0, 30, 10, in_frame), [400])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError): SELECT.select(self.values, self.ranges, 4.0, 25.0, 0, 10)
        with self.assertRaises(ValueError): SELECT.select(self.values, self.ranges, 4.0, 25.0, 30, 0)

    def test_level_camera_axis_and_degeneracy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")
        R = SELECT.level_camera(30., 20.)
        axis = R.T @ np.array([0., 0., 1.])
        self.assertAlmostEqual(math.degrees(math.atan2(axis[1], axis[0])), 30., places=6)
        self.assertAlmostEqual(math.degrees(math.asin(axis[2])), 20., places=6)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)
        self.assertAlmostEqual(float(R[0][2]), 0.0, places=9)   # roll-free: image x axis stays horizontal
        with self.assertRaises(ValueError): SELECT.level_camera(0., 90.)

    def test_predicted_pixel_rejects_behind_and_out_of_frame(self):
        K = [[1500., 0., 960.], [0., 1500., 540.], [0., 0., 1.]]
        R = SELECT.level_camera(0., 0.)
        self.assertIsNone(SELECT.predicted_pixel(R, [0., 0., 0.], [-50., 0., 0.], K, [0.] * 5, [1920, 1080], 0))
        centre = SELECT.predicted_pixel(R, [0., 0., 0.], [50., 0., 0.], K, [0.] * 5, [1920, 1080], 0)
        self.assertAlmostEqual(centre[0], 960., places=6)
        self.assertIsNone(SELECT.predicted_pixel(R, [0., 0., 0.], [50., 40., 0.], K, [0.] * 5, [1920, 1080], 0))
        self.assertIsNone(SELECT.predicted_pixel(R, [0., 0., 0.], [50., 0., 0.], K, [0.] * 5, [1920, 1080], 1000))


class CalibrationImageTest(unittest.TestCase):
    TREE = {"tree": [{"type": "blob", "path": "calibration/sony5100/calibration_images/00000.jpg"},
                     {"type": "blob", "path": "calibration/sony5100/calibration_images/00010.jpg"},
                     {"type": "blob", "path": "calibration/sony5100/sony5100.json"},
                     {"type": "blob", "path": "calibration/sonyG/calibration_images/00000.jpg"},
                     {"type": "tree", "path": "calibration/sony5100/calibration_images"}]}

    def test_only_that_cameras_images(self):
        paths = FETCH.image_paths(self.TREE, "sony5100")
        self.assertEqual(paths, ["calibration/sony5100/calibration_images/00000.jpg",
                                 "calibration/sony5100/calibration_images/00010.jpg"])
        self.assertEqual(FETCH.image_paths(self.TREE, "sonyG"),
                         ["calibration/sonyG/calibration_images/00000.jpg"])
        self.assertEqual(FETCH.image_paths(self.TREE, "gopro3"), [])

    def test_held_out_error_recovers_a_known_board(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/numpy unavailable")
        K = [[1500., 0., 960.], [0., 1500., 540.], [0., 0., 1.]]
        dist = [-0.05, 0.01, 0., 0., 0.]
        grid = np.zeros((42, 3), np.float32)
        grid[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        samples = []
        for i in range(3):
            rvec = np.array([0.1 * i, -0.2 + 0.1 * i, 0.05])
            tvec = np.array([-3.0 + i, -2.5, 12.0 + i])
            corners, _ = cv2.projectPoints(grid, rvec, tvec, np.asarray(K), np.asarray(dist))
            samples.append((grid, corners.reshape(-1, 2)))
        exact = CALIB.held_out_error(K, dist, samples)
        self.assertLess(exact["rms_px"], 1e-3)
        self.assertEqual(exact["boards"], 3)
        wrong = CALIB.held_out_error([[1400., 0., 960.], [0., 1400., 540.], [0., 0., 1.]], dist, samples)
        self.assertGreater(wrong["rms_px"], exact["rms_px"] * 10)


class TrackingTest(unittest.TestCase):
    def test_centroid_finds_a_dark_blob_and_refuses_flat_sky(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")
        sky = np.full((120, 160), 200, np.float32)
        self.assertIsNone(TRACK.centroid(sky, (80, 60), 14, 25.0, 1500))
        sky[58:62, 78:86] = 60
        found = TRACK.centroid(sky, (80, 60), 14, 25.0, 1500)
        self.assertAlmostEqual(found["x"], 81.5, delta=0.6)
        self.assertAlmostEqual(found["y"], 59.5, delta=0.6)
        self.assertEqual(found["extent_x"], 8)
        self.assertEqual(found["extent_y"], 4)
        self.assertIsNone(TRACK.centroid(sky, (80, 60), 14, 25.0, 4))

    def test_smoothness_noise_measures_only_the_wobble(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")
        rng = np.random.default_rng(0)
        track = {}
        for i in range(60):
            track[i] = {"x": 100 + 3.0 * i + 0.01 * i * i + rng.normal(0, 0.5),
                        "y": 50 + 0.5 * i + rng.normal(0, 0.5)}
        noise = TRACK.smoothness_noise(track)
        self.assertAlmostEqual(noise["rms_px"], 0.5 * 2 ** 0.5, delta=0.3)
        self.assertIsNone(TRACK.smoothness_noise({0: {"x": 1., "y": 1.}}))


class CameraModelAuditTest(unittest.TestCase):
    K = [[1500., 0., 960.], [0., 1500., 540.], [0., 0., 1.]]
    D = [-0.01, 0.02, 0.0, 0.0, -0.1]

    def test_parameters_apply_in_a_fixed_order(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")
        K, d = AUDIT.apply_parameters(self.K, self.D, [-3.6, -0.2, 0.5, -0.7, 12.0, -8.0, 0.99],
                                      ("k", "pp", "f"))
        self.assertAlmostEqual(d[0], -0.2)
        self.assertAlmostEqual(d[1], 0.5)
        self.assertAlmostEqual(d[4], -0.7)
        self.assertAlmostEqual(K[0][2], 972.0)
        self.assertAlmostEqual(K[1][2], 532.0)
        self.assertAlmostEqual(K[0][0], 1485.0)

    def test_published_hypothesis_changes_nothing(self):
        K, d = AUDIT.apply_parameters(self.K, self.D, [-3.6], ())
        self.assertEqual(K.tolist(), self.K)
        self.assertEqual(list(d), self.D)

    def test_initial_vector_matches_the_free_set(self):
        self.assertEqual(AUDIT.initial((), self.D, -3.6), [-3.6])
        self.assertEqual(len(AUDIT.initial(("k",), self.D, -3.6)), 4)
        self.assertEqual(len(AUDIT.initial(("k", "pp", "f"), self.D, -3.6)), 7)
        self.assertEqual(AUDIT.initial(("f",), self.D, -3.6)[1], 1.0)

    def test_every_hypothesis_is_named_once(self):
        names = [n for n, _ in AUDIT.HYPOTHESES]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("published", names)


class SizeRangeTest(unittest.TestCase):
    FOCAL = 1545.7

    def _points(self, ranges, size_m=0.24, block_offset=0.0, scale=1.0):
        return [{"frame_id": i, "t": block_offset + i * 0.1, "range_m": r,
                 "sqrt_pixels": scale * self.FOCAL * size_m / r, "extent_y": 1.0, "extent_x": 1.0,
                 "x": 900.0, "y": 500.0, "tracking_status": [0.0, 0.0], "bracket_gap_s": 0.1}
                for i, r in enumerate(ranges)]

    def test_fit_recovers_the_generating_size(self):
        points = self._points([40., 60., 80., 100.])
        self.assertAlmostEqual(E3S.fit_size(points, self.FOCAL, "sqrt_pixels"), 0.24, places=9)

    def test_perfect_size_gives_zero_error_and_a_scale_error_is_proportional(self):
        points = self._points([40., 60., 80.])
        self.assertTrue(all(abs(e) < 1e-12 for e in E3S.relative_errors(points, 0.24, self.FOCAL, "sqrt_pixels")))
        errors = E3S.relative_errors(points, 0.24 * 1.1, self.FOCAL, "sqrt_pixels")
        for e in errors:
            self.assertAlmostEqual(e, 0.1, places=9)   # range error is scale-free, as the estimator is

    def test_blocks_are_fifteen_seconds_and_frames_are_not_the_sample(self):
        self.assertEqual(E3S.block_index(0.0), 0)
        self.assertEqual(E3S.block_index(14.999), 0)
        self.assertEqual(E3S.block_index(15.0), 1)
        grouped = E3S.group_blocks(self._points([50.] * 3, block_offset=14.8))
        self.assertEqual(sorted(grouped), [0, 1])

    def test_leave_one_block_out_never_fits_on_the_evaluated_block(self):
        blocks = {0: self._points([60.] * 25, block_offset=0.0),
                  1: self._points([70.] * 25, block_offset=20.0),
                  2: self._points([80.] * 25, block_offset=40.0, scale=1.25)}
        result = E3S.leave_one_block_out(blocks, self.FOCAL, "sqrt_pixels")
        self.assertEqual(sorted(result), [0, 1, 2])
        # Evaluating block 0 fits on blocks 1 and 2, whose pooled median size sits midway between the two
        # scales: 0.27 against the block's own 0.24, so +12.5 percent. The odd block is never in its own fit.
        self.assertAlmostEqual(result[0]["median_signed_rel"], 0.125, places=6)
        self.assertAlmostEqual(result[1]["median_signed_rel"], 0.125, places=6)
        # Block 2 is judged by a fit that saw only the other scale: 0.24 / 0.30 - 1.
        self.assertAlmostEqual(result[2]["median_signed_rel"], -0.2, places=6)
        self.assertEqual(result[2]["frames"], 25)
        self.assertNotAlmostEqual(result[2]["fitted_size_m"], result[2]["points"][0]["sqrt_pixels"]
                                  * result[2]["points"][0]["range_m"] / self.FOCAL, places=3)

    def test_bins_refuse_to_report_without_enough_blocks(self):
        per_block = {0: {"points": self._points([55.] * 25), "errors": [0.05] * 25,
                         "frames": 25, "median_abs_rel": 0.05}}
        bins = E3S.bin_summary(per_block)
        self.assertEqual(bins["50-70 m"]["status"], "INSUFFICIENT")
        self.assertNotIn("median_abs_rel", bins["50-70 m"])

    def test_bins_report_once_three_blocks_contribute(self):
        per_block = {i: {"points": self._points([55.] * 25), "errors": [0.05 + 0.01 * i] * 25,
                         "frames": 25, "median_abs_rel": 0.05} for i in range(3)}
        bins = E3S.bin_summary(per_block)
        entry = bins["50-70 m"]
        self.assertEqual(entry["status"], "REPORTED")
        self.assertEqual(entry["blocks"], 3)
        self.assertAlmostEqual(entry["median_abs_rel"], 0.06, places=6)
        self.assertAlmostEqual(entry["median_abs_error_m"], 0.06 * 55, places=4)

    def test_gates_are_the_preregistered_ones(self):
        good = {i: {"frames": 100, "median_abs_rel": 0.10} for i in range(6)}
        bins = {"a": {"status": "REPORTED"}, "b": {"status": "REPORTED"}, "c": {"status": "REPORTED"}}
        self.assertEqual(E3S.verdict(good, bins)["status"], "SIZE_RANGE_USABLE")
        few = {i: {"frames": 100, "median_abs_rel": 0.10} for i in range(5)}
        self.assertIn("G1_coverage", E3S.verdict(few, bins)["failed_gates"])
        large = {i: {"frames": 100, "median_abs_rel": 0.40} for i in range(6)}
        self.assertIn("G2_central", E3S.verdict(large, bins)["failed_gates"])
        unstable = {i: {"frames": 100, "median_abs_rel": 0.02 + 0.08 * i} for i in range(6)}
        self.assertIn("G3_stability", E3S.verdict(unstable, bins)["failed_gates"])
        self.assertIn("G4_range_coverage", E3S.verdict(good, {"a": {"status": "REPORTED"}})["failed_gates"])

    def test_thresholds_match_the_preregistration_text(self):
        text = (Path(__file__).resolve().parents[1] / "results/eth_ds5_e3s_2026-09-10/PREREGISTRATION.md").read_text()
        self.assertIn("0.25", text)
        self.assertIn("0.20", text)
        self.assertIn("at least 6 evaluation blocks", text)
        self.assertIn("at least 500 evaluated frames", text)
        self.assertEqual((E3S.G2_MEDIAN_ABS_REL, E3S.G3_BLOCK_SPREAD), (0.25, 0.20))
        self.assertEqual((E3S.MIN_EVAL_BLOCKS, E3S.MIN_EVAL_FRAMES), (6, 500))
        self.assertEqual(E3S.RANGE_BINS, ((30., 50.), (50., 70.), (70., 90.), (90., 110.)))

    def test_border_frames_are_excluded(self):
        track = {"track": {"1": {"x": 10.0, "y": 500.0, "pixels": 100, "extent_x": 5, "extent_y": 5},
                           "2": {"x": 900.0, "y": 500.0, "pixels": 100, "extent_x": 5, "extent_y": 5}}}
        pose = [[t, 0., 0., 60., 0., 0., 0., .01, .01, .01, 0.] for t in (0.0, 0.2)]
        rows = E3S.load_points(track, pose, {1: 0.05, 2: 0.10}, [0., 0., 0.], 0.0, 1 / 30.)
        self.assertEqual([r["frame_id"] for r in rows], [2])

    def test_zero_distortion_leaves_size_untouched(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")
        K = [[1500., 0., 960.], [0., 1500., 540.], [0., 0., 1.]]
        m = E3S.radial_magnification([[960., 540.], [1400., 800.]], K, [0.] * 5)
        self.assertTrue(np.allclose(m, 1.0))
        barrel = E3S.radial_magnification([[1400., 800.]], K, [-0.1, 0., 0., 0., 0.])
        self.assertLess(barrel[0], 1.0)


class AttitudeGateTest(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest("numpy/scipy unavailable")
        self.np, self.Rotation = np, Rotation

    def test_local_acceleration_is_exact_on_a_quadratic(self):
        np = self.np
        times = np.linspace(0, 4, 41)
        positions = np.stack([0.5 * 3.0 * times ** 2, 2.0 * times, np.full_like(times, 7.0)], axis=1)
        accel = GATE.local_acceleration(times, positions, half=4)
        inner = ~np.isnan(accel[:, 0])
        self.assertTrue(np.allclose(accel[inner, 0], 3.0, atol=1e-8))
        self.assertTrue(np.allclose(accel[inner, 1:], 0.0, atol=1e-8))

    def test_direction_error_is_zero_when_aligned_and_flat_when_opposed(self):
        np = self.np
        axes = np.array([[1.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
        accel = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
        errors, good = GATE.direction_error_deg(axes, accel)
        self.assertTrue(good.all())
        self.assertAlmostEqual(errors[0], 0.0, places=6)
        self.assertAlmostEqual(errors[1], 180.0, places=6)

    def test_the_generating_convention_wins_by_a_margin(self):
        """Accelerations built from one reading of the angles must single that reading out."""
        np, Rotation = self.np, self.Rotation
        rng = np.random.default_rng(3)
        rpy = np.column_stack([rng.uniform(-12, 12, 400), rng.uniform(-12, 12, 400), rng.uniform(-180, 180, 400)])
        truth = GATE.candidate_axes(rpy, "xyz", False, True, -1)
        tilt = np.degrees(np.arccos(np.clip(np.cos(np.radians(rpy[:, 0])) * np.cos(np.radians(rpy[:, 1])), -1, 1)))
        horizontal = truth[:, :2] / np.maximum(np.linalg.norm(truth[:, :2], axis=1, keepdims=True), 1e-9)
        accel = np.zeros((400, 3))
        accel[:, :2] = (GATE.GRAVITY * np.tan(np.radians(tilt)))[:, None] * horizontal
        scores, usable = GATE.score_conventions(rpy, accel, tilt)
        ranked = sorted(scores, key=lambda n: scores[n]["median_direction_error_deg"])
        self.assertEqual(scores[ranked[0]]["order"], "xyz")
        self.assertEqual(scores[ranked[0]]["body_sign"], -1)
        self.assertLess(scores[ranked[0]]["median_direction_error_deg"], 1e-6)
        margin = scores[ranked[1]]["median_direction_error_deg"] - scores[ranked[0]]["median_direction_error_deg"]
        self.assertGreater(margin, GATE.C1_MIN_MARGIN_DEG)

    def test_thrust_and_drag_fit_recovers_known_coefficients(self):
        np = self.np
        rng = np.random.default_rng(11)
        n = 300
        tilt = rng.uniform(2, 14, n)
        heading = rng.uniform(-np.pi, np.pi, n)
        axes = np.column_stack([np.cos(heading), np.sin(heading), np.full(n, 0.9)])
        velocity = np.column_stack([rng.uniform(-6, 6, n), rng.uniform(-6, 6, n), np.zeros(n)])
        speed = np.linalg.norm(velocity, axis=1)[:, None]
        thrust = (GATE.GRAVITY * np.tan(np.radians(tilt)))[:, None] * axes[:, :2]
        accel = np.zeros((n, 3))
        accel[:, :2] = 0.9 * thrust - 0.03 * speed * velocity[:, :2]
        fit = GATE.thrust_and_drag_fit(axes, tilt, velocity, accel, np.ones(n, bool))
        self.assertAlmostEqual(fit["thrust_gain"], 0.9, places=6)
        self.assertAlmostEqual(fit["drag_coefficient"], -0.03, places=6)
        self.assertGreater(fit["correlation"], 0.99)

    def test_occupancy_counts_frames_and_distinct_blocks(self):
        records = [{"range_m": 75.0, "aspect_deg": 50.0, "block": b} for b in (1, 1, 2)]
        table = GATE.occupancy(records)
        self.assertEqual(table["70-90 m / 45-60 deg"], {"frames": 3, "blocks": 2})
        self.assertEqual(table["30-50 m / 0-45 deg"], {"frames": 0, "blocks": 0})

    def test_gate_thresholds_are_declared_constants(self):
        self.assertEqual(GATE.C1_MAX_MEDIAN_DEG, 30.0)
        self.assertEqual(GATE.C1_MIN_MARGIN_DEG, 10.0)
        self.assertEqual(GATE.C2B_MIN_RESIDUAL_IMPROVEMENT, 0.20)
        self.assertEqual(GATE.C3_MAX_ASPECT_RANGE_CORRELATION, 0.7)
        self.assertEqual(GATE.C4_MAX_TIMING_FRACTION, 0.05)


class AttitudeReliabilityTest(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")
        self.np = np

    def _flight(self, n=900, seed=5):
        """A synthetic flight whose horizontal acceleration really is produced by its tilt."""
        np = self.np
        rng = np.random.default_rng(seed)
        times = np.arange(n) * 0.111
        heading = np.cumsum(rng.normal(0, 0.05, n))
        tilt = 6.0 + 4.0 * np.sin(np.linspace(0, 9, n))
        roll = tilt * np.cos(heading)
        pitch = tilt * np.sin(heading)
        rpy = np.column_stack([roll, pitch, np.degrees(heading)])
        axes = RELY.candidate_axes(rpy, **RELY.CONVENTION)
        horizontal = axes[:, :2] / np.linalg.norm(axes[:, :2], axis=1, keepdims=True)
        true_tilt = np.degrees(np.arccos(np.clip(np.cos(np.radians(roll)) * np.cos(np.radians(pitch)), -1, 1)))
        accel = np.zeros((n, 3))
        accel[:, :2] = 0.9 * (RELY.GRAVITY * np.tan(np.radians(true_tilt)))[:, None] * horizontal
        velocity = np.cumsum(accel, axis=0) * 0.111
        positions = np.cumsum(velocity, axis=0) * 0.111
        return times, positions, rpy

    def test_a_real_attitude_stream_passes_at_a_long_window(self):
        times, positions, rpy = self._flight()
        result = RELY.evaluate_window(times, positions, rpy, half=7, permutations=40, seed=1)
        self.assertNotEqual(result.get("status"), "INSUFFICIENT_SAMPLES")
        for direction in result["directions"].values():
            self.assertTrue(direction["beats_null"], direction)
            self.assertGreater(direction["held_out_correlation"], direction["null_percentile_99"])

    def test_a_shuffled_attitude_stream_does_not_beat_its_own_null(self):
        np = self.np
        times, positions, rpy = self._flight()
        scrambled = rpy[np.random.default_rng(7).permutation(len(rpy))]
        result = RELY.evaluate_window(times, positions, scrambled, half=7, permutations=40, seed=1)
        if result.get("status") == "INSUFFICIENT_SAMPLES":
            self.skipTest("scrambled flight left too few usable samples")
        self.assertFalse(all(d["pass"] for d in result["directions"].values()))

    def test_fit_and_score_never_scores_on_the_fitted_rows(self):
        np = self.np
        rng = np.random.default_rng(0)
        regressors = rng.normal(size=(200, 2))
        target = regressors @ np.array([0.9, -0.03]) + rng.normal(0, 0.01, 200)
        out = RELY.fit_and_score(regressors[:100], target[:100], regressors[100:], target[100:])
        self.assertAlmostEqual(out["thrust_gain"], 0.9, places=2)
        self.assertAlmostEqual(out["drag_coefficient"], -0.03, places=2)
        self.assertGreater(out["held_out_correlation"], 0.99)

    def test_constant_prediction_scores_zero_instead_of_nan(self):
        np = self.np
        regressors = np.zeros((40, 2))
        target = np.arange(40, dtype=float)
        self.assertEqual(RELY.fit_and_score(regressors, target, regressors, target)["held_out_correlation"], 0.0)

    def test_the_window_sweep_and_thresholds_match_the_preregistration(self):
        text = (Path(__file__).resolve().parents[1]
                / "results/eth_ds5_e3p_reliability_2026-09-10/PREREGISTRATION.md").read_text()
        self.assertIn("{2, 3, 4, 5, 7, 9}", text)
        self.assertIn("200", text)
        self.assertIn("0.35", text)
        self.assertEqual(RELY.HALF_WINDOWS, (2, 3, 4, 5, 7, 9))
        self.assertEqual(RELY.PERMUTATIONS, 200)
        self.assertEqual(RELY.MIN_HELD_OUT_CORRELATION, 0.35)
        self.assertEqual(RELY.THRUST_GAIN_RANGE, (0.5, 1.5))


class ExtractionTest(unittest.TestCase):
    def test_single_video(self):
        self.assertEqual(video_member("header\n----------\nPath = cam0.mp4\nSize = 123\n")["Size"], "123")

    def test_traversal_rejected(self):
        with self.assertRaises(ValueError): video_member("----------\nPath = ../cam0.mp4\nSize = 123\n")

    def test_symlink_rejected(self):
        with self.assertRaises(ValueError): video_member("----------\nPath = cam0.mp4\nSize = 123\nSymbolic Link = /etc/passwd\n")

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            video_member("----------\nPath = cam0.mp4\nSize = 123\n\nPath = sub/cam0.mp4\nSize = 123\n")


class TimestampTest(unittest.TestCase):
    PERIOD = 1001 / 30000

    def test_affine_hypothesis_detected_and_raw_rejected(self):
        pts = [i * self.PERIOD for i in range(2000)]
        published = [1.01 * p + 5. for p in pts]
        r = VERIFY.timestamp_consistency(pts, published, 1.01, 5., self.PERIOD)
        self.assertTrue(r["alignments"]["packet_offset_0"]["synchronised_affine"]["within_half_frame"])
        self.assertFalse(r["alignments"]["packet_offset_0"]["raw_camera_time"]["within_half_frame"])
        self.assertNotIn("packet_offset_1", r["alignments"])

    def test_extra_container_frame_reports_both_alignments(self):
        pts = [i * self.PERIOD for i in range(11)]
        published = [1.0 * p + 5. for p in pts[1:]]
        r = VERIFY.timestamp_consistency(pts, published, 1.0, 5., self.PERIOD)
        self.assertEqual(sorted(r["alignments"]), ["packet_offset_0", "packet_offset_1"])
        self.assertAlmostEqual(r["alignments"]["packet_offset_1"]["synchronised_affine"]["offset_first_row_s"], 0.)
        self.assertAlmostEqual(r["alignments"]["packet_offset_0"]["synchronised_affine"]["offset_first_row_frames"], 1.)

    def test_published_fit_identifies_generating_scale(self):
        sync = {0: (1.000004, 10.), 1: (1.00003, 7.)}
        ids = list(range(1, 501))
        published = [1.00003 * self.PERIOD * (i + 2) + 10. for i in ids]
        fit = VERIFY.published_fit(ids, published, self.PERIOD, sync)
        self.assertEqual(fit["closest_camera_scale"], 1)
        self.assertAlmostEqual(fit["intercept_minus_cam0_shift_frames"], 2., places=3)
        self.assertLess(fit["max_abs_residual_s"], 1e-9)

    def test_provenance_detects_split_scale_and_shift_rows(self):
        """The real file's slope is cam1's scale while its origin only fits cam0's shift."""
        sync = {0: (1.000004517, 10.48625553), 1: (1.000031087, 7.72945441), 2: (0.997549813, 9.80204540)}
        slope, intercept = 0.033367703956, 10.551275280
        r = VERIFY.stamp_provenance(slope, intercept, sync, self.PERIOD)
        self.assertEqual(r["scale_matches_camera"], 1)
        self.assertEqual(r["origin_matches_camera"], 0)
        self.assertFalse(r["single_row_explains_file"])
        cam0 = r["cameras"]["0"]
        self.assertEqual(cam0["nearest_integer_origin"], 2)
        self.assertLess(cam0["origin_distance_to_integer"], 0.06)
        self.assertLess(abs(cam0["shift_revision_needed_s"]), 0.002)

    def test_provenance_accepts_a_self_consistent_table(self):
        sync = {0: (1.000004517, 10.48625553), 1: (1.000031087, 7.72945441)}
        slope = 1.000004517 * self.PERIOD
        intercept = 10.48625553 + 2 * slope
        r = VERIFY.stamp_provenance(slope, intercept, sync, self.PERIOD)
        self.assertTrue(r["single_row_explains_file"])
        self.assertEqual(r["scale_matches_camera"], 0)
        self.assertEqual(r["cameras"]["0"]["nearest_integer_origin"], 2)

    def test_alignment_candidates_stay_unresolved(self):
        edits = [{"handler": "soun", "edits": [{"media_time": 0, "media_time_frames": 0.}]},
                 {"handler": "vide", "edits": [{"media_time": 1001, "media_time_frames": 1.0}]}]
        r = VERIFY.index_alignment_candidates(edits, 20970, 20969)
        self.assertEqual(r["missing_rows"], 1)
        self.assertFalse(r["resolved"])
        supported = [c for c in r["candidates"] if c["supported_by_edit_list"]]
        self.assertEqual([c["container_index_of_frame_id"] for c in supported], ["frame_id"])
        self.assertEqual(len(r["candidates"]), 3)

    def test_alignment_candidate_unsupported_without_a_video_edit(self):
        r = VERIFY.index_alignment_candidates([{"handler": "vide", "edits": []}], 20970, 20969)
        self.assertIsNone(r["video_edit_list_frames"])
        self.assertFalse(any(c["supported_by_edit_list"] for c in r["candidates"]))

    def test_edit_frames_use_the_track_timescale_not_an_assumed_rate(self):
        """A 25 fps track at a 12800 timescale: 512 media units is one frame, not 512/1001."""
        tracks = [{"handler": "vide", "timescale": 12800, "sample_duration": 512,
                   "edits": [{"media_time": 512, "media_time_frames": 512 / 512}]}]
        self.assertEqual(VERIFY.video_edit_frames(tracks), 1.0)

    def test_video_edit_frames_ignores_other_tracks(self):
        tracks = [{"handler": "soun", "edits": [{"media_time": 0, "media_time_frames": 0.}]},
                  {"handler": "vide", "edits": [{"media_time": 1001, "media_time_frames": 1.0}]}]
        self.assertEqual(VERIFY.video_edit_frames(tracks), 1.0)
        self.assertIsNone(VERIFY.video_edit_frames([{"handler": "soun", "edits": [{"media_time": 5, "media_time_frames": 5.}]}]))

    def test_edit_lists_parsed_per_track_from_the_real_file(self):
        video = Path("/home/fair/workspaces/aerial_gym_ws/datasets/eth_ds5_cam0_extracted/cam0.mp4")
        if not video.is_file():
            self.skipTest("extracted cam0.mp4 unavailable")
        tracks = VERIFY.edit_list_offsets(video)
        self.assertEqual([t["handler"] for t in tracks], ["vide", "soun", "meta"])
        self.assertEqual(VERIFY.video_edit_frames(tracks), 1.0)
        self.assertEqual([t["edits"][0]["media_time"] for t in tracks if t["handler"] != "vide"], [0, 0])

    def test_calibration_candidate_never_validated_by_metadata(self):
        r = VERIFY.calibration_candidate_check({"width": 1920, "height": 1080, "avg_frame_rate": "30000/1001"},
                                               {"encoder": "AVC Coding", "com.apple.quicktime.model": "X"},
                                               {"resolution": [1920, 1080], "fps": 29.97003})
        self.assertTrue(r["resolution_matches"] and r["fps_matches"])
        self.assertFalse(r["calibration_validated"])
        self.assertIn("com.apple.quicktime.model", r["camera_model_tags"])


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.pose = [[t, t, 2 * t, 3., 0., 0., 170. + 20. * t, .01, .01, .01, 1.] for t in (0., 1., 2., 5.)]

    def test_interpolation_and_yaw_wrap(self):
        gt = REVIEW.interpolate_pose(self.pose, 0.5, max_gap_s=1.5)
        self.assertAlmostEqual(gt["xyz_m"][1], 1.)
        self.assertAlmostEqual(gt["speed_mps"], (1 + 4) ** .5)
        # 170 -> 190 wraps through the branch cut; the result is the same angle, normalised into range
        yaw = gt["rpy_deg_uninterpreted"][2]
        self.assertAlmostEqual(REVIEW.wrap_deg(yaw - 180.), 0.)
        self.assertTrue(-180. <= yaw <= 180.)

    def test_interpolated_angles_stay_in_range(self):
        wrapping = [[0., 0, 0, 0, 350., -170., 359.9, 0, 0, 0, 0],
                    [0.1, 0, 0, 0, -350., 170., 0.1, 0, 0, 0, 0]]
        rpy = REVIEW.interpolate_pose(wrapping, 0.05, max_gap_s=1)["rpy_deg_uninterpreted"]
        for angle in rpy:
            self.assertTrue(-180. <= angle <= 180., angle)
        self.assertAlmostEqual(REVIEW.wrap_deg(rpy[2] - 0.), 0.)

    def test_no_extrapolation_or_long_gap(self):
        self.assertIsNone(REVIEW.interpolate_pose(self.pose, -0.1, max_gap_s=1.5))
        self.assertIsNone(REVIEW.interpolate_pose(self.pose, 5.1, max_gap_s=1.5))
        self.assertIsNone(REVIEW.interpolate_pose(self.pose, 0.5))  # default 0.5 s gap limit refuses 1 s brackets
        self.assertIsNone(REVIEW.interpolate_pose(self.pose, 3., max_gap_s=1.5))
        self.assertIsNotNone(REVIEW.interpolate_pose(self.pose, 3., max_gap_s=5.))

    def test_bearing(self):
        g = REVIEW.bearing_from_camera([0., 0., 0.], [0., 3., 4.])
        self.assertAlmostEqual(g["slant_range_m"], 5.)
        self.assertAlmostEqual(g["azimuth_deg_from_east_ccw"], 90.)
        self.assertAlmostEqual(g["elevation_deg"], 53.130102354)

    def test_motion_candidates_find_only_moving_blob(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")
        prev, cur, nxt = [np.full((60, 80, 3), 100, np.uint8) for _ in range(3)]
        for img in (prev, cur, nxt):
            img[5:15, 5:15] = 0  # static dark square must not be a candidate
        cur[30:34, 40:44] = 255
        boxes, moving = REVIEW.motion_candidates(prev, cur, nxt)
        self.assertEqual(len(boxes), 1)
        x1, y1, x2, y2 = boxes[0]["xyxy"]
        self.assertTrue(x1 <= 40 and y1 <= 30 and x2 >= 44 and y2 >= 34)
        panel = REVIEW.render_panel(cur, boxes, ["a", "b"])
        self.assertGreater(panel.shape[0], cur.shape[0])


class DistortionTest(unittest.TestCase):
    def test_fold_back_radius_of_the_real_calibration(self):
        limit = REPROJ.valid_distortion_radius([-0.011232359677, 0.045931232417, 0.000263868094,
                                                -0.001253638454, -0.151703077572])
        self.assertTrue(1.0 < limit < 1.1)          # image corner is at 0.713, comfortably inside
        r = limit * 0.999
        self.assertGreater(self._radial(r, -0.011232359677, 0.045931232417, -0.151703077572),
                           self._radial(r * 0.99, -0.011232359677, 0.045931232417, -0.151703077572))
        beyond = limit * 1.3
        self.assertLess(self._radial(beyond, -0.011232359677, 0.045931232417, -0.151703077572),
                        self._radial(limit, -0.011232359677, 0.045931232417, -0.151703077572))

    @staticmethod
    def _radial(r, k1, k2, k3):
        return r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)

    def test_no_distortion_never_folds(self):
        self.assertEqual(REPROJ.valid_distortion_radius([0., 0., 0., 0., 0.]), float("inf"))


class ReprojectionTest(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV/numpy unavailable")
        self.np, self.cv2 = np, cv2
        self.K = [[1500., 0., 960.], [0., 1500., 540.], [0., 0., 1.]]
        self.dist = [-0.01, 0.04, 0.0002, -0.001, -0.15]
        self.camera = [14.84, 6.939, 1.494]
        R0, _ = cv2.Rodrigues(np.array([0.3, -1.2, 0.1]))
        self.R = R0
        rng = np.random.default_rng(0)
        self.world = [self.camera + R0.T @ np.array([x, y, z]) for x, y, z in
                      rng.uniform([-15., -8., 30.], [15., 8., 100.], size=(12, 3))]
        self.pixels = REPROJ.reproject(self.R, self.camera, self.world, self.K, self.dist)

    def test_recovers_rotation_and_centre(self):
        rays_c = REPROJ.camera_rays(self.pixels, self.K, self.dist)
        rays_w = self.np.asarray(self.world) - self.np.asarray(self.camera)
        rays_w /= self.np.linalg.norm(rays_w, axis=1, keepdims=True)
        R, angular = REPROJ.fit_rotation(rays_c, rays_w)
        self.assertLess(self.np.abs(R - self.R).max(), 1e-6)
        self.assertLess(angular.max(), 1e-4)
        centre = REPROJ.pnp_centre(self.world, self.pixels, self.K, self.dist)
        self.assertLess(self.np.linalg.norm(self.np.asarray(centre) - self.camera), 1e-3)

    def _reviewed(self, pixels):
        return [{"frame_id": i + 1, "opencv_index": i, "project_timestamp_s": 1. + 0.1 * i, "centre": list(p)} for i, p in enumerate(pixels)]

    def _pose(self):
        return [[1. + 0.1 * i] + list(w) + [0., 0., 0., .01, .01, .01, 1.] for i, w in enumerate(self.world)]

    def test_evaluate_consistent_and_wrong_focal_inconsistent(self):
        ok = REPROJ.evaluate(self._reviewed(self.pixels), self._pose(), self.camera, self.K, self.dist, 1 / 30., 0)
        self.assertEqual(ok["status"], "CONSISTENT")
        wrong_K = [[1200., 0., 960.], [0., 1200., 540.], [0., 0., 1.]]
        bad = REPROJ.evaluate(self._reviewed(self.pixels), self._pose(), self.camera, wrong_K, self.dist, 1 / 30., 0)
        self.assertEqual(bad["status"], "INCONSISTENT")
        few = REPROJ.evaluate(self._reviewed(self.pixels)[:3], self._pose(), self.camera, self.K, self.dist, 1 / 30., 0)
        self.assertEqual(few["status"], "INSUFFICIENT_REVIEWED_BOXES")

    def test_points_behind_the_camera_are_refused(self):
        world = list(self.world) + [self.camera - self.R.T @ self.np.array([0., 0., 50.])]
        pixels = list(self.pixels) + [[960., 540.]]
        reviewed = [{"frame_id": i + 1, "opencv_index": i, "project_timestamp_s": 1. + 0.1 * i, "centre": list(p)}
                    for i, p in enumerate(pixels)]
        pose = [[1. + 0.1 * i] + list(w) + [0., 0., 0., .01, .01, .01, 1.] for i, w in enumerate(world)]
        r = REPROJ.evaluate(reviewed, pose, self.camera, self.K, self.dist, 1 / 30., 0)
        self.assertEqual(r["status"], "POINTS_BEHIND_FITTED_CAMERA")
        self.assertGreaterEqual(r["points_behind_camera"], 1)

    def test_points_past_the_fold_back_radius_are_refused(self):
        """A target far outside the field of view still projects into the image once the model folds."""
        limit = REPROJ.valid_distortion_radius(self.dist)
        far = self.camera + self.R.T @ self.np.array([limit * 2.5 * 40., 0., 40.])
        world = list(self.world) + [far]
        reviewed = [{"frame_id": i + 1, "opencv_index": i, "project_timestamp_s": 1. + 0.1 * i,
                     "centre": list(p)} for i, p in enumerate(list(self.pixels) + [[500., 500.]])]
        pose = [[1. + 0.1 * i] + list(w) + [0., 0., 0., .01, .01, .01, 1.] for i, w in enumerate(world)]
        r = REPROJ.evaluate(reviewed, pose, self.camera, self.K, self.dist, 1 / 30., 0)
        self.assertEqual(r["status"], "POINTS_OUTSIDE_VALID_DISTORTION_RADIUS")
        self.assertGreater(r["max_normalized_radius"], r["valid_radius_limit"])

    def test_discrimination_reports_pixels_per_frame(self):
        pose = [[t, 0., 100., 0., 0., 0., 0., .01, .01, .01, 1.] for t in (0.,)]
        pose = [[0.0, 0., 100., 0., 0., 0., 0., .01, .01, .01, 1.],
                [0.1, 1., 100., 0., 0., 0., 0., .01, .01, .01, 1.]]
        d = REPROJ.alignment_discrimination(pose, [0., 0., 0.], [0.02], 1 / 30., 1500.)
        self.assertEqual(d["frames"], 1)
        self.assertAlmostEqual(d["median_px"], 1500. * math.atan(1 / 3. / 100.), delta=0.05)

    def test_discrimination_empty_outside_support(self):
        pose = [[0.0, 0., 100., 0., 0., 0., 0., .01, .01, .01, 1.],
                [0.1, 1., 100., 0., 0., 0., 0., .01, .01, .01, 1.]]
        self.assertEqual(REPROJ.alignment_discrimination(pose, [0., 0., 0.], [5.0], 1 / 30., 1500.)["frames"], 0)

    def test_refine_finds_a_non_integer_optimum_and_flags_it(self):
        curve = lambda x: (x + 3.35) ** 2 + 1.0
        r = REPROJ.refine_shift(curve, -3)
        self.assertAlmostEqual(r["shift_frames"], -3.35, places=2)
        self.assertFalse(r["is_integer_relabelling"])
        self.assertGreater(r["distance_to_nearest_integer_frames"], 0.2)

    def test_refine_accepts_an_integer_optimum(self):
        r = REPROJ.refine_shift(lambda x: (x + 2.0) ** 2, -2)
        self.assertAlmostEqual(r["shift_frames"], -2.0, places=3)
        self.assertTrue(r["is_integer_relabelling"])

    def test_refine_skips_shifts_without_a_score(self):
        r = REPROJ.refine_shift(lambda x: None if x < -1.5 else (x + 1.0) ** 2, -1)
        self.assertGreaterEqual(r["shift_frames"], -1.5)

    def test_search_range_covers_more_than_the_frame_ambiguity(self):
        self.assertLessEqual(min(REPROJ.ALIGNMENT_SHIFTS), -8)
        self.assertGreaterEqual(max(REPROJ.ALIGNMENT_SHIFTS), 8)

    def test_true_offset_recovered_from_the_winning_shift(self):
        self.assertEqual(REPROJ.true_index_offset(0, 0), 0)     # edit-list candidate
        self.assertEqual(REPROJ.true_index_offset(0, 1), -1)    # reader dropped the last frame
        self.assertEqual(REPROJ.true_index_offset(-1, 0), -1)   # the first pilot's own rendering
        self.assertEqual(REPROJ.true_index_offset(-1, -1), 0)

    def test_review_csv_requires_visible_and_valid_box(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "review.csv"
            path.write_text("frame_id,opencv_index,project_timestamp_s,drone0_visible,box_x1,box_y1,box_x2,box_y2\n"
                            "1,0,1.0,yes,10,10,20,20\n2,1,2.0,no,,,,\n")
            rows = REPROJ.load_review(path)
            self.assertEqual([r["frame_id"] for r in rows], [1])
            self.assertEqual(rows[0]["centre"], [15., 15.])
            path.write_text("frame_id,opencv_index,project_timestamp_s,drone0_visible,box_x1,box_y1,box_x2,box_y2\n1,0,1.0,yes,20,10,10,20\n")
            with self.assertRaises(ValueError): REPROJ.load_review(path)


if __name__ == "__main__":
    unittest.main()
