import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "tools/navrl_sim2real_ingest.py"
_SPEC = importlib.util.spec_from_file_location("navrl_sim2real_ingest_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class Sim2RealIngestTests(unittest.TestCase):
    def _write_csv(self, path, *, include_optional=True):
        fields = list(_MODULE.REQUIRED_COLUMNS)
        if include_optional:
            fields += list(_MODULE.OPTIONAL_COLUMNS)
        fields += ["trial_id", "calibration_id"]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            row = {
                "topic": "camera",
                "seq": "0",
                "source_stamp_ns": "100",
                "host_receive_stamp_ns": "110",
                "frame_id": "camera_optical_frame",
                "parent_frame_id": "base_link",
                "sync_group": "0",
                "policy_input_stamp_ns": "",
                "command_publish_stamp_ns": "",
                "trial_id": "t0",
                "calibration_id": "cal0",
            }
            writer.writerow({key: value for key, value in row.items() if key in fields})

    def test_real_csv_is_lossless_and_manifested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.csv"
            output = root / "events.jsonl"
            self._write_csv(source)
            count = _MODULE.convert_csv(
                source, output, run_id="trial-run", source_kind="real_log"
            )
            self.assertEqual(count, 1)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[0]["source_kind"], "real_log")
            self.assertEqual(rows[0]["input_sha256"], _MODULE._sha256(source))
            self.assertEqual(rows[1]["seq"], 0)
            self.assertEqual(rows[1]["extra"], {"trial_id": "t0", "calibration_id": "cal0"})

    def test_missing_required_column_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.csv"
            source.write_text("topic,seq\ncamera,0\n", encoding="utf-8")
            with self.assertRaises(_MODULE.IngestError):
                _MODULE.convert_csv(source, root / "out.jsonl", run_id="r", source_kind="real_log")

    def test_bad_integer_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.csv"
            self._write_csv(source, include_optional=False)
            text = source.read_text(encoding="utf-8").replace(",0,100,110,", ",not-a-number,100,110,")
            source.write_text(text, encoding="utf-8")
            with self.assertRaises(_MODULE.IngestError):
                _MODULE.convert_csv(source, root / "out.jsonl", run_id="r", source_kind="real_log")

    def test_synthetic_source_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events.csv"
            self._write_csv(source)
            output = root / "events.jsonl"
            _MODULE.convert_csv(source, output, run_id="fixture", source_kind="synthetic_fixture")
            manifest = json.loads(output.read_text().splitlines()[0])
            self.assertEqual(manifest["source_kind"], "synthetic_fixture")


if __name__ == "__main__":
    unittest.main()
