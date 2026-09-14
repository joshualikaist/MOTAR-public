"""Generic envelope validation with deliberate corruptions; no simulator execution."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("record_integrity", ROOT / "tools/check_record_integrity.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


def fixture():
    contract = {"schema_version": 1, "records_key": "records", "expected_count": 2,
                "fields": {"id": {"type": "integer", "minimum": 0},
                           "seed": {"type": "integer"}, "digest": {"type": "string", "format": "sha256"},
                           "value": {"type": "number", "nullable": True}, "valid": {"type": "boolean"}},
                "unique_fields": ["id"], "context_key": "context", "context_fields": ["seed", "digest"]}
    rows = [{"id": i, "seed": 17, "digest": "a"*64, "value": .5, "valid": True} for i in range(2)]
    document = {"context": {"seed": 17, "digest": "a"*64}, "records": rows}
    return document, contract


class RecordIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.data, self.contract = fixture()

    def audit(self):
        return check.audit_bytes(json.dumps(self.data).encode(), self.contract)

    def test_valid_and_nonmutating(self):
        before = deepcopy(self.data)
        result = self.audit()
        self.assertEqual(result["status"], "VALID_FOR_DECLARED_CONTRACT")
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["duplicate_records"], 0)
        self.assertEqual(self.data, before)

    def test_null_optional_is_not_zero(self):
        self.data["records"][0]["value"] = None
        result = self.audit()
        self.assertEqual(result["status"], "VALID_FOR_DECLARED_CONTRACT")
        self.assertEqual(result["coverage"]["value"]["null"], 1)

    def test_required_null_fails_even_when_context_matches(self):
        self.data["context"]["seed"] = None
        for r in self.data["records"]:
            r["seed"] = None
        self.assertEqual(self.audit()["issue_counts"]["NULL_REQUIRED_VALUE"], 3)

    def test_missing_key_has_distinct_coverage(self):
        del self.data["records"][0]["seed"]
        result = self.audit()
        self.assertEqual(result["coverage"]["seed"], {"missing": 1, "null": 0, "invalid": 0})

    def test_bool_is_not_an_integer_identity(self):
        self.data["records"][0]["id"] = True
        self.assertIn("WRONG_TYPE", self.audit()["issue_counts"])

    def test_number_is_not_a_boolean(self):
        self.data["records"][0]["valid"] = 1
        self.assertIn("WRONG_TYPE", self.audit()["issue_counts"])

    def test_path_is_not_a_sha256_and_is_not_echoed(self):
        self.data["records"][0]["digest"] = "/private/example/model.bin"
        result = self.audit()
        self.assertIn("INVALID_SHA256_FORMAT", result["issue_counts"])
        self.assertNotIn("/private/example", json.dumps(result))

    def test_duplicate_identity_is_retained_and_reported(self):
        self.data["records"][1]["id"] = 0
        result = self.audit()
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["duplicate_records"], 1)

    def test_count_mismatch_is_not_silent_truncation(self):
        self.data["records"].append(deepcopy(self.data["records"][0]))
        self.assertIn("RECORD_COUNT_MISMATCH", self.audit()["issue_counts"])

    def test_negative_id_rejected(self):
        self.data["records"][0]["id"] = -1
        self.assertIn("BELOW_MINIMUM", self.audit()["issue_counts"])

    def test_context_mismatch(self):
        self.data["records"][0]["seed"] = 99
        self.assertEqual(self.audit()["issue_counts"]["CONTEXT_MISMATCH"], 1)

    def test_strict_json_duplicate_keys_and_tokens(self):
        for raw, code in ((b'{"a":1,"a":2}', "DUPLICATE_OBJECT_KEY"),
                          (b'{"a":NaN}', "NONSTANDARD_NUMERIC_TOKEN"),
                          (b'{"a":Infinity}', "NONSTANDARD_NUMERIC_TOKEN")):
            self.assertIn(code, check.audit_bytes(raw, self.contract)["issue_counts"])

    def test_overflow_to_infinity_in_uncontracted_nested_field(self):
        raw = json.dumps(self.data).replace('"records":', '"extra":{"nested":[1e400]},"records":').encode()
        result = check.audit_bytes(raw, self.contract)
        self.assertIn("NONFINITE_DOCUMENT", result["issue_counts"])
        self.assertEqual(result["nonfinite_values"], 1)

    def test_malformed_encoding_and_json(self):
        for raw in (b'\xff', b'{"records":[', b'not-json'):
            self.assertIn("MALFORMED_JSON", check.audit_bytes(raw, self.contract)["issue_counts"])

    def test_no_record_array(self):
        self.assertIn("MISSING_RECORD_ARRAY", check.audit_bytes(b'{}', self.contract)["issue_counts"])

    def test_nonobject_row_counts_as_missing_not_dropped(self):
        self.data["records"][0] = None
        result = self.audit()
        self.assertEqual(result["record_count"], 2)
        self.assertEqual(result["coverage"]["id"]["missing"], 1)

    def test_invalid_context(self):
        self.data["context"] = 1
        self.assertIn("INVALID_CONTEXT", self.audit()["issue_counts"])

    def test_bounded_issue_samples_keep_total_counts(self):
        self.data["records"] = [{"id": i} for i in range(100)]
        result = self.audit()
        self.assertEqual(len(result["issues"]), 40)
        self.assertTrue(result["issues_truncated"])
        self.assertGreater(result["issue_count"], 400)

    def test_input_size_guard(self):
        previous = check.MAX_BYTES
        try:
            check.MAX_BYTES = 4
            with self.assertRaises(ValueError):
                check.audit_bytes(b'{"x": 1}', self.contract)
        finally:
            check.MAX_BYTES = previous

    def test_bad_contract_rejected_before_audit(self):
        for change in ({"schema_version": True}, {"expected_count": True}, {"unique_fields": []},
                       {"unexpected": 1}, {"context_fields": ["not_a_field"]}):
            with self.assertRaises(ValueError):
                check.audit_bytes(b'{}', {**self.contract, **change})

    def test_cli_preserves_inputs_and_refuses_output_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, contract, output = (root/name for name in ("input.json", "contract.json", "report.json"))
            source.write_text(json.dumps(self.data))
            contract.write_text(json.dumps(self.contract))
            before = source.read_bytes()
            args = ["--input", str(source), "--contract", str(contract), "--output", str(output)]
            self.assertEqual(check.main(args), 0)
            result = json.loads(output.read_text())
            self.assertTrue(result["input_unchanged"])
            self.assertEqual(result["sha256"], hashlib.sha256(before).hexdigest())
            with self.assertRaises(FileExistsError):
                check.main(args)
            self.assertEqual(source.read_bytes(), before)

    def test_invalid_cli_still_writes_audit_not_repaired_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, contract, output = (root/name for name in ("input.json", "contract.json", "report.json"))
            self.data["records"][0]["id"] = -1
            source.write_text(json.dumps(self.data))
            contract.write_text(json.dumps(self.contract))
            self.assertEqual(check.main(["--input", str(source), "--contract", str(contract),
                                         "--output", str(output)]), 2)
            self.assertEqual(json.loads(source.read_text())["records"][0]["id"], -1)

    def test_cli_is_stdlib_only(self):
        code = ("import runpy,sys; sys.argv=['check_record_integrity','--help']; "
                "\ntry: runpy.run_path(sys.argv[1] if len(sys.argv)>1 and sys.argv[1]!='--help' else "
                +repr(str(ROOT/"tools/check_record_integrity.py"))+",run_name='__main__')"
                "\nexcept SystemExit as e: assert e.code==0"
                "\nassert not any(m in sys.modules for m in ('torch','warp','isaacgym','aerial_gym'))")
        subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
