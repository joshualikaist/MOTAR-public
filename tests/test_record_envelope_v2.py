"""Synthetic bytes/temporary Git repositories only; no policy, simulator or checkpoint loader."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import record_envelope_v2 as v2
import check_record_integrity as checker


class EnvelopeV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base/"repo"
        self.repo.mkdir()
        self.checkpoint = self.repo/"synthetic.bin"
        self.config = self.repo/"config.json"
        self.checkpoint.write_bytes(b"synthetic opaque bytes\x00\x01\xff")
        self.config.write_bytes(b'{"example":true}\n')
        (self.repo/"source.txt").write_text("synthetic producer source\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Envelope test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, stderr=subprocess.PIPE, text=True).strip()

    def capture(self, **overrides):
        args = dict(repository=self.repo, seed=7, density_bars=12, checkpoint_path=self.checkpoint,
                    config_path=self.config, expected_records=2, source_files=("source.txt",))
        args.update(overrides)
        return v2.preflight(**args)

    def document(self):
        return {**self.capture().metadata, "records": [{"record_id": "one", "value": .25}, {"record_id": "two", "value": None}]}

    def test_valid_publish_and_receipt(self):
        capture = self.capture()
        path = self.base/"artifact.json"
        result = v2.publish(capture, [{"record_id": "one"}, {"record_id": "two"}], path)
        self.assertEqual(result["status"], "POSTFLIGHT_PASS")
        self.assertTrue(all(x == "PASS" for x in result["checks"].values()))
        self.assertEqual(result["artifact_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(json.loads(path.read_text())["run_id"], capture.metadata["run_id"])
        self.assertTrue(path.with_name(path.name+".receipt.json").exists())

    def test_real_checkpoint_and_config_bytes_define_identity(self):
        metadata = self.capture().metadata
        self.assertEqual(metadata["checkpoint_sha256"], hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        self.assertEqual(metadata["config_sha256"], hashlib.sha256(self.config.read_bytes()).hexdigest())
        self.assertEqual(metadata["checkpoint_path"], str(self.checkpoint))
        self.assertNotEqual(metadata["checkpoint_path"], metadata["checkpoint_sha256"])

    def test_null_seed_fails_before_artifact_generation(self):
        with self.assertRaisesRegex(v2.EnvelopeError, "INVALID_SEED"):
            self.capture(seed=None)

    def test_null_density_fails_before_artifact_generation(self):
        with self.assertRaisesRegex(v2.EnvelopeError, "INVALID_DENSITY_BARS"):
            self.capture(density_bars=None)

    def test_boolean_is_not_integer_metadata(self):
        for key in ("seed", "density_bars", "expected_records"):
            with self.assertRaises(v2.EnvelopeError):
                self.capture(**{key: True})

    def test_missing_checkpoint_and_config_fail(self):
        for key in ("checkpoint_path", "config_path"):
            with self.assertRaisesRegex(v2.EnvelopeError, "INPUT_FILE_UNAVAILABLE"):
                self.capture(**{key: self.base/"absent"})

    def test_path_uppercase_and_wrong_length_sha_rejected(self):
        document = self.document()
        for digest in (str(self.checkpoint), "a"*63, "a"*65, "A"*64):
            changed = deepcopy(document)
            changed["checkpoint_sha256"] = digest
            with self.assertRaisesRegex(v2.EnvelopeError, "INVALID_CHECKPOINT_SHA256"):
                v2.validate_document(changed)

    def test_checkpoint_single_byte_mutation_detected(self):
        capture = self.capture()
        data = self.checkpoint.read_bytes()
        self.checkpoint.write_bytes(bytes([data[0]^1])+data[1:])
        with self.assertRaisesRegex(v2.EnvelopeError, "CHECKPOINT_HASH_MISMATCH"):
            v2.publish(capture, [{"record_id": "1"}, {"record_id": "2"}], self.base/"bad.json")
        self.assertFalse((self.base/"bad.json").exists())

    def test_config_mutation_detected(self):
        capture = self.capture()
        self.config.write_bytes(b'{"example":false}\n')
        with self.assertRaisesRegex(v2.EnvelopeError, "CONFIG_HASH_MISMATCH"):
            v2.verify_inputs(capture.metadata, self.repo)

    def test_source_mutation_detected(self):
        capture = self.capture()
        (self.repo/"source.txt").write_text("changed\n")
        with self.assertRaisesRegex(v2.EnvelopeError, "SOURCE_FILE_HASH_MISMATCH"):
            v2.verify_inputs(capture.metadata, self.repo)

    def test_git_head_change_detected(self):
        capture = self.capture()
        self.git("-c", "user.name=Envelope test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-qm", "changed")
        with self.assertRaisesRegex(v2.EnvelopeError, "SOURCE_GIT_COMMIT_CHANGED"):
            v2.verify_inputs(capture.metadata, self.repo)

    def test_dirty_is_observed_not_assumed_clean(self):
        self.assertFalse(self.capture().metadata["source_dirty"])
        (self.repo/"untracked.txt").write_text("dirty\n")
        self.assertTrue(self.capture().metadata["source_dirty"])

    def test_null_or_symbolic_commit_rejected(self):
        document = self.document()
        for commit in (None, "HEAD", "a"*39):
            changed = deepcopy(document); changed["source_git_commit"] = commit
            with self.assertRaisesRegex(v2.EnvelopeError, "INVALID_SOURCE_GIT_COMMIT"):
                v2.validate_document(changed)

    def test_git_unavailable_fails_closed(self):
        with self.assertRaisesRegex(v2.EnvelopeError, "GIT_PROVENANCE_UNAVAILABLE"):
            self.capture(repository=self.base)

    def test_context_row_mismatch_rejected(self):
        document = self.document()
        for key, value in (("seed", 9), ("density_bars", 9), ("checkpoint_sha256", "0"*64)):
            changed = deepcopy(document); changed["records"][0][key] = value
            with self.assertRaisesRegex(v2.EnvelopeError, "CONTEXT_RECORD_MISMATCH"):
                v2.validate_document(changed)

    def test_context_row_equal_allowed_but_not_required(self):
        document = self.document()
        document["records"][0]["seed"] = document["seed"]
        v2.validate_document(document)

    def test_duplicate_record_id_rejected(self):
        document = self.document(); document["records"][1]["record_id"] = "one"
        with self.assertRaisesRegex(v2.EnvelopeError, "DUPLICATE_RECORD_ID"):
            v2.validate_document(document)

    def test_expected_count_cannot_be_changed_with_row_count(self):
        document = self.document(); document["records"].pop(); document["record_count"] = 1
        with self.assertRaisesRegex(v2.EnvelopeError, "EXPECTED_RECORD_COUNT_MISMATCH"):
            v2.validate_document(document, expected_records=2)

    def test_nan_rejected(self):
        document = self.document(); document["records"][0]["value"] = float("nan")
        with self.assertRaises(v2.EnvelopeError):
            v2.validate_document(document)

    def test_inf_and_overflow_json_rejected(self):
        document = self.document(); document["records"][0]["value"] = float("inf")
        with self.assertRaises(v2.EnvelopeError):
            v2.validate_document(document)
        raw = v2.encoded(self.document()).replace(b'0.25', b'1e400')
        self.assertEqual(v2.audit_v2_bytes(raw)["status"], "INVALID_FOR_DECLARED_CONTRACT")

    def test_duplicate_json_keys_rejected(self):
        raw = v2.encoded(self.document()).replace(b'"schema_version":2', b'"schema_version":2,"schema_version":2')
        self.assertIn("STRICT_JSON_INVALID", v2.audit_v2_bytes(raw)["issue_counts"])

    def test_circular_record_rejected_without_hanging(self):
        document = self.document(); document["records"][0]["cycle"] = document
        with self.assertRaises(v2.EnvelopeError):
            v2.validate_document(document)

    def test_path_traversal_and_symlink_source_escape_rejected(self):
        for path in ("../outside", "/etc/hosts", "./source.txt"):
            with self.assertRaises(v2.EnvelopeError):
                self.capture(source_files=(path,))
        outside = self.base/"outside.txt"; outside.write_text("external\n")
        (self.repo/"link.txt").symlink_to(outside)
        with self.assertRaisesRegex(v2.EnvelopeError, "SOURCE_PATH_ESCAPES_REPOSITORY"):
            self.capture(source_files=("link.txt",))

    def test_capture_metadata_cannot_be_mutated_through_returned_copy(self):
        capture = self.capture(); metadata = capture.metadata; metadata["seed"] = 999
        self.assertEqual(capture.metadata["seed"], 7)

    def test_run_id_unique_across_captures_stable_within_capture(self):
        first, second = self.capture(), self.capture()
        self.assertNotEqual(first.metadata["run_id"], second.metadata["run_id"])
        self.assertEqual(first.metadata["run_id"], first.metadata["run_id"])

    def test_same_bytes_at_different_paths_have_same_identity(self):
        copied = self.repo/"copy.bin"; copied.write_bytes(self.checkpoint.read_bytes())
        first, second = self.capture().metadata, self.capture(checkpoint_path=copied).metadata
        self.assertNotEqual(first["checkpoint_path"], second["checkpoint_path"])
        self.assertEqual(first["checkpoint_sha256"], second["checkpoint_sha256"])

    def test_schema_hash_and_version_are_checked(self):
        for key, value in (("schema_sha256", "0"*64), ("schema_version", 1), ("schema_version", True)):
            document = self.document(); document[key] = value
            with self.assertRaises(v2.EnvelopeError):
                v2.validate_document(document)

    def test_timestamp_and_record_id_required(self):
        document = self.document(); document["created_at_utc"] = "2026-02-30T00:00:00.000000Z"
        with self.assertRaises(v2.EnvelopeError):
            v2.validate_document(document)
        document = self.document(); del document["records"][0]["record_id"]
        with self.assertRaisesRegex(v2.EnvelopeError, "RECORD_ID_REQUIRED"):
            v2.validate_document(document)

    def test_existing_output_is_never_overwritten(self):
        output = self.base/"existing.json"; output.write_bytes(b"preserved")
        with self.assertRaisesRegex(v2.EnvelopeError, "OUTPUT_ALREADY_EXISTS"):
            v2.publish(self.capture(), [], output)
        self.assertEqual(output.read_bytes(), b"preserved")

    def test_postflight_detects_artifact_byte_mutation(self):
        capture = self.capture(); document = {**capture.metadata, "records": [{"record_id": "a"}, {"record_id": "b"}]}
        raw = v2.encoded(document); output = self.base/"artifact.json"; output.write_bytes(raw+b" ")
        with self.assertRaisesRegex(v2.EnvelopeError, "ARTIFACT_HASH_MISMATCH"):
            v2.postflight(output, capture, hashlib.sha256(raw).hexdigest())

    def test_failed_postflight_keeps_raw_and_no_success_receipt(self):
        capture = self.capture(); output = self.base/"artifact.json"
        with patch.object(v2, "postflight", side_effect=v2.EnvelopeError("CHECKPOINT_HASH_MISMATCH")):
            with self.assertRaises(v2.EnvelopeError):
                v2.publish(capture, [{"record_id": "a"}, {"record_id": "b"}], output)
        self.assertTrue(output.exists())
        self.assertFalse(output.with_name(output.name+".receipt.json").exists())
        self.assertEqual(json.loads(output.with_name(output.name+".failure.json").read_text())["status"], "INVALID_ARTIFACT")

    def test_structural_check_does_not_read_embedded_file_paths(self):
        raw = v2.encoded(self.document())
        with patch.object(v2, "verify_inputs", side_effect=AssertionError("must be explicit")):
            self.assertEqual(v2.audit_v2_bytes(raw)["file_identity_validation"], "NOT_RUN")

    def test_v2_checker_cli_file_verification_is_explicit(self):
        capture = self.capture(); artifact = self.base/"artifact.json"
        v2.publish(capture, [{"record_id": "a"}, {"record_id": "b"}], artifact)
        with redirect_stdout(io.StringIO()):
            result = checker.main(["--input", str(artifact), "--envelope-v2", "--expected-records", "2",
                                   "--verify-files", "--repository", str(self.repo), "--output", str(self.base/"check.json")])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads((self.base/"check.json").read_text())["file_identity_validation"], "PASS")

    def test_historical_file_sha_and_invalid_v1_verdict_are_preserved(self):
        path = ROOT/"results/task_diagnostics_invariance_2026-09-14/on_episode_forensics.json"
        raw = path.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "1475b0c226e9dd546938687e575dcfe22d45af0c7603f6d6b5fde2e92278870a")
        contract = json.loads((ROOT/"docs/record_envelope_audit_2026-09-14.json").read_text())
        result = checker.audit_bytes(raw, contract)
        self.assertEqual(result["status"], "INVALID_FOR_DECLARED_CONTRACT")
        self.assertEqual(result["issue_counts"], {"INVALID_SHA256_FORMAT": 129, "NULL_REQUIRED_VALUE": 258})
        self.assertEqual(v2.audit_v2_bytes(raw)["status"], "INVALID_FOR_DECLARED_CONTRACT")

    def test_generic_module_import_is_stdlib_only(self):
        code = "import sys; sys.path.insert(0,'tools'); import record_envelope_v2; assert not any(x in sys.modules for x in ('torch','warp','aerial_gym','isaacgym'))"
        subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)

    def test_invalid_schema_file_fails_before_generation(self):
        path = self.base/"broken-schema.json"
        for value in (b'not json', b'{}'):
            path.write_bytes(value)
            with patch.object(v2, "SCHEMA", path), self.assertRaises(v2.EnvelopeError):
                self.capture()

    def test_non_regular_checkpoint_rejected_without_read(self):
        with self.assertRaisesRegex(v2.EnvelopeError, "INPUT_NOT_REGULAR_FILE"):
            self.capture(checkpoint_path=self.repo)


class StatusBoundaryTests(unittest.TestCase):
    """Passing unit tests is not a live validation, and the documents must keep saying so.

    The tempting failure here is a later edit that promotes the status to METADATA_FIXED because
    the suite is green. Green means the producer refuses the right inputs over synthetic files; it
    says nothing about a runtime artifact, because none has been produced through this producer.
    """

    STATUS = ROOT/"docs/record_envelope_v2_status.json"
    DOC = ROOT/"docs/record_envelope_v2_2026-09-14.md"

    def status(self):
        return json.loads(self.STATUS.read_text())

    LIVE_VALUES = ("NOT_RUN", "NOT_APPLICABLE", "FAIL", "PASS")

    def test_live_generation_validation_is_one_of_the_declared_values(self):
        status = self.status()
        self.assertEqual(status["producer_unit_validation"], "PASS")
        self.assertIn(status["live_generation_validation"], self.LIVE_VALUES)

    def test_live_pass_requires_a_live_artifact_receipt(self):
        """PASS is the one value that cannot be asserted from a document alone.

        The failure this guards against is a later edit that promotes the status because the suite
        is green, or because a feasibility check was run and read as a success. PASS therefore has
        to point at a receipt that a real producer wrote, and that receipt has to exist.
        """
        status = self.status()
        if status["live_generation_validation"] != "PASS":
            self.assertIn("live_validation_evidence", status)
            return
        receipt = status.get("live_validation_receipt")
        self.assertIsNotNone(receipt, "PASS without a live receipt path")
        self.assertTrue((ROOT/receipt).is_file(), receipt)

    def test_a_not_applicable_status_records_why(self):
        status = self.status()
        if status["live_generation_validation"] != "NOT_APPLICABLE":
            return
        self.assertIn("schema_design_limitation", status)
        evidence = ROOT/status["live_validation_evidence"]
        self.assertTrue(evidence.is_file(), str(evidence))
        text = evidence.read_text()
        self.assertIn("LIVE_GENERIC_VALIDATION_NOT_APPLICABLE", text)
        for field in ("density_bars", "checkpoint_path", "config_path"):
            self.assertIn(field, text)

    def test_no_document_claims_the_metadata_is_fixed(self):
        """The token may appear only inside a denial, never as an assertion.

        Matching one exact sentence would break the moment the wording changed and would quietly
        stop guarding anything; this checks the claim instead of the phrasing.
        """
        for path in (self.STATUS, self.DOC):
            text = path.read_text()
            position = text.find("METADATA_FIXED")
            while position != -1:
                window = text[max(0, position - 90):position]
                self.assertRegex(window, r"\bnot\b|\bNothing\b",
                                 "%s asserts METADATA_FIXED" % path.name)
                position = text.find("METADATA_FIXED", position + 1)

    def test_the_historical_verdict_is_recorded_as_still_invalid(self):
        historical = self.status()["historical_artifact"]
        self.assertEqual(historical["byte_integrity"], "MATCH")
        self.assertEqual(historical["metadata_contract"], "INVALID_FOR_DECLARED_CONTRACT")
        self.assertFalse(historical["repaired"])
        self.assertFalse(historical["backfilled"])
        artifact = ROOT/historical["path"]
        self.assertTrue(artifact.is_file())
        self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(), historical["sha256"])

    def test_the_document_states_the_boundary_and_links_the_audit(self):
        text = self.DOC.read_text()
        # The document must state the status the machine-readable record actually carries.
        self.assertIn("live_generation_validation = %s" % self.status()["live_generation_validation"],
                      text)
        self.assertIn("record_integrity_review_2026-09-14", text)
        self.assertIn("INVALID_FOR_DECLARED_CONTRACT", text)


if __name__ == "__main__":
    unittest.main()
