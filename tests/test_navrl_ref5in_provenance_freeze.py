"""Lock the byte identity of the ref5in runtime artefacts the frozen checkpoint recorded.

`eval_navrl_v2_density_sweep.sh:240-247` compares the sha256 of the robot config module and the
robot URDF against the values stored inside the checkpoint, and exits 2 on any difference. That
check runs BEFORE the `NAVRL_V2_FORCE` provenance override is consulted, so there is no way to
proceed past it: a single changed byte in either file makes the frozen ref5in checkpoint
permanently unevaluatable on that tree.

This has already happened once. Commit 921fb1d (2026-08-20) retargeted one docstring line in
`navrl_ref5in_quad_config.py` from `docs/` to `docs/archive/` while tidying moved documents. The
edit was semantically inert, but it landed on all eight branches and blocked every subsequent
ref5in evaluation until it was reverted on 2026-08-21.

The lesson these tests encode: files under `aerial_gym/config/robot_config/` and
`resources/robots/` are provenance-frozen artefacts with respect to every checkpoint already
trained. Comments and docstrings in them are load-bearing bytes. If one of these tests fails, do
NOT update the expected digest to match the file -- that silently discards the checkpoint lineage.
Restore the file instead, or retire the checkpoint deliberately and say so in WORKLOG.md.
"""

import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Recorded by the frozen ref5in D1 checkpoint
# (sha256 197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e) and independently
# echoed by the last passing ref5in cell, seed 367 camera_20m, in
# results/navrl_ref5in_camera_range_control_seed367/cells/camera_20m/1bars.json ->
# condition.robot_config_sha256 / condition.robot_asset_sha256.
FROZEN_REF5IN_ARTEFACTS = {
    "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py":
        "ebb71802f19b630ba6c2ac4c04b113c269d8bbd3e40e094e126913caa8731297",
    "resources/robots/quad/quad_navrl_ref5in.urdf":
        "5c160b0d19caebf9a4a3c38be861a77637ee0fb2b80febf4ac54d8b143db6a32",
}

REDIRECT_STUB = ROOT / "docs" / "reference_platform_proposal_2026-08.md"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Ref5inProvenanceFreeze(unittest.TestCase):
    def test_frozen_artefacts_match_the_checkpoint(self):
        for relative, expected in sorted(FROZEN_REF5IN_ARTEFACTS.items()):
            path = ROOT / relative
            with self.subTest(artefact=relative):
                self.assertTrue(path.is_file(), "%s is missing" % relative)
                self.assertEqual(
                    _sha256(path),
                    expected,
                    "%s no longer matches the sha256 the frozen ref5in checkpoint recorded. "
                    "Every byte-exact ref5in evaluation is blocked until it does again "
                    "(eval_navrl_v2_density_sweep.sh:240-247, unconditional exit 2). Restore the "
                    "file; do not edit the expected digest." % relative,
                )

    def test_every_doc_path_named_in_the_frozen_config_resolves(self):
        # The reason the freeze was broken was a docstring pointing at a moved document. A path
        # named inside a frozen file cannot be repointed, so it has to stay resolvable some other
        # way -- currently a redirect stub at the original location.
        config = ROOT / "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py"
        text = config.read_text(encoding="utf-8")
        referenced = set()
        for token in text.replace("`", " ").replace("'", " ").split():
            if token.startswith("docs/") and token.endswith(".md"):
                referenced.add(token)
        self.assertTrue(referenced, "expected the frozen config to name at least one doc path")
        for relative in sorted(referenced):
            with self.subTest(doc=relative):
                self.assertTrue(
                    (ROOT / relative).is_file(),
                    "%s is named by the frozen ref5in config but does not exist. Do NOT fix this "
                    "by editing the config -- that breaks the checkpoint provenance freeze. Put a "
                    "redirect at the referenced path instead." % relative,
                )

    def test_the_redirect_stub_explains_why_it_may_not_be_deleted(self):
        self.assertTrue(REDIRECT_STUB.is_file())
        text = REDIRECT_STUB.read_text(encoding="utf-8")
        self.assertIn("provenance-frozen", text)
        self.assertIn("921fb1d", text)
        self.assertIn("archive/reference_platform_proposal_2026-08.md", text)


if __name__ == "__main__":
    unittest.main()
