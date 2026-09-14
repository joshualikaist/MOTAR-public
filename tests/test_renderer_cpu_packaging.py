"""Static standalone packaging contracts. No installs, network, renderer or simulator imports."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "requirements-renderer-cpu.txt"


def entries():
    return [line.strip() for line in PROFILE.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


class RendererCpuPackagingTest(unittest.TestCase):
    def test_runtime_dependencies_have_exact_versions_or_immutable_sources(self):
        names = []
        for line in entries():
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)(?:==([^\s]+)| @ (\S+))", line)
            self.assertIsNotNone(match, line)
            names.append(match.group(1).lower().replace("_", "-"))
        self.assertEqual(len(names), len(set(names)))

    def test_legacy_numpy_and_networkx_contract_is_explicit(self):
        self.assertIn("numpy==1.23.0", entries())
        self.assertIn("networkx==2.2", entries())
        self.assertIn("scipy==1.10.1", entries())
        self.assertIn("trimesh==4.11.5", entries())

    def test_urdfpy_is_not_the_broken_same_version_pypi_artifact(self):
        self.assertIn("urdfpy @ git+https://github.com/mmatl/urdfpy.git@"
                      "5466842899b33bd549e8f9e2a9a987bd5e37373b", entries())
        self.assertNotIn("urdfpy==0.0.22", entries())

    def test_cpu_torch_wheel_has_platform_and_content_pin(self):
        torch = [line for line in entries() if line.startswith("torch @ ")]
        self.assertEqual(len(torch), 1)
        self.assertEqual(torch[0], "torch @ https://download.pytorch.org/whl/cpu/"
                         "torch-2.4.1%2Bcpu-cp38-cp38-linux_x86_64.whl#sha256="
                         "0c0a7cc4f7c74ff024d5a5e21230a01289b65346b27a626f6c815d94b4b8c955")

    def test_profile_is_not_a_simulator_or_local_checkout_install(self):
        text = "\n".join(entries()).lower()
        for forbidden in ("isaacgym", "pytorch3d", "aerial_gym", "rl-games", "sample-factory",
                          "file://", "/home/", "-e ", "-r ", "--extra-index-url"):
            self.assertNotIn(forbidden, text)
        self.assertIn("warp-lang==1.0.0", entries())

    def test_profile_does_not_claim_cross_platform_or_hermetic_installation(self):
        text = PROFILE.read_text()
        for limitation in ("Linux x86-64 / CPython 3.8 only", "NEW venv", "pip==25.0.1",
                           "not a hermetic/hash-locked build environment"):
            self.assertIn(limitation, text)


if __name__ == "__main__":
    unittest.main()
