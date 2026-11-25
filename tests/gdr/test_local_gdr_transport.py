"""Compile both transport variants against CPU-only CUDA and verbs doubles."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class LocalGdrTransportTest(unittest.TestCase):
    def test_transport_contract(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler required")
        fixture = Path(__file__).resolve().parent
        csrc = fixture.parents[1] / "csrc"
        with tempfile.TemporaryDirectory() as temp:
            for backend in ("native", "mapped"):
                with self.subTest(backend=backend):
                    binary = Path(temp) / backend
                    flags = ["-DFLEXKV_GDR_MAPPED"] if backend == "mapped" else []
                    subprocess.run([
                        compiler, "-std=c++11", "-pedantic-errors", "-pthread", *flags,
                        "-I", str(fixture / "fakes"), "-I", str(csrc),
                        str(fixture / "transport_contract.cpp"), "-o", str(binary),
                    ], check=True, timeout=30)
                    for fault in ("", "partial", "sequence", "wc"):
                        subprocess.run([str(binary), fault], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
