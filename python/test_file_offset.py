"""Tests Config.file_offset: a binary model with other data around it.

Needs the build_binary executable (on PATH, or its path in $KENLM_BUILD_BINARY).
Run from the repository root: python -m pytest python/test_file_offset.py
"""

import mmap
import os
import shutil
import subprocess
import tempfile
import unittest

import kenlm

ARPA = os.path.join(os.path.dirname(__file__), "..", "lm", "test.arpa")
SENTENCE = "looking on a little the more , loin of a word"


@unittest.skipUnless(
    os.environ.get("KENLM_BUILD_BINARY") or shutil.which("build_binary"),
    "build_binary not found",
)
class FileOffsetTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        build_binary = os.environ.get("KENLM_BUILD_BINARY") or shutil.which(
            "build_binary"
        )
        self.plain = os.path.join(self.dir.name, "plain.binary")
        subprocess.run([build_binary, ARPA, self.plain], check=True, capture_output=True)
        with open(self.plain, "rb") as f:
            self.binary = f.read()

    def embedded(self, before, after=b""):
        path = os.path.join(self.dir.name, "embedded.binary")
        with open(path, "wb") as f:
            f.write(b"\1" * before + self.binary + after)
        return path

    def config(self, offset, method=None):
        config = kenlm.Config()
        config.file_offset = offset
        if method is not None:
            config.load_method = method
        return config

    def test_default_offset_is_zero(self):
        self.assertEqual(kenlm.Config().file_offset, 0)

    def test_offset_model_scores_like_the_plain_file(self):
        expected = kenlm.Model(self.plain).score(SENTENCE)
        page = mmap.PAGESIZE
        for before in (page, 5 * page):
            for after in (b"", b"trailing data", b"\0" * (2 * page + 3)):
                for method in (
                    kenlm.LoadMethod.LAZY,
                    kenlm.LoadMethod.POPULATE_OR_READ,
                    kenlm.LoadMethod.READ,
                ):
                    path = self.embedded(before, after)
                    model = kenlm.Model(path, self.config(before, method))
                    self.assertEqual(model.score(SENTENCE), expected)

    def test_unaligned_offset_is_an_error_unless_reading(self):
        offset = mmap.PAGESIZE + 512
        path = self.embedded(offset)
        with self.assertRaises(OSError):
            kenlm.Model(path, self.config(offset, kenlm.LoadMethod.LAZY))
        model = kenlm.Model(path, self.config(offset, kenlm.LoadMethod.READ))
        self.assertEqual(model.score(SENTENCE), kenlm.Model(self.plain).score(SENTENCE))

    def test_no_model_at_the_offset_is_an_error(self):
        path = self.embedded(2 * mmap.PAGESIZE)
        with self.assertRaises(OSError):
            kenlm.Model(path, self.config(mmap.PAGESIZE))


if __name__ == "__main__":
    unittest.main()
