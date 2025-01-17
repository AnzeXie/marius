import unittest
import shutil
from pathlib import Path
import pytest
import os
import marius as m
from test.python.constants import TMP_TEST_DIR, TESTING_DATA_DIR


class TestFB15K(unittest.TestCase):

    @classmethod
    def setUp(self):
        if not Path(TMP_TEST_DIR).exists():
            Path(TMP_TEST_DIR).mkdir()

    @classmethod
    def tearDown(self):
        pass
        if Path(TMP_TEST_DIR).exists():
            shutil.rmtree(Path(TMP_TEST_DIR))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_one_epoch(self):
        pass
