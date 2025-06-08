"""Unit tests for Gen3 Nickel raw data ingest."""

import os
import unittest

import lsst.utils.tests
import lsst.pex.exceptions

from lsst.obs.nickel import Nickel
from lsst.obs.base.ingest_tests import IngestTestBase

testDataPackage = "testdata_nickel"
try:
    testDataDirectory = lsst.utils.getPackageDir(testDataPackage)
except lsst.pex.exceptions.NotFoundError:
    testDataDirectory = None

@unittest.skipIf(testDataDirectory is None, "testdata_nickel must be set up")
class TestNickelIngest(IngestTestBase, lsst.utils.tests.TestCase):
    instrumentClassName = "lsst.obs.nickel.Nickel"  # ← ADD THIS LINE

    def setUp(self):
        self.ingestdir = os.path.dirname(__file__)
        self.instrument = Nickel()
        self.file = os.path.join(testDataDirectory, "nickel", "raw", "d1032.fits")
        self.dataId = dict(instrument="Nickel", exposure=1032, detector=0)
        super().setUp()

    # def test_ingest(self):
    #     self.runIngestTest()


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
