"""Unit tests for Gen3 Nickel raw data ingest."""

import os
import unittest

import lsst.utils.tests
import lsst.pex.exceptions
from lsst.afw.image import FilterLabel
from lsst.obs.nickel import Nickel
from lsst.obs.base.ingest_tests import IngestTestBase

testDataPackage = "testdata_nickel"
try:
    testDataDirectory = lsst.utils.getPackageDir(testDataPackage)
except lsst.pex.exceptions.NotFoundError:
    testDataDirectory = None


@unittest.skipIf(testDataDirectory is None, "testdata_nickel must be set up")
class TestNickelIngest(IngestTestBase, lsst.utils.tests.TestCase):
    instrumentClassName = "lsst.obs.nickel.Nickel"

    def setUp(self):
        self.ingestdir = os.path.dirname(__file__)
        self.instrument = Nickel()
        self.file = os.path.join(testDataDirectory, "nickel", "raw", "d1032.fits")
        self.dataIds = [dict(instrument="Nickel", exposure=1032, detector=0)]
        self.visits = None
        self.outputRun = "test_run"
        self.filterLabel = FilterLabel(band="b", physical="B")
        super().setUp()


    # def testDefineVisits(self):
    #     self.skipTest("Nickel does not define visits; skipping testDefineVisits.")


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
