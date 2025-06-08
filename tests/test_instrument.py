import unittest
import lsst.utils.tests
import lsst.obs.nickel
from lsst.obs.nickel import Nickel
from lsst.obs.base.instrument_tests import InstrumentTests, InstrumentTestData

class TestNickelCam(InstrumentTests, lsst.utils.tests.TestCase):
    def setUp(self):
        physical_filters = {"B", "V", "R", "I"}  # Your known filters

        self.data = InstrumentTestData(
            name="Nickel",                  # Matches camera.yaml
            nDetectors=1,                   # Replace with actual number
            firstDetectorName="CCD0",          # Replace with actual detector name
            physical_filters=physical_filters,
        )
        self.instrument = Nickel()

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
