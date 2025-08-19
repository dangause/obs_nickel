# This file is part of obs_nickel.
#
# Developed for the LSST Data Management System.
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

"""Tests of the Nickel instrument class.
"""

import unittest

import lsst.utils.tests
from lsst.obs.base.instrument_tests import InstrumentTests, InstrumentTestData
import lsst.obs.nickel


class TestNickel(InstrumentTests, lsst.utils.tests.TestCase):
    def setUp(self):
        # Match what is actually registered by the instrument
        physical_filters = {"B", "V", "R", "I"}

        self.data = InstrumentTestData(
            name="Nickel",
            nDetectors=1,
            firstDetectorName="CCD0",
            physical_filters=physical_filters
        )

        self.instrument = lsst.obs.nickel.Nickel()



class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == '__main__':
    lsst.utils.tests.init()
    unittest.main()
