import unittest
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

from lsst.obs.nickel.translator import NickelTranslator


class TestNickelTranslator(unittest.TestCase):

    def setUp(self):
        self.header = {
            "TELESCOP": "Nickel 1m",
            "INSTRUME": "Nickel Direct Camera",
            "CAMERA": "NickelC2",
            "OBSTYPE": "OBJECT",
            "OBJECT": "NGC_3982",
            "PROGRAM": "NEWCAM",
            "FILTNAM": "B",
            "EXPTIME": 120.0,
            "OBSNUM": 1032,
            "TEMPDET": -109.7,
            "AIRMASS": 1.28,
            "DATE": "2024-06-25T05:17:49.85",
            "RA": "11:56:28.09",
            "DEC": "55:07:31.0",
            "RADESYSS": "FK5"
        }
        self.translator = NickelTranslator(self.header)

    def test_can_translate(self):
        self.assertTrue(NickelTranslator.can_translate(self.header))

    def test_datetime_begin(self):
        dt = self.translator.to_datetime_begin()
        self.assertIsInstance(dt, Time)
        self.assertAlmostEqual(dt.mjd, Time("2024-06-25T05:17:49.85", format="isot", scale="utc").mjd, places=6)

    def test_temperature(self):
        temp = self.translator.to_temperature()
        self.assertAlmostEqual(temp.to_value(u.K), 163.45, places=2)

    def test_tracking_radec(self):
        coord = self.translator.to_tracking_radec()
        self.assertIsInstance(coord, SkyCoord)
        self.assertAlmostEqual(coord.ra.hour, 11 + 56/60 + 28.09/3600, places=4)
        self.assertAlmostEqual(coord.dec.degree, 55 + 7/60 + 31.0/3600, places=4)

    def test_exposure_id(self):
        self.assertEqual(self.translator.to_exposure_id(), 1032)

    def test_physical_filter(self):
        self.assertEqual(self.translator.to_physical_filter(), "B")

    def test_airmass(self):
        self.assertAlmostEqual(self.translator.to_boresight_airmass(), 1.28)

    def test_object(self):
        self.assertEqual(self.translator.to_object(), "NGC_3982")

    def test_observation_type(self):
        self.assertEqual(self.translator.to_observation_type(), "object")


if __name__ == "__main__":
    unittest.main()
