import unittest
from lsst.obs.nickel.translator import NickelTranslator

class TestNickelTranslator(unittest.TestCase):
    def test_can_translate(self):
        header = {
            "TELESCOP": "Nickel 1m",
            "FILTER": "V",
            "EXPTIME": 30.0,
            "DATE-OBS": "2024-10-10T05:12:34.567"
        }
        translator = NickelTranslator(header)
        self.assertTrue(translator.can_translate(header))

if __name__ == "__main__":
    unittest.main()
