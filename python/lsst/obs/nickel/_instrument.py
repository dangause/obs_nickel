# python/lsst/obs/nickel/_instrument.py

from lsst.obs.base import Instrument
from .camera import makeCamera
from .rawFormatter import NickelRawFormatter

class Nickel(Instrument):
    """Instrument class for the Nickel telescope."""

    def __init__(self):
        super().__init__(makeCamera())

    def getRawFormatter(self, dataId):
        """Return the formatter for raw data."""
        return NickelRawFormatter
