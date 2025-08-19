# python/lsst/obs/nickel/rawFormatter.py

__all__ = ["NickelRawFormatter"]

from astro_metadata_translator import FitsTranslator
from lsst.obs.base import FitsRawFormatterBase
from .translator import NickelTranslator
from .nickelFilters import NICKEL_FILTER_DEFINITIONS


class NickelRawFormatter(FitsRawFormatterBase):
    """Raw data formatter for the Nickel telescope."""

    translatorClass = NickelTranslator
    filterDefinitions = NICKEL_FILTER_DEFINITIONS

    def getDetector(self, dataId):
        from ._instrument import Nickel
        camera = Nickel().getCamera()
        if isinstance(dataId, int):
            detector_id = dataId
        elif isinstance(dataId, dict):
            detector_id = dataId["detector"]
        else:
            raise TypeError(f"Unexpected dataId format: {dataId}")
        return camera[detector_id]

