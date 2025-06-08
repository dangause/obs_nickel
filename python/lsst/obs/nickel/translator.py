from astro_metadata_translator import FitsTranslator
from astropy.time import Time

__all__ = ["NickelTranslator"]

class NickelTranslator(FitsTranslator):
    """Metadata translator for the Nickel telescope at Lick Observatory."""

    name = "Nickel"
    supported_instruments = {"Nickel"}  # matches getName()
    supported_telescopes = {"Nickel 1m"}

    def can_translate(self, header, filename=None):
        #TODO: fix this!
        # Fallback to other header keys in case TELESCOP is missing
        instrume = header.get("INSTRUME", "").lower()
        camera = header.get("CAMERA", "").lower()
        return "nickel" in instrume or "nickel" in camera


    def to_observation_id(self):
        # OBSNUM exists
        return str(self._header.get("OBSNUM", self._generic_observation_id()))

    def to_datetime_begin(self):
        # DATE field exists and is ISO UTC — astropy handles it
        return self._parse_time(self._header.get("DATE"))

    def to_exposure_time(self):
        return float(self._header.get("EXPTIME"))

    def to_detector_name(self):
        # No DETECTOR keyword — return hardcoded name from camera.yaml
        return "CCD0"

    def to_instrument(self):
        return "Nickel"

    def to_physical_filter(self):
        # Use FILTNAM (e.g., 'B', 'V', 'I')
        return self._header.get("FILTNAM")

    def to_telescope(self):
        return "Nickel 1m"

    def _parse_time(self, value):
        if value:
            return Time(value, format="isot", scale="utc")
        raise ValueError("DATE (DATE-OBS) not found or malformed")
