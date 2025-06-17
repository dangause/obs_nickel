from __future__ import annotations

__all__ = ("NickelTranslator",)

import logging
from typing import Any

import astropy.units as u
import astropy.time
from astropy.coordinates import Angle, EarthLocation, SkyCoord
from astro_metadata_translator.translators.fits import FitsTranslator
from astro_metadata_translator.translators.helpers import tracking_from_degree_headers
from astro_metadata_translator.translator import cache_translation

log = logging.getLogger(__name__)

_NICKEL_LOCATION = EarthLocation.from_geodetic(lat=37.3414, lon=-121.6429, height=1293 * u.m)


class NickelTranslator(FitsTranslator):
    """Metadata translator for the Nickel telescope at Lick Observatory."""

    name = "Nickel"
    supported_instrument = {"Nickel"}

    _const_map = {
        "boresight_rotation_angle": Angle(0.0 * u.deg),
        "boresight_rotation_coord": "sky",
    }

    _trivial_map: dict[str, str | tuple[str, dict[str, Any]]] = {
        "exposure_time": ("EXPTIME", {"unit": u.s, "default": 0.0 * u.s}),
        "dark_time": ("EXPTIME", {"unit": u.s, "default": 0.0 * u.s}),
        "boresight_airmass": ("AIRMASS", {"default": float("nan")}),
        "observation_id": ("OBSNUM", {"default": "0"}),
        "object": ("OBJECT", {"default": "UNKNOWN"}),
        "telescope": ("TELESCOP", {"default": "Nickel 1m"}),
        "science_program": ("PROGRAM", {"default": "unknown"}),
        "relative_humidity": ("HUMIDITY", {"default": 0.0}),

    }

    _observing_day_offset = astropy.time.TimeDelta(12 * 3600, format="sec", scale="tai")

    @classmethod
    def can_translate(cls, header, filename=None):
        val = header.get("INSTRUME", "").strip().lower()
        result = "nickel" in val
        return result

    @cache_translation
    def to_instrument(self) -> str:
        return "Nickel"

    @cache_translation
    def to_exposure_id(self) -> int:
        return int(self._header.get("OBSNUM", 0))

    @cache_translation
    def to_visit_id(self) -> int:
        return self.to_exposure_id()

    @cache_translation
    def to_datetime_begin(self) -> astropy.time.Time:
        return self._from_fits_date("DATE-BEG", scale="utc")

    @cache_translation
    def to_datetime_end(self) -> astropy.time.Time:
        return self._from_fits_date("DATE-END", scale="utc")


    @cache_translation
    def to_observation_type(self) -> str:
        object_str = self._header.get("OBJECT", "").strip().lower()
        obstype = self._header.get("OBSTYPE", "").strip().lower()

        # Use OBSTYPE first if it's present
        if obstype == "dark":
            return "bias" if "bias" in object_str else "dark"
        if obstype == "object":
            if "flat" in object_str:
                return "flat"
            if "focus" in object_str:
                return "focus"
            return "science"

        # Fallback heuristics
        if "bias" in object_str:
            return "bias"
        if "flat" in object_str:
            return "flat"
        if "focus" in object_str:
            return "focus"

        return "science"


    @cache_translation
    def to_physical_filter(self) -> str:
        return str(self._header.get("FILTNAM", "UNKNOWN")).strip()

    @cache_translation
    def to_location(self) -> EarthLocation:
        return _NICKEL_LOCATION

    @cache_translation
    def to_tracking_radec(self) -> SkyCoord:
        ra = self._header.get("RA")
        dec = self._header.get("DEC")
        frame = self._header.get("RADESYSS", "FK5").strip()
        if ra and dec:
            return SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame=frame.lower())
        raise RuntimeError("Missing RA/DEC in header")

    @cache_translation
    def to_temperature(self) -> u.Quantity:
        temp_celsius = self._header.get("TEMPDET", -999.0)
        return (temp_celsius + 273.15) * u.K

    @cache_translation
    def to_detector_num(self) -> int:
        return 0

    @cache_translation
    def to_detector_name(self) -> str:
        return "0"

    @cache_translation
    def to_detector_unique_name(self) -> str:
        return "0"

    @cache_translation
    def to_detector_serial(self) -> str:
        return ""

    @cache_translation
    def to_detector_group(self) -> str:
        return ""

    @cache_translation
    def to_detector_exposure_id(self) -> int:
        return self.to_exposure_id()

    @cache_translation
    def to_focus_z(self) -> u.Quantity:
        return 0.0 * u.m

    @cache_translation
    def to_altaz_begin(self):
        return None

    @cache_translation
    def to_pressure(self):
        return None


