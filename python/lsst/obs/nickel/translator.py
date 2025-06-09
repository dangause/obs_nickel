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
    """Metadata translator for Lick Nickel telescope FITS headers."""

    name = "Nickel"
    supported_instrument = "Nickel Direct Camera"

    _const_map = {
        "boresight_rotation_angle": Angle(0.0 * u.deg),
        "boresight_rotation_coord": "sky",
    }

    _trivial_map: dict[str, str | tuple[str, dict[str, Any]]] = {
        "exposure_time": ("EXPTIME", {"unit": u.s, "default": 0.0 * u.s}),
        "dark_time": ("EXPTIME", {"unit": u.s, "default": 0.0 * u.s}),
        "boresight_airmass": "AIRMASS",
        "observation_id": ("OBSNUM", {"default": "0"}),
        "observation_type": ("OBSTYPE", {"default": "object"}),
        "object": ("OBJECT", {"default": "UNKNOWN"}),
        "telescope": ("TELESCOP", {"default": "Nickel 1m"}),
        "instrument": ("INSTRUME", {"default": "Nickel Direct Camera"}),
        "relative_humidity": ("HUMIDITY", {"default": 0.0}),
        "temperature": ("TEMPDET", {"unit": u.K, "default": 273.15 * u.K}),
        "science_program": ("PROGRAM", {"default": "unknown"}),
    }

    _observing_day_offset = astropy.time.TimeDelta(12 * 3600, format="sec", scale="tai")

    @classmethod
    def can_translate(cls, header, filename=None):
        val = header.get("INSTRUME", "") or header.get("CAMERA", "")
        return "nickel" in val.lower()
    



    @cache_translation
    def to_exposure_id(self) -> int:
        return int(self._header.get("OBSNUM", 0))

    @cache_translation
    def to_visit_id(self) -> int:
        return self.to_exposure_id()

    @cache_translation
    def to_datetime_begin(self) -> astropy.time.Time:
        return self._from_fits_date("DATE", scale="utc")

    @cache_translation
    def to_datetime_end(self) -> astropy.time.Time:
        return self.to_datetime_begin() + self.to_exposure_time()

    @cache_translation
    def to_physical_filter(self) -> str:
        return str(self._header.get("FILTNAM", "UNKNOWN")).strip()

    @cache_translation
    def to_location(self) -> EarthLocation:
        return _NICKEL_LOCATION

    @cache_translation
    def to_tracking_radec(self) -> SkyCoord:
        return tracking_from_degree_headers(self, ("RADESYS",), (("RA", "DEC"),), unit=(u.hourangle, u.deg))

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
