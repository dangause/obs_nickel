from astro_metadata_translator import FitsTranslator

class NickelTranslator(FitsTranslator):
    name = "nickel"

    @classmethod
    def can_translate(cls, header):
        return header.get("INSTRUME", "").lower().startswith("nickel")

    def to_exposure_id(self, header):       return int(header["IDNUM"])
    def to_detector_num(self, header):     return int(header.get("CAMERAID", 0))
    def to_exposure_time(self, header):    return float(header["EXPTIME"])
    def to_datetime_begin(self, header):   from astropy.time import Time
                                          import astropy.units as u
                                          t = Time(header["DATE"], format="isot", scale="utc")
                                          return t + int(header["TSEC"])*u.s + int(header["TUSEC"])*u.us
    def to_datetime_end(self, header):     return self.to_datetime_begin(header) + self.to_exposure_time(header)*u.s
    def to_physical_filter(self, header):  return header.get("FILTNAM", "").strip()
    def to_observation_type(self, header): return header.get("OBSTYPE", "").lower()
    def to_object(self, header):           return header.get("OBJECT", "").strip()
