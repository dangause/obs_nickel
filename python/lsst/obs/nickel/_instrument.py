from lsst.obs.base._instrument import Instrument
from lsst.obs.base.yamlCamera import makeCamera
from lsst.obs.base import FilterDefinition, FilterDefinitionCollection
from lsst.utils.introspection import get_full_type_name
from .nickelFilters import NICKEL_FILTER_DEFINITIONS
from .rawFormatter import NickelRawFormatter
from .translator import NickelTranslator
import os

__all__ = ["Nickel"]


class Nickel(Instrument):
    """Instrument class for the Nickel telescope at Lick Observatory."""

    translatorClass = NickelTranslator

    def __init__(self, collection_prefix=None):
        super().__init__(collection_prefix=collection_prefix)

    def getCamera(self):
        # Locate camera.yaml relative to this file
        cameraYamlPath = os.path.join(os.path.dirname(__file__), "camera.yaml")
        return makeCamera(cameraYamlPath)

    def getName(self):
        return "Nickel"

    def register(self, registry, update=False):
        camera = self.getCamera()
        with registry.transaction():
            registry.syncDimensionData("instrument", {
                "name": self.getName(),
                "class_name": get_full_type_name(type(self)),
                "detector_max": len(camera),
                "visit_max": 2**25,
                "exposure_max": 2**25,
            }, update=update)

            for det in camera:
                registry.syncDimensionData("detector", {
                    "instrument": self.getName(),
                    "id": det.getId(),
                    "full_name": det.getName(),
                    "name_in_raft": det.getName(),  # adjust as needed
                    "raft": det.getName(),          # adjust as needed
                    "purpose": str(det.getType()).split(".")[-1],
                }, update=update)

            self._registerFilters(registry, update=update)


    @property
    def filterDefinitions(self):
        return NICKEL_FILTER_DEFINITIONS

    def getRawFormatter(self, dataId):
        return NickelRawFormatter
