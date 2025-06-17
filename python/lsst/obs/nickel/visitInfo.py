from lsst.obs.base import MakeRawVisitInfoViaObsInfo
from .translators.nickel import NickelTranslator

class NickelVisitInfo(MakeRawVisitInfoViaObsInfo):
    metadataTranslator = NickelTranslator
