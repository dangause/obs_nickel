from lsst.obs.base import MakeRawVisitInfoViaObsInfo

class NickelVisitInfo(MakeRawVisitInfoViaObsInfo):
    metadataTranslator = NickelTranslator
