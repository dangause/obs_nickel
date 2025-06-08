from lsst.obs.base import FilterDefinition, FilterDefinitionCollection

NICKEL_FILTER_DEFINITIONS = FilterDefinitionCollection(
    FilterDefinition(physical_filter="B", band="B"),
    FilterDefinition(physical_filter="V", band="V"),
    FilterDefinition(physical_filter="R", band="R"),
    FilterDefinition(physical_filter="I", band="I"),
)