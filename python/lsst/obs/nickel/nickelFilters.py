from lsst.obs.base import FilterDefinition, FilterDefinitionCollection

NICKEL_FILTER_DEFINITIONS = FilterDefinitionCollection(
    FilterDefinition(physical_filter="B", band="b"),
    FilterDefinition(physical_filter="V", band="v"),
    FilterDefinition(physical_filter="R", band="r"),
    FilterDefinition(physical_filter="I", band="i"),
)