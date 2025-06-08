from lsst.obs.base import FilterDefinition, FilterDefinitionCollection

NICKEL_FILTER_DEFINITIONS = FilterDefinitionCollection(
    FilterDefinition("Nickel-U", band="u"),
    FilterDefinition("Nickel-B", band="b"),
    FilterDefinition("Nickel-V", band="v"),
    FilterDefinition("Nickel-R", band="r"),
    FilterDefinition("Nickel-I", band="i"),
)
