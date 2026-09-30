from polpo.dispatch import prefixdispatch


@prefixdispatch("derivative")
def select_mesh_paths(path, struct_subset=None, derivative="enigma"):
    raise ValueError(f"Unknown derivative: {derivative}")


@prefixdispatch("derivative")
def read_geometry(path, derivative="enigma"):
    raise ValueError(f"Unknown derivative: {derivative}")
