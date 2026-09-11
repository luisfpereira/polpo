def MeshDatasetLoader(struct_subset=None, derivative="fsl", mesh_reader=False):
    if derivative.startswith("fsl"):
        from polpo.fsl.mesh import MeshDatasetLoader as FslMeshDatasetLoader

        return FslMeshDatasetLoader(
            struct_subset=struct_subset, mesh_reader=mesh_reader
        )

    elif derivative.startswith("enigma"):
        from polpo.enigma.mesh import MeshDatasetLoader as EnigmaMeshDatasetLoader

        return EnigmaMeshDatasetLoader(
            struct_subset=struct_subset, mesh_reader=mesh_reader
        )

    else:
        raise ValueError(f"Unknown derivative: {derivative}")
