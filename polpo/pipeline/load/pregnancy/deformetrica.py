import polpo.preprocessing.dict as ppdict
from polpo.mesh.surface import PvSurface
from polpo.preprocessing import CachablePipeline, Map
from polpo.preprocessing.load.pregnancy.random import (
    TwoRandomMeshesPipe as _TwoRandomMeshesPipe,
)
from polpo.preprocessing.mesh.io import PvReader, PvWriter


class TwoRandomMaternalMeshesPipe(CachablePipeline):
    def __init__(
        self,
        outputs_dir,
        mesh_names=("mesh_a", "mesh_b"),
        struct_name="L_Hipp",
        align=True,
        target_reduction=0.6,
        as_pv_surface=True,
        use_cache=True,
    ):
        meshes_writer = (
            lambda dir_and_meshes: [
                (f"{dir_and_meshes[0]}/{name}", mesh)
                for mesh, name in zip(dir_and_meshes[1], mesh_names)
            ]
        ) + Map(PvWriter(ext="vtk"))

        vtk_loader = ppdict.HashWithIncoming(
            step=Map(PvReader() + (PvSurface if as_pv_surface else None)),
        )

        cache_loader = (
            lambda outputs_dir: [outputs_dir / f"{name}.vtk" for name in mesh_names]
        ) + vtk_loader

        pregnancy_loader = (
            _TwoRandomMeshesPipe(
                struct_name=struct_name,
                target_reduction=target_reduction,
                align=align,
                as_pv_surface=False,
            )
            + (lambda data: (outputs_dir, data))
            + meshes_writer
            + vtk_loader
        )

        super().__init__(
            cache_dir=outputs_dir,
            no_cache_pipe=pregnancy_loader,
            cache_pipe=cache_loader,
            to_cache_pipe=lambda x: x,  # TODO: handle this better
            use_cache=use_cache,
        )


def get_two_random_meshes(
    outputs_dir,
    mesh_names=("mesh_a", "mesh_b"),
    struct_name="L_Hipp",
    align=True,
    target_reduction=0.6,
    as_pv_surface=True,
    use_cache=True,
):
    pipe = TwoRandomMaternalMeshesPipe(
        outputs_dir,
        mesh_names=mesh_names,
        struct_name=struct_name,
        align=align,
        target_reduction=target_reduction,
        as_pv_surface=as_pv_surface,
        use_cache=use_cache,
    )
    return pipe()
