import random

import polpo.preprocessing.dict as ppdict
from polpo.jacobs.mesh import MeshDatasetLoader
from polpo.jacobs.utils import get_subject_ids
from polpo.preprocessing import Map, Pipeline
from polpo.preprocessing.mesh.registration import RigidAlignment
from polpo.pyvista.decimation import PvDecimate
from polpo.surface_mesh.core import PvSurface


class TwoRandomMeshesPipe(Pipeline):
    def __init__(
        self,
        struct_name="L_Hipp",
        target_reduction=0.6,
        align=True,
        same_subject=False,
        as_pv_surface=False,
    ):
        # TODO: add possibility of loading carmona meshes?

        # TODO: update to use same_subject
        subject_ids = random.sample(get_subject_ids(include_male=False, sort=True), 2)

        pipe = (
            MeshDatasetLoader(
                subject_subset=subject_ids,
                struct_subset=[struct_name],
                session_subset=None,
                derivative="enigma",
                mesh_reader=None,
            )
            # TODO: split here when getting new dataset
            + ppdict.DictMap(ppdict.ExtractRandomValue())
            + ppdict.ExtractUniqueKey(nested=True)
            + ppdict.DictToValuesList()
            + (
                RigidAlignment(
                    target=lambda x: x[0],
                    known_correspondences=True,
                )
                if align
                else None
            )
            + Map(
                PvDecimate(target_reduction=target_reduction, volume_preservation=True)
                if target_reduction
                else None
            )
            + (Map(PvSurface) if as_pv_surface else None)
        )

        super().__init__(steps=pipe.steps)


def get_two_random_meshes(
    struct_name="L_Hipp",
    target_reduction=0.6,
    align=True,
    same_subject=False,
    as_pv_surface=False,
):
    pipe = TwoRandomMeshesPipe(
        struct_name=struct_name,
        target_reduction=target_reduction,
        align=align,
        same_subject=same_subject,
        as_pv_surface=as_pv_surface,
    )
    return pipe()
