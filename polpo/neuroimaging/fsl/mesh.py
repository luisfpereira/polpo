import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.preprocessing import Map
from polpo.preprocessing.path import (
    FileFinder,
    IsFileType,
    PathShortener,
)
from polpo.preprocessing.str import ContainsAny
from polpo.pyvista.io import PvReader

from .naming import get_all_subcortical_structs
from .validation import validate_structs

MeshReader = PvReader


def MeshDatasetLoader(struct_subset=None, mesh_reader=False):
    # pipeline takes dirname

    if mesh_reader is None:
        mesh_reader = MeshReader()
    elif mesh_reader is False:
        mesh_reader = None

    if struct_subset is None:
        struct_subset = get_all_subcortical_structs()

    validate_structs(struct_subset)

    rules = [IsFileType("vtk"), ContainsAny(struct_subset)]
    # e.g. sub-01_ses-01-L_Hipp_first.vtk
    path_to_struct_id = PathShortener() + (
        lambda x: x.split("-")[-1].split(".")[0][:-6]
    )

    return FileFinder(rules=rules, as_list=True) + ppdict.HashWithIncoming(
        key_step=Map(path_to_struct_id),
        step=Map(mesh_reader),
        key_sorter=putils.custom_order(struct_subset),
    )
