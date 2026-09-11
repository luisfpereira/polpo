import polpo.preprocessing.dict as ppdict
from polpo.freesurfer.mri import LabelSelector, LabelSplitter
from polpo.freesurfer.naming import aseg_id_to_name, get_all_subcortical_structs
from polpo.pyvista.conversion import PvFromData
from polpo.pyvista.filter import PvSubsetSplitter
from polpo.skimage import MarchingCubes

from .defaults import DATA_DIR
from .mri import SubcorticalSegmentationsLoader


def MeshLoaderFromMri(
    derivative,
    data_dir=None,
    subject_subset=None,
    session_subset=None,
    struct_subset=None,
    split_before_meshing=False,
    n_jobs=1,
):
    if data_dir is None:
        data_dir = DATA_DIR

    # two basic mri2mesh pipelines

    # subj, session
    segmentations_loader = SubcorticalSegmentationsLoader(
        data_dir=data_dir,
        subject_subset=subject_subset,
        session_subset=session_subset,
        derivative=derivative,
        as_image=True,
    )

    if struct_subset is None:
        struct_subset = get_all_subcortical_structs()

    if split_before_meshing:
        mri2multimesh = LabelSplitter(
            labels=struct_subset,
            binary=False,
        ) + ppdict.DictMap(MarchingCubes() + PvFromData())

    else:
        mri2multimesh = (
            LabelSelector(
                labels=struct_subset,
                binary=False,
            )
            + MarchingCubes(return_values=True)
            + PvFromData(array_names=("labels",))
            + (PvSubsetSplitter() + ppdict.RenameKeys(aseg_id_to_name))
        )

    pipe = segmentations_loader + ppdict.NestedDictMap(mri2multimesh)

    # subj, session, struct
    return pipe
