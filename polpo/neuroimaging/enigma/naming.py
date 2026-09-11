from polpo.freesurfer.naming import aseg_id_to_name, name_to_aseg_id  # noqa: F401
from polpo.neuroi.naming import _expand_subcortical_structs

SUBCORTICAL_STRUCTS = {
    # https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/first.html
    # same as in neuroi.naming, but keeping it independent
    "Thal",
    "Caud",
    "Puta",
    "Pall",
    "Hipp",
    "Amyg",
    "Accu",
}


def get_all_subcortical_structs(prefixed=True, only_bilateral=False, interleave=False):
    return _expand_subcortical_structs(
        SUBCORTICAL_STRUCTS,
        prefixed=prefixed,
        only_bilateral=only_bilateral,
        interleave=interleave,
    )
