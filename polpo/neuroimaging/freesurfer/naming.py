from polpo.neuroi.naming import (  # noqa: F401
    get_all_subcortical_structs,
    get_subcortical_struct_long_name,
)

ASEG_ID_TO_NAME = {
    16: "BrStem",
    10: "L_Thal",
    49: "R_Thal",
    11: "L_Caud",
    50: "R_Caud",
    12: "L_Puta",
    51: "R_Puta",
    13: "L_Pall",
    52: "R_Pall",
    17: "L_Hipp",
    53: "R_Hipp",
    18: "L_Amyg",
    54: "R_Amyg",
    26: "L_Accu",
    58: "R_Accu",
}

NAME_TO_FREE_NAME = {
    "BrStem": "Brain-Stem",
    "L_Thal": "Left-Thalamus-Proper",
    "R_Thal": "Right-Thalamus-Proper",
    "L_Caud": "Left-Caudate",
    "R_Caud": "Right-Caudate",
    "L_Puta": "Left-Putamen",
    "R_Puta": "Right-Putamen",
    "L_Pall": "Left-Pallidum",
    "R_Pall": "Right-Pallidum",
    "L_Hipp": "Left-Hippocampus",
    "R_Hipp": "Right-Hippocampus",
    "L_Amyg": "Left-Amygdala",
    "R_Amyg": "Right-Amygdala",
    "L_Accu": "Left-Accumbens-area",
    "R_Accu": "Right-Accumbens-area",
}


NAME_TO_ASEG_ID = {v: k for k, v in ASEG_ID_TO_NAME.items()}
FREE_NAME_TO_NAME = {v: k for k, v in NAME_TO_FREE_NAME.items()}


def name_to_aseg_id(struct):
    return NAME_TO_ASEG_ID[struct]


def aseg_id_to_name(struct):
    return ASEG_ID_TO_NAME[struct]
