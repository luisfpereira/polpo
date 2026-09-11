"""Encoding of hippocampus subfields.

https://www.nitrc.org/projects/ashs

Check out [PTC2024]_ for more details.

Structure_ID names, numbers, and colors:
----------------------------------------
1   255    0    0        1  1  1    "CA1"
2     0  255    0        1  1  1    "CA2+3"
3     0    0  255        1  1  1    "DG"
4   255  255    0        1  1  1    "ERC"
5     0  255  255        1  1  1    "PHC"
6   255    0  255        1  1  1    "PRC"
7    80  179  221        1  1  1    "SUB"
8   255  215    0        1  1  1    "AntHipp"
9   184  115   51        1  1  1    "PostHipp"
2, 6 are expected to grow in volume with progesterone
4, 5 are expected to shrink in volume with progesterone

References
----------
.. [PTC2024] L. Pritschet, C.M. Taylor, et al., 2024. Neuroanatomical changes observed
    over the course of a human pregnancy. Nat Neurosci 27, 2253–2260.
    https://doi.org/10.1038/s41593-024-01741-0
"""

ASHS_ID_TO_NAME = {
    1: "CA1",
    2: "CA2+3",
    3: "DG",
    4: "ERC",
    5: "PHC",
    6: "PRC",
    7: "SUB",
    8: "AntHipp",
    9: "PostHipp",
}

NAME_TO_ASHS_ID = {v: k for k, v in ASHS_ID_TO_NAME.items()}

ASHS_NAME_TO_COLOR = {
    "CA1": (255, 0, 0, 255),
    "CA2+3": (0, 255, 0, 255),
    "DG": (0, 0, 255, 255),
    "ERC": (255, 255, 0, 255),
    "PHC": (0, 255, 255, 255),
    "PRC": (255, 0, 255, 255),
    "SUB": (80, 179, 221, 255),
    "AntHipp": (255, 215, 0, 255),
    "PostHipp": (184, 115, 51, 255),
}
