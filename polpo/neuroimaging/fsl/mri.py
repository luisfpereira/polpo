from polpo.preprocessing import Contains
from polpo.preprocessing.path import FileFinder, IsFileType


def SubcorticalSegmentationFinder():
    return FileFinder(
        rules=[
            IsFileType("nii.gz"),
            Contains("all_fast_firstseg"),
        ]
    )
