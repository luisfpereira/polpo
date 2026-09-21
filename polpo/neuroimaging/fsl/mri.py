from polpo.pipeline import Contains
from polpo.pipeline.path import FileFinder, IsFileType


def SubcorticalSegmentationFinder():
    return FileFinder(
        rules=[
            IsFileType("nii.gz"),
            Contains("all_fast_firstseg"),
        ]
    )
