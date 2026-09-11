from polpo.neuroi.mri import LabelSelector as BaseLabelSelector
from polpo.neuroi.mri import LabelSplitter as BaseLabelSplitter
from polpo.nibabel import MriImageLoader  # noqa: F401
from polpo.preprocessing.path import FileFinder

from .naming import name_to_aseg_id


def SubcorticalSegmentationFinder(raw=False):
    # mri/aseg.mgz
    filename = "aseg.mgz" if not raw else "aseg.auto.mgz"
    return FileFinder(rules=lambda x: x == "mri") + FileFinder(
        rules=lambda x: x == filename
    )


class LabelSelector(BaseLabelSelector):
    def __init__(self, labels=(), binary=True):
        super().__init__(
            labels=labels,
            binary=binary,
            encoding=name_to_aseg_id,
        )


class LabelSplitter(BaseLabelSplitter):
    def __init__(self, labels=(), binary=True):
        super().__init__(
            labels=labels,
            binary=binary,
            encoding=name_to_aseg_id,
        )
