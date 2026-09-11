from polpo.neuroi.mri import LabelSelector as BaseLabelSelector

from .naming import NAME_TO_ASHS_ID


class LabelSelector(BaseLabelSelector):
    def __init__(self, labels=(), binary=True):
        super().__init__(
            labels=labels,
            binary=binary,
            encoding=NAME_TO_ASHS_ID.get,
        )
