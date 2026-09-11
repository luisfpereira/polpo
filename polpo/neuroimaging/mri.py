import numpy as np

from polpo.preprocessing import PreprocessingStep


def SubcorticalSegmentationFinder(tool="free"):
    if tool.startswith("fast") or tool.startswith("free"):
        from polpo.freesurfer.mri import (
            SubcorticalSegmentationFinder as FreeSubcorticalSegmentationFinder,
        )

        return FreeSubcorticalSegmentationFinder()

    if tool.startswith("fsl"):
        from polpo.fsl.mri import (
            SubcorticalSegmentationFinder as FslSubcorticalSegmentationFinder,
        )

        return FslSubcorticalSegmentationFinder()

    else:
        raise ValueError(f"Oops, don't know how to handle: {tool}")


class LabelSelector(PreprocessingStep):
    def __init__(self, labels=(), binary=True, encoding=None):
        if encoding is None:
            encoding = lambda x: x

        self.labels = labels
        self.binary = binary
        self.encoding = encoding

    def __call__(self, image):
        ids = [self.encoding(label) for label in self.labels]
        mask = np.isin(image, ids)

        if self.binary:
            return mask

        return np.where(mask, image, 0)


class LabelSplitter(PreprocessingStep):
    def __init__(self, labels=(), binary=True, encoding=None):
        self.selectors = {
            label: LabelSelector([label], binary=binary, encoding=encoding)
            for label in labels
        }

    def __call__(self, image):
        return {label: selector(image) for label, selector in self.selectors.items()}
