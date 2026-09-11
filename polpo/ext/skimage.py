import skimage

from polpo.preprocessing.base import PreprocessingStep
from polpo.utils import params_to_kwargs


class MarchingCubes(PreprocessingStep):
    """
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes
    """

    def __init__(
        self,
        level=0,
        step_size=1,
        allow_degenerate=False,
        method="lewiner",
        return_normals=False,
        return_values=False,
    ):
        super().__init__()
        self.level = level
        self.step_size = step_size
        self.allow_degenerate = allow_degenerate
        self.method = method
        self.return_normals = return_normals
        self.return_values = return_values

    def __call__(self, data):
        if isinstance(data, (list, tuple)):
            img_fdata, mask = data
        else:
            img_fdata = data
            mask = None

        (
            vertices,
            faces,
            normals,
            values,
        ) = skimage.measure.marching_cubes(
            img_fdata,
            mask=mask,
            **params_to_kwargs(self, ignore=("return_values", "return_normals")),
        )

        out = (vertices, faces)

        if self.return_normals:
            out = out + (normals,)

        if self.return_values:
            out = out + (values,)

        return out
