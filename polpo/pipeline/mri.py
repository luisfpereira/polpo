import numpy as np

from polpo.freesurfer.mri import MriImageLoader  # noqa: F401

from .base import PreprocessingStep


class LocalToTemplateTransform(PreprocessingStep):
    """Get transform from local to template system.

    Parameters
    ----------
    template_affine : array-like
        Target affine matrix.
    """

    def __init__(self, template_affine=None):
        super().__init__()
        self.template_affine = template_affine

        if self.template_affine is not None:
            self._template_affine_inv = np.linalg.inv(self.template_affine)

    def __call__(self, data):
        """Apply step.

        Parameters
        ----------
        data : array-like or tuple[array-like; 2]
            (local, template) affine matrices.

        Returns
        -------
        transformation_matrix : np.array
            Transformation matrix.
        """
        if isinstance(data, (list, tuple)):
            local_affine, template_affine = data
            template_affine_inv = np.linalg.inv(template_affine)

        else:
            if self.template_affine is None:
                raise ValueError("Template affine is undefined.")
            template_affine_inv = self._template_affine_inv
            local_affine = data

        return template_affine_inv @ local_affine
