import nibabel as nib

from polpo.preprocessing.base import PreprocessingStep


class MriImageLoader(PreprocessingStep):
    """Load image from .nii/.mgz file.

    Parameters
    ----------
    filename : str
        File to load.
    as_nib : bool
        Whether to return the nibabel object.
    return_affine : bool
        Whether to return the affine transformation.
        Ignore if `as_nib` is True.
    """

    def __init__(self, filename=None, as_nib=False, return_affine=False):
        super().__init__()
        self.filename = filename
        self.as_nib = as_nib
        self.return_affine = return_affine

    def __call__(self, filename=None):
        """Apply step.

        Parameters
        ----------
        filename : str
            File to load.

        Returns
        -------
        img : nibabel.nifti1.Nifti1Image
            If `as_nib` is True.
        img_data : np.array
            If `as_nib` is False.
        affine : nibabel.nifti1.Nifti1Image
            If `as_nib` is False and `affine` is True.
        """
        filename = filename or self.filename

        img = nib.load(filename)
        if self.as_nib:
            return img

        # this is the costly step
        img_data = img.get_fdata()
        if not self.return_affine:
            return img_data

        return img_data, img.affine
