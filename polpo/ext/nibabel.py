import nibabel as nib
import numpy as np


def load_image(filename, as_nib=False, return_affine=False):
    """Load a volumetric neuroimaging image using NiBabel.

    Parameters
    ----------
    filename : path-like
        Path to the image file.
    as_nib : bool
        Whether to return the NiBabel image object instead of its data.
    return_affine : bool
        Whether to return the affine transformation alongside the image data.
        Ignored if `as_nib` is True.

    Returns
    -------
    img : nibabel.spatialimages.SpatialImage
        Loaded image if `as_nib` is True.
    data : ndarray
        Image data if `as_nib` is False.
    affine : ndarray, shape=[4, 4]
        Affine transformation, returned alongside `data` if
        `return_affine` is True and `as_nib` is False.
    """
    img = nib.load(filename)

    if as_nib:
        return img

    data = img.get_fdata()
    return (data, img.affine) if return_affine else data


def compute_label_volumes(path, labels, encoding=None):
    """Compute label volumes from a segmentation image.

    Parameters
    ----------
    path : path-like
        Path to the segmentation image.
    labels : iterable
        Labels identifying the structures.
    encoding : callable or None
        Function mapping labels to image values.

    Returns
    -------
    volumes : dict
        Mapping from labels to volumes in cubic millimeters.
    """
    if encoding is None:
        encoding = lambda x: x

    img = nib.load(path)
    data = np.asanyarray(img.dataobj)

    voxel_volume = abs(np.linalg.det(img.affine[:3, :3]))

    return {
        label: np.count_nonzero(data == encoding(label)) * voxel_volume
        for label in labels
    }
