"""Low-level device and tensor utilities for Deformetrica."""

import deformetrica.support.kernels as kernel_factory  # noqa: F401
from deformetrica.core import default
from deformetrica.support import utilities


def move_data(
    *arrays,
    gpu_mode=default.gpu_mode,
    tensor_scalar_type=default.tensor_scalar_type,
):
    """Move arrays to the device selected by Deformetrica.

    Parameters
    ----------
    *arrays : array-like
        Arrays to move.
    gpu_mode : GpuMode
        Deformetrica GPU execution mode.
    tensor_scalar_type
        Scalar tensor type used for the converted arrays.

    Returns
    -------
    arrays : list
        Arrays converted and moved using Deformetrica's data utilities.
    """
    device, _ = utilities.get_best_device(gpu_mode)

    return [
        utilities.move_data(array, dtype=tensor_scalar_type, device=device)
        for array in arrays
    ]
