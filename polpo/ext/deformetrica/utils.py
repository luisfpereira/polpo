from core import default
from support import utilities


def move_data(
    *arrays,
    gpu_mode=default.gpu_mode,
    tensor_scalar_type=default.tensor_scalar_type,
):
    device, _ = utilities.get_best_device(gpu_mode)

    return [
        utilities.move_data(array, dtype=tensor_scalar_type, device=device)
        for array in arrays
    ]
