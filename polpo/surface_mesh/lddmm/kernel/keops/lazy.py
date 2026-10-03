import geomstats.backend as gs

if gs.__name__.endswith("pytorch"):
    from pykeops.torch import Vi, Vj
else:
    from pykeops.numpy import Vi, Vj

from geomstats.varifold.keops.lazy import GaussianKernel  # noqa:F401

# TODO: some of this can go to geomstats

# TODO: control backend better
BACKEND = "CPU"


def GaussianKernelGradient(sigma=1.0, init_index=0, dim=3):
    r"""Gradient of the Gaussian kernel with respect to its first argument.

    .. math::

        \nabla_x K(x, y)
        =
        -\frac{2}{\sigma^2}(x-y)K(x,y).
    """
    x, y = Vi(init_index, dim), Vj(init_index + 1, dim)

    kernel = GaussianKernel(
        sigma=sigma,
        init_index=init_index,
        dim=dim,
    )

    gamma = 1 / (sigma * sigma)
    return -2 * gamma * (x - y) * kernel


class KernelOperator:
    def __init__(self, kernel, kernel_grad, dim=3):
        index = max(
            kernel.new_variable_index(),
            kernel_grad.new_variable_index(),
        )

        weights_b = Vj(index, dim)
        self._kernel_prod = (kernel * weights_b).sum_reduction(axis=1)

        weights_a = Vi(index + 1, dim)
        weights_b = Vj(index + 2, dim)

        weight_inner_prod = (weights_a * weights_b).sum(-1)
        self._kernel_grad_prod = (weight_inner_prod * kernel_grad).sum_reduction(axis=1)

    def kernel_prod(self, points_a, points_b, weights_b):
        return self._kernel_prod(
            points_a,
            points_b,
            weights_b,
            backend=BACKEND,
        )

    def kernel_grad_prod(
        self,
        points_a,
        points_b,
        weights_a,
        weights_b,
    ):
        return self._kernel_grad_prod(
            points_a,
            points_b,
            weights_a,
            weights_b,
            backend=BACKEND,
        )
