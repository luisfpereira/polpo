from polpo.surface_mesh.deformetrica import LddmmMetric


def instantiate_lddmm_metric(
    dirname,
    kernel_width,
    attachment_kernel_width,
    regularization,
    max_iter,
    tol,
):
    """Instantiate a configured LDDMM metric.

    Parameters
    ----------
    dirname : path-like
        Directory used to store LDDMM artifacts.
    kernel_width : float
        Width of the deformation kernel.
    attachment_kernel_width : float
        Width of the varifold attachment kernel.
    regularization : float
        Regularization parameter controlling the attachment term.
    max_iter : int
        Maximum number of registration optimization iterations.
    tol : float
        Convergence tolerance for registration optimization.

    Returns
    -------
    metric : LddmmMetric
        Configured LDDMM metric.
    """
    metric = LddmmMetric(
        dirname,
        kernel_width=kernel_width,
        attachment_metric="varifold",
        attachment_kernel_width=attachment_kernel_width,
    )

    metric.config.set_attachment(
        noise_std=regularization,
    )
    metric.config.set_optimization(
        max_iter=max_iter,
        tol=tol,
    )

    return metric
