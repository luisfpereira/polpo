"""Deformetrica registration utilities."""

from api.deformetrica import Deformetrica


def estimate_registration(
    source,
    target,
    output_dir,
    config,
    target_id="target",
):
    r"""Estimate a Deformetrica registration between two surfaces.

    The registration estimates control points :math:`c` and momenta
    :math:`\mu` defining an initial velocity field and corresponding
    deformation flow. The optimization minimizes an objective of the form

    .. math::

        C(c, \mu)
        =
        \frac{1}{\alpha^2}
        d\left(q, \phi_1^{c,\mu}(\bar q)\right)^2
        +
        \left\|v_0^{c,\mu}\right\|_K^2,

    where :math:`\bar q` is the source surface, :math:`q` is the target
    surface, :math:`\phi_1^{c,\mu}` is the deformation at time one,
    :math:`d` is the attachment discrepancy, and :math:`\alpha` controls
    the weight of the attachment term.

    Parameters
    ----------
    source : path-like
        Path to the source surface mesh.
    target : path-like
        Path to the target surface mesh.
    output_dir : path-like
        Directory where Deformetrica registration outputs are written.
    config : RegistrationConfig
        Registration configuration.
    target_id : str
        Identifier assigned to the target surface.

    Returns
    -------
    None
        The registration results are written to ``output_dir``.
    """
    deformetrica = Deformetrica(
        output_dir,
        verbosity=config.verbosity,
    )

    deformetrica.estimate_registration(
        template_specifications=config.template_specifications(source),
        dataset_specifications={
            "visit_ages": [[]],
            "dataset_filenames": [[{"shape": target}]],
            "subject_ids": [target_id],
        },
        model_options=config.model_options(output_dir),
        estimator_options=config.estimator_options(),
    )
