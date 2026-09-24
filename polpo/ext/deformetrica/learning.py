"""Deterministic atlas and regression estimation utilities."""

from deformetrica.api.deformetrica import Deformetrica

import polpo.utils as putils
from polpo.ext.deformetrica.io import find_template


def estimate_deterministic_atlas(
    targets,
    output_dir,
    config,
    source=None,
):
    r"""Estimate a deterministic atlas from a collection of shapes.

    Estimate an average shape and the deformations from this average to each
    sample in the collection.

    The problem can be viewed as a regularized analogue of Fréchet mean
    estimation. It minimizes an objective of the form

    .. math::

    C(c, \mu)
    =
    \sum_i
    \left[
        \frac{1}{\alpha^2}
        d\left(q_i, \phi_1^{c,\mu_i}(\bar q)\right)^2
        +
        \left\|v_0^{c_i,\mu_i}\right\|_K^2
    \right].

    Here, :math:\bar q is the estimated template and :math:q_i is the
    :math:i-th target shape.
    The control points :math:`c` are shared across subjects, while the momenta
    :math:`\mu_i` parameterize the deformation from the template :math:\bar q to subject
    :math:`i`.
    The associated velocity field is

    .. math::

    v_t(x)
    =
    \sum_{k=1}^{N_c}
    K\left(x, c_k^{(t)}\right)\mu_k^{(t)},

    where :math:K is the deformation kernel, and
    :math:\phi_1^{c_i,\mu_i} denotes the corresponding flow evaluated at
    time 1.

    The discrepancy :math:d measures the mismatch between shapes, for
    example using landmark, current, or varifold attachment. The parameter
    :math:\alpha controls the tradeoff between data attachment and
    deformation regularity.

    Parameters
    ----------
    targets : sequence or dict
        Paths to the target meshes. If a dictionary is provided, its keys are
        used as subject identifiers.
    output_dir : path-like
        Directory where results are written.
    config : DeterministicAtlasConfig
        Atlas estimation configuration.
    source : path-like
        Path to the initial template mesh. If not provided, the first target
        is used.

    Returns
    -------
    atlas : path-like
        Path to the estimated atlas.
    """
    if source is None:
        source = putils.get_first(targets)

    template_specifications = config.build_template_specifications(source)

    if isinstance(targets, dict):
        target_paths = targets.values()
        subject_ids = [str(key) for key in targets]
    else:
        target_paths = targets
        subject_ids = [str(index) for index in range(len(targets))]

    dataset_specifications = {
        "dataset_filenames": [[{"shape": target}] for target in target_paths],
        "visit_ages": None,
        "subject_ids": subject_ids,
    }

    deformetrica = Deformetrica(
        output_dir,
        verbosity=config.verbosity,
    )
    deformetrica.estimate_deterministic_atlas(
        template_specifications=template_specifications,
        dataset_specifications=dataset_specifications,
        model_options=config.build_model_options(),
        estimator_options=config.build_estimator_options(),
    )

    return find_template(output_dir)


def estimate_geodesic_regression(
    source,
    targets,
    times,
    output_dir,
    config,
    subject_id="patient",
):
    r"""Estimate geodesic regression.

    Estimate a geodesic trajectory fitting surface observations indexed by time.

    Parameters
    ----------
    source : path-like
        Path to the source surface mesh.
    targets : sequence of path-like
        Paths to the observed surface meshes.
    times : sequence of float
        Observation times.
    output_dir : path-like
        Directory where results are written.
    config : GeodesicRegressionConfig
        Geodesic-regression configuration.
    subject_id : str
        Subject identifier used in the Deformetrica dataset specification.
    """
    # TODO: expand to multiple subjects?
    # TODO: update subject_id
    template_specifications = config.build_template_specifications(source)

    dataset_specifications = {
        "visit_ages": [times],
        "dataset_filenames": [[{"shape": target} for target in targets]],
        "subject_ids": [str(subject_id)],
    }

    deformetrica = Deformetrica(
        output_dir,
        verbosity=config.verbosity,
    )

    deformetrica.estimate_geodesic_regression(
        template_specifications=template_specifications,
        dataset_specifications=dataset_specifications,
        model_options=config.build_model_options(times),
        estimator_options=config.build_estimator_options(),
    )


def estimate_spline_regression(
    source,
    targets,
    times,
    output_dir,
    config,
    subject_id="patient",
    weights=None,
):
    r"""Estimate spline regression for longitudinal surface observations.

    Estimate a deformation trajectory fitting surface observations indexed by
    time. The spline model augments a geodesic trajectory with time-dependent
    external forces.

    The objective has the form

    .. math::

        C_S(c, \mu, u_t)
        =
        \frac{1}{\alpha^2 d}
        \sum_{i=1}^d
        w_i\,
        d\left(
            x_{t_i},
            \phi_{t_i}(x_{t_0})
        \right)^2
        +
        \int_0^1 \|u^{(t)}\|^2 \, dt
        +
        \lambda \|v_0^{c,\mu}\|_K^2,

    where :math:`x_{t_i}` is the observation at time :math:`t_i`,
    :math:`w_i` is its weight, and :math:`d` is the attachment discrepancy.
    The control points :math:`c` and momenta :math:`\mu` define the initial
    velocity field

    .. math::

        v_t(x)
        =
        \sum_{k=1}^{N_c}
        K\left(x, c_k^{(t)}\right)
        \mu_k^{(t)},

    where :math:`K` is the deformation kernel. The external forces
    :math:`u^{(t)}` allow smooth departures from a geodesic trajectory, while
    :math:`\lambda` controls the geodesic regularization through
    ``config.model.geodesic_weight``. When the external forces are zero, the
    model reduces to geodesic regression.

    Parameters
    ----------
    source : path-like
        Path to the source surface mesh.
    targets : sequence of path-like
        Paths to the observed surface meshes.
    times : sequence of float
        Observation times corresponding to ``targets``.
    output_dir : path-like
        Directory where Deformetrica outputs are written.
    config : SplineRegressionConfig
        Spline-regression configuration.
    subject_id : str
        Subject identifier used in the Deformetrica dataset specification.
    weights : array-like
        Weights applied to the observations in the regression objective. If
        not provided, all observations receive unit weight.

    Returns
    -------
    None
        The regression results are written to ``output_dir``.
    """
    # TODO: expand to multiple subjects?
    # TODO: update subject_id
    template_specifications = config.build_template_specifications(source)

    dataset_specifications = {
        "visit_ages": [times],
        "dataset_filenames": [[{"shape": target} for target in targets]],
        "subject_ids": [str(subject_id)],
    }

    deformetrica = Deformetrica(
        output_dir,
        verbosity=config.verbosity,
    )

    deformetrica.estimate_spline_regression(
        template_specifications=template_specifications,
        dataset_specifications=dataset_specifications,
        model_options=config.build_model_options(
            times,
            weights=weights,
        ),
        estimator_options=config.build_estimator_options(),
    )
