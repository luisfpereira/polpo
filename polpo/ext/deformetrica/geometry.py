import logging

from core.model_tools.deformations.exponential import Exponential  # noqa: F401
from launch.compute_parallel_transport import (
    compute_parallel_transport,
    compute_pole_ladder,
)
from launch.compute_shooting import compute_shooting

logger = logging.getLogger(__name__)


def shoot(
    source,
    control_points,
    momenta,
    output_dir,
    config,
):
    """Shoot a geodesic deformation."""
    template_specifications = {
        "shape": {
            "deformable_object_type": "SurfaceMesh",
            "noise_std": -1,
            "filename": source,
        }
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    compute_shooting(
        template_specifications,
        initial_control_points=control_points,
        initial_momenta=momenta,
        output_dir=output_dir,
        **config.to_kwargs(),
    )


def _parallel_transport_pole_ladder(
    control_points,
    momenta,
    momenta_to_transport,
    output_dir,
    config,
    control_points_to_transport=None,
):
    """Parallel transport momenta using the pole-ladder scheme.

    The reference geodesic is defined by ``control_points`` and ``momenta``.
    The momentum ``momenta_to_transport`` is transported along this geodesic
    using Deformetrica's pole-ladder implementation.

    Parameters
    ----------
    control_points : path-like
        Path to the initial control points defining the reference geodesic.
    momenta : path-like
        Path to the initial momenta defining the reference geodesic.
    momenta_to_transport : path-like
        Path to the initial momenta to transport.
    output_dir : path-like
        Directory where Deformetrica outputs are written.
    config : ParallelTransportConfig
        Parallel transport configuration.
    control_points_to_transport : path-like
        Path to the control points associated with the momenta to transport.

    Returns
    -------
    transported_control_points : array-like
        Control points at the end of the transport.
    transported_momenta : array-like
        Transported momenta.
    """
    # NB: returns only the final transported quantities

    output_dir.mkdir(parents=True, exist_ok=True)

    transported_cp, transported_mom = compute_pole_ladder(
        initial_control_points=control_points,
        initial_momenta=momenta,
        initial_momenta_to_transport=momenta_to_transport,
        initial_control_points_to_transport=control_points_to_transport,
        output_dir=output_dir,
        tmin=0,
        tmax=1,
        **config.to_kwargs(),
    )

    return transported_cp, transported_mom


def _parallel_transport_fanning(
    source,
    control_points,
    momenta,
    momenta_to_transport,
    output_dir,
    config,
    control_points_to_transport=None,
):
    """Parallel transport momenta using the fanning scheme.

    The reference geodesic is defined by ``control_points`` and ``momenta``.
    The momentum ``momenta_to_transport`` is transported along this geodesic
    using Deformetrica's parallel transport implementation. The source surface
    is flowed along the resulting deformation.

    Parameters
    ----------
    source : path-like
        Path to the source surface.
    control_points : path-like
        Path to the initial control points defining the reference geodesic.
    momenta : path-like
        Path to the initial momenta defining the reference geodesic.
    momenta_to_transport : path-like
        Path to the initial momenta to transport.
    output_dir : pathlib.Path
        Directory where Deformetrica outputs are written.
    config : ParallelTransportConfig
        Parallel transport configuration.
    control_points_to_transport : str or pathlib.Path
        Path to the control points associated with the momenta to transport.

    Returns
    -------
    result
        Output returned by Deformetrica's fanning parallel transport routine.
    """
    template_specifications = {
        "shape": {
            "deformable_object_type": "SurfaceMesh",
            "noise_std": -1,  # not used
            "filename": source,
        }
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    return compute_parallel_transport(
        template_specifications,
        output_dir=output_dir,
        initial_control_points=control_points,
        initial_momenta=momenta,
        initial_momenta_to_transport=momenta_to_transport,
        initial_control_points_to_transport=control_points_to_transport,
        tmin=0,
        tmax=1,
        **config.to_kwargs(),
    )


def parallel_transport(
    control_points,
    momenta,
    momenta_to_transport,
    output_dir,
    config,
    source=None,
    control_points_to_transport=None,
):
    """Parallel transport momenta along an LDDMM geodesic.

    The transport method is selected from ``config``. Pole ladder transports
    the momenta directly, while fanning additionally requires a source surface.

    Parameters
    ----------
    control_points : path-like
        Path to the initial control points defining the reference geodesic.
    momenta : path-like
        Path to the initial momenta defining the reference geodesic.
    momenta_to_transport : path-like
        Path to the initial momenta to transport.
    output_dir : path-like
        Directory where Deformetrica outputs are written.
    config : ParallelTransportConfig
        Configuration defining the transport method and its parameters.
    source : path-like
        Path to the source surface required by the fanning scheme.
    control_points_to_transport : path-like
        Path to the control points associated with the momenta to transport.

    Returns
    -------
    result
        Result produced by the selected parallel transport method.

    Raises
    ------
    ValueError
        If fanning transport is selected without a source surface.
    """
    if config.method == "pole_ladder":
        if source is not None:
            logger.warning("source is ignored when pole ladder is used")

        return _parallel_transport_pole_ladder(
            control_points,
            momenta,
            momenta_to_transport,
            output_dir,
            config,
            control_points_to_transport=control_points_to_transport,
        )

    if source is None:
        raise ValueError("source needs to be defined to use the fanning scheme.")

    return _parallel_transport_fanning(
        source,
        control_points,
        momenta,
        momenta_to_transport,
        output_dir,
        config,
        control_points_to_transport=control_points_to_transport,
    )
