"""LDDMM geometry for surface shapes."""

import logging
import shutil
from pathlib import Path

import numpy as np

import polpo.ext.deformetrica as pdefo

from .paths import LddmmPaths
from .representations import (
    ControlPoints,
    Momenta,
    TangentVector,
    Velocity,
)
from .results import (
    RegistrationResult,
    ShootResult,
    TransportResult,
)

try:
    # TODO: make it work with no torch
    import torch
except ImportError:
    pass


class LddmmMetric:
    """LDDMM metric for surface shapes.

    Provides logarithm and exponential maps, parallel transport, and RKHS
    norms for tangent vectors represented by control points and momenta.
    Expensive geometric operations are persisted to disk and can be reused
    according to a configurable cache policy.

    Parameters
    ----------
    dir_config : path-like or LddmmPaths
        Configuration of the directories used to store registration,
        shooting, and parallel-transport results.
    kernel_width : float
        Width of the deformation kernel.
    transport_zero_tol : float
        Norm threshold below which a tangent vector is treated as zero during
        parallel transport. Set to ``None`` to disable this behavior.
    attachment_metric : {"landmark", "current", "varifold"}
        Attachment discrepancy used during registration.
    attachment_kernel_width : float
        Width of the attachment kernel for kernel-based discrepancies.
    cache_policy : {"reuse", "overwrite", "validate", "read_only"}
        Policy controlling reuse of persisted geometric computations.
        ``"reuse"`` trusts existing results, ``"overwrite"`` always
        recomputes, ``"validate"`` reuses results whose configuration
        fingerprint matches the current configuration, and ``"read_only"``
        only permits reuse of valid existing results.

    Notes
    -----
    Registration uses frozen control points by default. This gives tangent
    vectors obtained from different registrations a shared finite-dimensional
    RKHS representation, avoiding the reprojection that would otherwise be
    required when comparing or parallel transporting tangent vectors defined
    on different control-point sets.
    """

    def __init__(
        self,
        dir_config,
        kernel_width=10.0,
        transport_zero_tol=1e-3,
        attachment_metric="landmark",
        attachment_kernel_width=None,
        cache_policy="validate",
    ):
        if isinstance(dir_config, Path):
            dir_config = LddmmPaths(dir_config)

        if cache_policy not in {"reuse", "overwrite", "validate", "read_only"}:
            raise ValueError(f"Unknown cache policy {cache_policy!r}.")

        self.dir_config = dir_config

        self.transport_zero_tol = transport_zero_tol

        self.config = (
            pdefo.config.LddmmConfig()
            .set_deformation(
                kernel_width=kernel_width,
            )
            .set_attachment(
                metric=attachment_metric,
                kernel_width=attachment_kernel_width,
            )
            .set_control_points(freeze=True)
        )

        self.cache_policy = cache_policy

    def _can_reuse(self, result, fingerprint):
        """Determine whether a persisted result can be reused.

        The decision follows the current cache policy. Existing results may be
        removed when the policy requires recomputation.

        Parameters
        ----------
        result : Result
            Persisted result whose cache is being inspected.
        fingerprint : str
            Fingerprint of the configuration required for the computation.

        Returns
        -------
        reusable : bool
            Whether the existing result can be reused.

        Raises
        ------
        FileNotFoundError
            If the cache policy is ``"read_only"`` and no persisted result exists.
        ValueError
            If the cache policy is ``"read_only"`` and the persisted result does
            not match the requested configuration.
        """
        exists = result.dirname.exists()

        if self.cache_policy == "overwrite":
            if exists:
                shutil.rmtree(result.dirname)

            return False

        if self.cache_policy == "reuse":
            return exists

        if not exists:
            if self.cache_policy == "read_only":
                raise FileNotFoundError(f"No cached result found at {result.dirname}.")

            return False

        cache = result.read_cache()
        if cache is not None and cache.get("fingerprint") == fingerprint:
            return True

        if self.cache_policy == "read_only":
            raise ValueError(f"Cached result at {result.dirname} is not valid.")

        shutil.rmtree(result.dirname)
        return False

    @property
    def _kernel(self):
        shoot_config = self.config.get_shoot_config()

        return pdefo.utils.kernel_factory.factory(
            kernel_type=shoot_config.kernel_type,
            kernel_width=shoot_config.kernel_width,
        )

    @property
    def _exponential(self):
        return pdefo.geometry.Exponential(kernel=self._kernel)

    def log(self, point, base_point):
        """Compute the LDDMM logarithm from a base point to a point.

        Registration is performed from ``base_point`` to ``point``. The resulting
        initial control points and momenta define the returned tangent vector.

        Parameters
        ----------
        point : point
            Target shape.
        base_point : point
            Shape at which the tangent vector is based.

        Returns
        -------
        tangent_vec : TangentVector
            Initial tangent vector whose exponential approximates ``point``.
        """
        id_ = f"{base_point.id}_to_{point.id}"
        result = RegistrationResult(id_, self.dir_config, base_point, point)

        config = self.config.get_registration_config()
        fingerprint = config.compute_fingerprint()

        if not self._can_reuse(result, fingerprint):
            pdefo.registration.estimate_registration(
                base_point.as_vtk_path(),
                point.as_vtk_path(),
                target_id=point.id,
                output_dir=result.dirname,
                config=config,
            )
            result.write(
                cache={
                    "fingerprint": fingerprint,
                    "params": config.build_cache_params(),
                }
            )

        return result.tangent_vec

    def exp(self, tangent_vec, base_point):
        """Compute the LDDMM exponential of a tangent vector.

        The deformation defined by the control points and momenta of
        ``tangent_vec`` is shot from ``base_point``.

        Parameters
        ----------
        tangent_vec : TangentVector
            Initial tangent vector defining the geodesic deformation.
        base_point : point
            Shape from which the deformation is shot.

        Returns
        -------
        point : point
            Shape reached at the end of the geodesic.
        """
        result = ShootResult(
            f"{base_point.id}_shoot_{tangent_vec.id}",
            self.dir_config,
            tangent_vec,
            base_point,
        )

        config = self.config.get_shoot_config()
        fingerprint = config.compute_fingerprint()

        if not self._can_reuse(result, fingerprint):
            pdefo.geometry.shoot(
                source=base_point.as_vtk_path(),
                control_points=tangent_vec.control_points.as_path(),
                momenta=tangent_vec.momenta.as_path(),
                output_dir=result.dirname,
                config=config,
            )
            result.write(
                cache={
                    "fingerprint": fingerprint,
                    "params": config.build_cache_params(),
                }
            )
        return result.point

    def _discrete_geodesic_bvp(self, initial_point, end_point):
        """Return the discrete geodesic between two shapes.

        The geodesic is obtained from the registration defining the logarithm
        from ``initial_point`` to ``end_point`` and uses the time discretization
        of the registration.

        Parameters
        ----------
        initial_point : point
            Starting point of the geodesic.
        end_point : point
            Endpoint of the geodesic.

        Returns
        -------
        flow : Flow
            Discrete sequence of shapes along the geodesic.
        """
        self.log(end_point, initial_point)

        result = RegistrationResult(
            f"{initial_point.id}_to_{end_point.id}",
            self.dir_config,
            initial_point,
            end_point,
        )

        return result.flow

    def _discrete_geodesic_ivp(self, initial_point, initial_tangent_vec):
        """Compute a discrete geodesic from initial data.

        The geodesic is obtained by shooting ``initial_tangent_vec`` from
        ``initial_point``. Its discretization therefore follows the integration
        settings of the shooting configuration.

        Parameters
        ----------
        initial_point : point
            Starting point of the geodesic.
        initial_tangent_vec : TangentVector
            Initial tangent vector defining the geodesic.

        Returns
        -------
        flow : Flow
            Discrete sequence of points along the geodesic.
        """
        self.exp(initial_tangent_vec, initial_point)

        result = ShootResult(
            f"{initial_point.id}_shoot_{initial_tangent_vec.id}",
            self.dir_config,
            initial_tangent_vec,
            initial_point,
        )

        return result.flow

    def discrete_geodesic(
        self,
        initial_point,
        end_point=None,
        initial_tangent_vec=None,
    ):
        """Compute a discrete geodesic from initial or boundary data.

        The geodesic can be specified either by its endpoint or by an initial
        tangent vector. When ``initial_tangent_vec`` is provided, the geodesic is
        computed by shooting from ``initial_point``. Otherwise, a registration
        between ``initial_point`` and ``end_point`` defines the geodesic.

        If both ``end_point`` and ``initial_tangent_vec`` are provided,
        ``initial_tangent_vec`` takes precedence.

        Parameters
        ----------
        initial_point : point
            Starting point of the geodesic.
        end_point : point
            Endpoint used for the boundary-value formulation.
        initial_tangent_vec : TangentVector
            Initial tangent vector used for the initial-value formulation.

        Returns
        -------
        flow : Flow
            Discrete sequence of points along the geodesic.

        Raises
        ------
        ValueError
            If neither ``end_point`` nor ``initial_tangent_vec`` is provided.
        """
        if end_point is None and initial_tangent_vec is None:
            raise ValueError(
                "Either ``end_point`` or ``initial_tangent_vec`` must be provided."
            )

        if initial_tangent_vec is not None:
            if end_point is not None:
                logging.warning(
                    "Ignoring ``end_point`` in the computation of the geodesic."
                )

            return self._discrete_geodesic_ivp(
                initial_point,
                initial_tangent_vec,
            )

        return self._discrete_geodesic_bvp(
            initial_point,
            end_point,
        )

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        """Parallel transport a tangent vector along an LDDMM geodesic.

        The reference geodesic can be specified either by an initial
        ``direction`` or by an ``end_point``. When an endpoint is provided, its
        logarithm at ``base_point`` is first computed and used as the reference
        direction.

        If both ``direction`` and ``end_point`` are provided, ``direction`` takes
        precedence.

        Parameters
        ----------
        tangent_vec : TangentVector
            Tangent vector to transport.
        base_point : point
            Starting point of the reference geodesic.
        direction : TangentVector
            Initial tangent vector defining the reference geodesic.
        end_point : point
            Endpoint defining the reference geodesic through a logarithm map.

        Returns
        -------
        transported : TangentVector
            Tangent vector transported along the reference geodesic.

        Raises
        ------
        ValueError
            If neither ``direction`` nor ``end_point`` is provided.
        """
        if direction is None and end_point is None:
            raise ValueError("Either ``direction`` or ``end_point`` must be provided.")

        if direction is not None:
            if end_point is not None:
                logging.warning(
                    "Ignoring ``end_point`` in the computation of parallel transport."
                )

            return self._parallel_transport_ivp(tangent_vec, base_point, direction)

        return self._parallel_transport_bvp(tangent_vec, base_point, end_point)

    def _parallel_transport_bvp(self, tangent_vec, base_point, end_point):
        """Parallel transport a tangent vector along a geodesic defined by endpoints.

        The reference direction is obtained by computing the logarithm of
        ``end_point`` at ``base_point``. Transport is then delegated to the IVP
        formulation.

        Parameters
        ----------
        tangent_vec : TangentVector
            Tangent vector to transport.
        base_point : point
            Starting point of the reference geodesic.
        end_point : point
            Endpoint of the reference geodesic.

        Returns
        -------
        transported : TangentVector
            Transported tangent vector.
        """
        direction = self.log(end_point, base_point)
        return self._parallel_transport_ivp(tangent_vec, base_point, direction)

    def _parallel_transport_ivp(self, tangent_vec, base_point, direction):
        """Parallel transport a tangent vector along a geodesic defined by a direction.

        The reference geodesic starts at ``base_point`` with initial tangent
        vector ``direction``. Transport is performed using the method configured
        in the parallel-transport configuration.

        Tangent vectors whose norm is below ``transport_zero_tol`` are treated as
        zero and bypass the numerical transport routine.

        Parameters
        ----------
        tangent_vec : TangentVector
            Tangent vector to transport.
        base_point : point
            Starting point of the reference geodesic.
        direction : TangentVector
            Initial tangent vector defining the reference geodesic.

        Returns
        -------
        transported : TangentVector
            Tangent vector at the end of the reference geodesic.
        """
        config = self.config.get_parallel_transport_config()
        method = config.method
        if (
            self.transport_zero_tol is not None
            and self.squared_norm(tangent_vec) < self.transport_zero_tol**2
        ):
            method = "zero"

        result = TransportResult(
            f"{tangent_vec.id}_along_{direction.id}",
            self.dir_config,
            tangent_vec,
            base_point,
            direction,
            method=method,
        )

        config = self.config.get_parallel_transport_config()
        fingerprint = config.compute_fingerprint()

        if not self._can_reuse(result, fingerprint):
            if method != "zero":
                pdefo.geometry.parallel_transport(
                    source=base_point.as_vtk_path() if method == "fanning" else None,
                    control_points=direction.control_points.as_path(),
                    momenta=direction.momenta.as_path(),
                    control_points_to_transport=tangent_vec.control_points.as_path(),
                    momenta_to_transport=tangent_vec.momenta.as_path(),
                    config=config,
                    output_dir=result.dirname,
                )
            else:
                result.write_data()

            result.write(
                cache={
                    "fingerprint": fingerprint,
                    "params": config.build_cache_params(),
                }
            )

        return result.transported

    def _move_data(self, *arrays):
        return pdefo.utils.move_data(*arrays)

    def _move_data_back(self, data, like):
        if isinstance(like, np.ndarray):
            return data.detach().cpu().numpy()

        if torch.is_tensor(like):
            return data.to(device=like.device, dtype=like.dtype)

        return data

    def squared_norm(self, tangent_vec, base_point=None):
        r"""Compute the squared LDDMM norm of a tangent vector.

        For control points :math:`c_i` and momenta :math:`p_i`, the squared norm
        is the RKHS quadratic form

        .. math::

            \\|v\\|_V^2
            =
            \\sum_{i,j}
            K(c_i, c_j)
            \\langle p_i, p_j \\rangle.

        Parameters
        ----------
        tangent_vec : TangentVector
            Tangent vector represented by control points and momenta.
        base_point : point
            Base point of the tangent vector. It is not required by the current
            RKHS representation and is ignored.

        Returns
        -------
        squared_norm : scalar
            Squared RKHS norm of the tangent vector.
        """
        # NB: base_point is ignored
        control_points_ = tangent_vec.control_points.as_array()
        control_points, momenta = self._move_data(
            control_points_,
            tangent_vec.momenta.as_array(),
        )

        snorm = self._exponential.scalar_product(
            control_points,
            momenta,
            momenta,
        )
        return self._move_data_back(snorm, like=control_points_)

    def norm(self, tangent_vec, base_point=None):
        """Compute the LDDMM norm of a tangent vector.

        Parameters
        ----------
        tangent_vec : TangentVector
            Tangent vector represented by control points and momenta.
        base_point : point
            Base point of the tangent vector. It is not required by the current
            RKHS representation and is ignored.

        Returns
        -------
        norm : scalar
            RKHS norm of the tangent vector.
        """
        return np.sqrt(self.squared_norm(tangent_vec, base_point))

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the LDDMM inner product of two tangent vectors."""
        # NB: base_point is ignored
        control_points_a_ = tangent_vec_a.control_points.as_array()

        control_points_a, momenta_a, control_points_b, momenta_b = self._move_data(
            control_points_a_,
            tangent_vec_a.momenta.as_array(),
            tangent_vec_b.control_points.as_array(),
            tangent_vec_b.momenta.as_array(),
        )

        velocity_b_at_a = self._kernel.convolve(
            control_points_a,
            control_points_b,
            momenta_b,
        )

        inner_product = (momenta_a * velocity_b_at_a).sum()

        return self._move_data_back(
            inner_product,
            like=control_points_a_,
        )

    def velocity_at(self, x, tangent_vec):
        """Evaluate the velocity field of a tangent vector at given locations."""
        # v(x)=\sum_i K\left(x, c_i\right) p_i
        x_ = x
        x, control_points, momenta = self._move_data(
            x,
            tangent_vec.control_points.as_array(),
            tangent_vec.momenta.as_array(),
        )

        velocity = self._kernel.convolve(x, control_points, momenta)
        return Velocity(x_, self._move_data_back(velocity, like=x_))

    def momenta_from_velocity(self, velocity):
        """Recover kernel momenta representing a sampled velocity field."""
        # returns array
        locations = velocity.locations
        locations_, velocity_ = self._move_data(locations, velocity.values)

        kernel_matrix = self._kernel.get_kernel_matrix(locations_)
        cholesky = torch.linalg.cholesky(kernel_matrix)

        momenta = torch.cholesky_solve(velocity_, cholesky)

        return self._move_data_back(momenta, locations)

    def represent_at(self, locations, tangent_vec):
        """Represent a tangent vector using control points at given locations."""
        velocity = self.velocity_at(locations, tangent_vec)
        momenta = self.momenta_from_velocity(velocity)

        return TangentVector(
            control_points=ControlPoints(locations),
            momenta=Momenta(momenta),
        )

    def tangent_vec_from_velocity(self, velocity):
        """Convert a sampled velocity field to a tangent-vector representation."""
        momenta = self.momenta_from_velocity(velocity)

        return TangentVector(
            control_points=ControlPoints(velocity.locations),
            momenta=Momenta(momenta),
        )
