import abc
import time

import geomstats.backend as gs
from geomstats.geometry.discrete_surfaces import (
    Surface,
    _is_iterable,
)
from geomstats.geometry.fiber_bundle import AlignerAlgorithm
from geomstats.numerics.geodesic import PathBasedLogSolver
from geomstats.numerics.optimization import ScipyMinimize
from geomstats.numerics.path import (
    UniformlySampledDiscretePath,
    UniformlySampledPathEnergy,
)

from polpo.logging import logger

# TODO: objective func strategy inspired by geomfum; generalize?


def _warn_max_iterations(iteration, max_iter):
    if iteration + 1 == max_iter:
        logger.warning(
            f"Maximum number of iterations {max_iter} reached. "
            "The estimate may be inaccurate"
        )


class WeightedFactor(abc.ABC):
    """Weighted factor.

    Parameters
    ----------
    weight : float
        Weight of the factor.
    """

    # TODO: create TelemeteredFactor taking in a Factor
    # TODO: add id in Telemetered?

    def __init__(self, weight):
        self.weight = weight

    @abc.abstractmethod
    def __call__(self, path):
        """Compute energy.

        Parameters
        ----------
        path : array-like, shape=[..., n_times, *point_shape]
            Piecewise linear path.

        Returns
        -------
        weighted_energy : float
            Weighted energy associated with the factor.
        """
        pass

    def initialize(self, point, base_point):
        pass


class VarifoldLoss(WeightedFactor):
    def __init__(self, varifold_metric, target_faces=None, lambda_=1.0):
        # if target_faces is None, uses `base_point`
        super().__init__(weight=lambda_)
        self.varifold_metric = varifold_metric
        self.target_faces = target_faces

        self._call = None

    def initialize(self, point, base_point=None):
        # base_point is ignored if target_faces is not None
        target_faces = (
            self.target_faces if self.target_faces is not None else base_point.faces
        )
        self._call = self.varifold_metric.loss(point, target_faces=target_faces)

    def __call__(self, path):
        return self.weight * self._call(path[-1])


class FactorSum(WeightedFactor):
    """Factor sum.

    Parameters
    ----------
    factors : list[WeightedFactor]
        Factors.
    """

    def __init__(self, factors, weight=1.0):
        super().__init__(weight=weight)
        self.factors = factors

    def __call__(self, path):
        """Compute energy.

        Parameters
        ----------
        path : array-like, shape=[..., n_times, *point_shape]
            Piecewise linear path.

        Returns
        -------
        weighted_energy : float
            Weighted energy associated with the factor.
        """
        return self.weight * gs.sum(gs.array([factor(path) for factor in self.factors]))

    def initialize(self, point, base_point):
        for factor in self.factors:
            if not hasattr(factor, "initialize"):
                continue

            factor.initialize(point, base_point)


class LossWithDataAttachment(FactorSum):
    # just syntax sugar
    def __init__(self, factor, data_attachment, weight=1.0):
        super().__init__([factor, data_attachment])
        self.factor = factor
        self.data_attachment = data_attachment

    def update_lambda(self, lambda_):
        self.data_attachment.weight = lambda_

    def update_current_lambda(self, ratio):
        self.data_attachment.weight *= ratio

        return self.data_attachment.weight


class _DiscreteSurfaceGeodesicBVPBatchMixin:
    @abc.abstractmethod
    def _discrete_geodesic_bvp_single(self, point, base_point):
        """Solve boundary value problem (BVP).

        Given an initial point and an end point, solve the geodesic equation
        via minimizing the Riemannian path energy.

        Parameters
        ----------
        point : Surface or list[Surface] or array-like, shape=[n_vertices, 3]
        base_point : array-like, shape=[n_vertices, 3]

        Returns
        -------
        discr_geod_path : array-like, shape=[n_times, n_nodes, *point_shape]
            Discrete geodesic.
        """

    def discrete_geodesic_bvp(self, point, base_point):
        """Solve boundary value problem (BVP).

        Given an initial point and an end point, solve the geodesic equation
        via minimizing the Riemannian path energy and a relaxation term.

        Parameters
        ----------
        point : Surface or list[Surface] or array-like, shape=[..., n_vertices, 3]
        base_point : Surface or list[Surface] or array-like, shape=[..., n_vertices, 3]

        Returns
        -------
        discr_geod_path : array-like, shape=[..., n_times, n_nodes, n_vertices, 3]
            Discrete geodesic.
        """
        if not gs.is_array(base_point):
            if _is_iterable(base_point):
                base_point = gs.stack([point_.vertices for point_ in base_point])
            else:
                base_point = base_point.vertices

        if gs.is_array(point):
            if point.ndim > 2:
                point = [
                    Surface(point_, faces=self._total_space.faces) for point_ in point
                ]
            else:
                point = Surface(point, faces=self._total_space.faces)

        if base_point.ndim == 2 and not _is_iterable(point):
            return self._discrete_geodesic_bvp_single(point, base_point)

        if base_point.ndim == 2:
            batch_shape = (len(base_point),)
            base_point = gs.broadcast_to(
                base_point, batch_shape + self._total_space.shape
            )

        elif not _is_iterable(point):
            point = [point] * base_point.shape[-3]

        return gs.stack(
            [
                self._discrete_geodesic_bvp_single(point_, base_point_)
                for base_point_, point_ in zip(base_point, point)
            ]
        )


class RelaxedPathStraightening(
    _DiscreteSurfaceGeodesicBVPBatchMixin, PathBasedLogSolver, AlignerAlgorithm
):
    """Class to solve the geodesic boundary value problem with path-straightening.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    n_nodes : int
        Number of path discretization points.
    lambda_ : float
        Discrepancy loss weight.
    discrepancy_loss : callable
        A generic discrepancy term. Receives point and outputs a callable
        which receives another point and outputs a scalar measuring
        discrepancy between point and another point.
        Scalar sums to energy of a path.
    path_energy : callable
        Method to compute Riemannian path energy.
    optimizer : ScipyMinimize
        An optimizer to solve path energy minimization problem.
    initialization : callable or array-like, shape=[n_nodes - 2, n_vertices, 3]
        A method to get initial guess for optimization or an initial path.

    References
    ----------
    .. [HSKCB2022] "Elastic shape analysis of surfaces with second-order
        Sobolev metrics: a comprehensive numerical framework".
        arXiv:2204.04238 [cs.CV], 25 Sep 2022
    """

    # TODO: update geomstats

    def __init__(
        self,
        total_space,
        loss,
        n_nodes=3,
        optimizer=None,
        initialization=None,
    ):
        PathBasedLogSolver.__init__(self, space=total_space)
        AlignerAlgorithm.__init__(self, total_space=total_space)

        if initialization is None:
            initialization = self._default_initialization

        if optimizer is None:
            optimizer = ScipyMinimize(
                method="L-BFGS-B",
                autodiff_jac=True,
                options={"disp": False, "maxiter": 1000},
                tol=1e-4,
            )

        self.loss = loss
        self.n_nodes = n_nodes
        self.optimizer = optimizer
        self.initialization = initialization

    @classmethod
    def from_default_loss(
        cls,
        total_space,
        n_nodes=3,
        lambda_=1.0,
        sigma=1.0,
        optimizer=None,
        initialization=None,
    ):
        from geomstats.varifold import (
            BinetKernel,
            GaussianKernel,
            SurfacesKernel,
            VarifoldMetric,
        )

        path_energy = UniformlySampledPathEnergy(total_space)

        position_kernel = GaussianKernel(sigma=sigma, init_index=0)
        tangent_kernel = BinetKernel(init_index=position_kernel.new_variable_index())
        kernel = SurfacesKernel(
            position_kernel,
            tangent_kernel,
        )
        varifold_metric = VarifoldMetric(kernel)
        varifold_loss = VarifoldLoss(varifold_metric, target_faces=total_space.faces)

        loss = LossWithDataAttachment(path_energy, varifold_loss)

        return cls(
            total_space=total_space,
            loss=loss,
            n_nodes=n_nodes,
            optimizer=optimizer,
            initialization=initialization,
        )

    def _default_initialization(self, point, base_point):
        """Linear initialization.

        Parameters
        ----------
        point : array-like, shape=[n_vertices, 3]
        base_point : array-like, shape=[n_vertices, 3]

        Returns
        -------
        midpoints : array-like, shape=[n_nodes - 1, n_vertices, 3]
        """
        return gs.broadcast_to(
            base_point, (self.n_nodes - 1,) + self._total_space.shape
        )

    def _discrete_geodesic_bvp_single(self, point, base_point):
        """Solve boundary value problem (BVP).

        Given an initial point and an end point, solve the geodesic equation
        via minimizing the Riemannian path energy.

        Parameters
        ----------
        point : Surface or list[Surface] or array-like, shape=[n_vertices, 3]
        base_point : array-like, shape=[n_vertices, 3]

        Returns
        -------
        discr_geod_path : array-like, shape=[n_times, n_nodes, *point_shape]
            Discrete geodesic.
        """
        if callable(self.initialization):
            init_midpoints = self.initialization(point, base_point)
        else:
            init_midpoints = self.initialization

        self.loss.initialize(point, base_point)

        base_point = gs.expand_dims(base_point, axis=0)

        def objective(midpoints):
            """Compute path energy of paths going through a midpoint.

            Parameters
            ----------
            midpoint : array-like, shape=[(self.n_nodes-1) * math.prod(n_vertices*3)]
                Midpoints of the path.

            Returns
            -------
            _ : array-like, shape=[...,]
                Energy of the path going through this midpoint.
            """
            midpoints = gs.reshape(
                midpoints, (self.n_nodes - 1,) + self._total_space.shape
            )
            path = gs.concatenate([base_point, midpoints])

            return self.loss(path)

        init_midpoints = gs.reshape(init_midpoints, (-1,))
        sol = self.optimizer.minimize(objective, init_midpoints)

        solution_midpoints = gs.reshape(
            gs.array(sol.x), (self.n_nodes - 1,) + self._total_space.shape
        )

        return gs.concatenate(
            [
                base_point,
                solution_midpoints,
            ],
            axis=0,
        )

    def align(self, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        point : Surface or list[Surface] or array-like, shape=[..., n_vertices, 3]
        base_point : array-like, shape=[..., n_vertices, 3]

        Returns
        -------
        aligned_point : array-like, shape=[..., n_vertices, 3]
            Aligned point.
        """
        discr_geod_path = self.discrete_geodesic_bvp(point, base_point)
        point_ndim_slc = (slice(None),) * self._total_space.point_ndim
        return discr_geod_path[(..., -1) + point_ndim_slc]


class LambdaAdaptiveRelaxedPathStraightening(
    _DiscreteSurfaceGeodesicBVPBatchMixin, PathBasedLogSolver, AlignerAlgorithm
):
    def __init__(
        self,
        total_space,
        # TODO: improve, also in geomstats
        straightner,
        initial_lambda=10.0,
        lambda_ratio=10.0,
        max_iter=5,
        sdist_tol=0.001,  # use distance to decimate
    ):
        # assumes loss to be a LossWithDataAttachment

        PathBasedLogSolver.__init__(self, space=total_space)
        AlignerAlgorithm.__init__(self, total_space=total_space)

        self.straightener = straightner
        self.initial_lambda = initial_lambda
        self.lambda_ratio = lambda_ratio
        self.max_iter = max_iter
        self.sdist_tol = sdist_tol

    @classmethod
    def from_default_loss(
        cls,
        total_space,
        n_nodes=3,
        lambda_=1.0,
        sigma=1.0,
        optimizer=None,
        initial_lambda=10.0,
        lambda_ratio=10.0,
        max_iter=5,
        sdist_tol=0.001,
        initialization=None,
    ):
        straightener = (
            RelaxedPathStraightening.from_default_loss(
                total_space,
                n_nodes=n_nodes,
                optimizer=optimizer,
                lambda_=lambda_,
                sigma=sigma,
                initialization=initialization,
            ),
        )

        return cls(
            total_space,
            straightener,
            initial_lambda=initial_lambda,
            lambda_ratio=lambda_ratio,
            max_iter=max_iter,
            sdist_tol=sdist_tol,
        )

    @classmethod
    def from_default_straightner(
        cls,
        total_space,
        loss,
        n_nodes=3,
        optimizer=None,
        initial_lambda=10.0,
        lambda_ratio=10.0,
        max_iter=5,
        sdist_tol=0.001,
        initialization=None,
    ):
        straightener = RelaxedPathStraightening(
            total_space,
            loss,
            n_nodes=n_nodes,
            optimizer=optimizer,
            initialization=initialization,
        )

        return cls(
            total_space,
            straightener,
            initial_lambda=initial_lambda,
            lambda_ratio=lambda_ratio,
            max_iter=max_iter,
            sdist_tol=sdist_tol,
        )

    def _new_initialization(self, discrete_path):
        geod_path = UniformlySampledDiscretePath(discrete_path, point_ndim=2)
        return lambda *args: geod_path(
            gs.linspace(0.0, 1.0, num=self.straightener.n_nodes)[1:]
        )

    def _discrete_geodesic_bvp_single(self, point, base_point, return_paths=False):
        initial_initialization = self.straightener.initialization

        loss = self.straightener.loss
        loss.update_lambda(self.initial_lambda)

        geods = []
        for iteration in range(self.max_iter):
            # TODO: make it own func for possible wrapping later

            # TODO: control verbosity
            logger.info(
                f"Running iteration {iteration}, lambda: {self.straightener.loss.data_attachment.weight}"
            )
            start_time = time.perf_counter()
            if geods:
                self.straightener.initialization = self._new_initialization(geods[-1])

            discrete_path = self.straightener.discrete_geodesic_bvp(point, base_point)
            geods.append(discrete_path)

            logger.info(
                f"Iteration {iteration} run in {time.perf_counter() - start_time:.2f} s"
            )

            # TODO: may use a different strategy here if notion of discrepancy loss
            # TODO: update this
            sdist = loss.data_attachment.varifold_metric.squared_dist(
                point,
                Surface(
                    vertices=discrete_path[-1],
                    faces=self.straightener._total_space.faces,
                ),
            )
            if sdist < self.sdist_tol:
                break

            loss.update_current_lambda(self.lambda_ratio)

        else:
            _warn_max_iterations(iteration, self.max_iter)

        self.straightener.initialization = initial_initialization

        if return_paths:
            return geods

        return geods[-1]

    def align(self, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        point : Surface or list[Surface] or array-like, shape=[..., n_vertices, 3]
        base_point : array-like, shape=[..., n_vertices, 3]

        Returns
        -------
        aligned_point : array-like, shape=[..., n_vertices, 3]
            Aligned point.
        """
        discr_geod_path = self.discrete_geodesic_bvp(point, base_point)
        point_ndim_slc = (slice(None),) * self._total_space.point_ndim
        return discr_geod_path[(..., -1) + point_ndim_slc]
