"""LDDMM estimators for atlas and longitudinal shape analysis."""

import polpo.ext.deformetrica as pdefo

from .results import (
    DeterministicAtlasResult,
    GeodesicRegressionResult,
    SplineRegressionResult,
)


class FrechetMean:
    """Estimate a Fréchet mean of shapes using deterministic atlas estimation.

    The estimator uses the LDDMM geometry defined by ``metric`` to configure
    Deformetrica's deterministic atlas estimation. The atlas configuration is
    initialized from the metric's registration configuration and can then be
    adjusted independently.

    Parameters
    ----------
    metric : LddmmMetric
        LDDMM metric defining the deformation, attachment, integration, and
        execution settings used for atlas estimation.

    Attributes
    ----------
    metric : LddmmMetric
        LDDMM metric used for atlas estimation.
    result_ : DeterministicAtlasResult
    """

    def __init__(self, metric):
        self.metric = metric

        self.result_ = None

    @property
    def estimate_(self):
        """Estimated Fréchet mean."""
        if self.result_ is None:
            raise RuntimeError("Estimator has not been fitted yet.")

        return self.result_.template

    def _make_config(self):
        """Return the atlas configuration derived from the current metric."""
        return pdefo.config.DeterministicAtlasConfig.from_registration_config(
            self.metric.config.get_registration_config()
        )

    def fit(self, X, atlas_id="atlas", initial_point=None):
        """Estimate the Fréchet mean of a collection of shapes.

        For multiple observations, the mean is estimated with Deformetrica's
        deterministic atlas model. For a single observation, that shape is
        used directly as the estimate.

        Results are persisted and reused according to the cache policy of the
        underlying metric.

        Parameters
        ----------
        X : sequence of points
            Shapes from which to estimate the Fréchet mean.
        atlas_id : str
            Identifier used for the estimated atlas and its persisted results.
        initial_point : Point
            Initial atlas template. If not provided, the first observation is used.

        Returns
        -------
        estimator : FrechetMean
            This fitted estimator.
        """
        result = DeterministicAtlasResult(atlas_id, self.metric.dir_config, points=X)

        config = self._make_config()
        fingerprint = config.compute_fingerprint()

        if not self.metric._can_reuse(result, fingerprint):
            if len(X) > 1:
                dataset = {point.id: point.as_vtk_path() for point in X}
                pdefo.learning.estimate_deterministic_atlas(
                    targets=dataset,
                    output_dir=result.dirname,
                    config=config,
                    source=None
                    if initial_point is None
                    else initial_point.as_vtk_path(),
                )

                momenta = pdefo.io.read_array(pdefo.io.find_momenta(result.dirname))
                for momenta_, point in zip(momenta, X):
                    filename = f"DeterministicAtlas__EstimatedParameters__Momenta__subject_{point.id}.txt"
                    pdefo.io.write_array(result.dirname / filename, momenta_)
            else:
                result.write_mesh()

            result.write(
                cache={
                    "fingerprint": fingerprint,
                    "params": config.build_cache_params(),
                }
            )

        self.result_ = result

        return self


class GeodesicRegression:
    """Geodesic regression for longitudinal shape data.

    Parameters
    ----------
    metric : LddmmMetric
        LDDMM metric defining the deformation model.
    t0 : float
        Reference time of the regression trajectory.

    Attributes
    ----------
    metric : LddmmMetric
        LDDMM metric defining the deformation model.
    t0 : float
        Reference time of the regression trajectory.
    result_ : GeodesicRegressionResult
    """

    def __init__(self, metric, t0=0.0):
        self.metric = metric

        self.t0 = t0

        self.result_ = None

    @property
    def tangent_vec_(self):
        """Estimated geodesic."""
        if self.result_ is None:
            raise RuntimeError("Estimator has not been fitted yet.")

        return self.result_.tangent_vec

    def _make_config(self):
        config = pdefo.config.GeodesicRegressionConfig.from_registration_config(
            self.metric.config.get_registration_config()
        )
        config.model.set_regression(t0=self.t0)

        return config

    def fit(
        self,
        X,
        times,
        base_point,
        regression_id,
    ):
        """Fit a regression to longitudinal shapes.

        Parameters
        ----------
        X : sequence of Point
            Observed shapes.
        times : array-like, shape (n_samples,)
            Observation times.
        base_point : Point
            Source shape representing the trajectory at reference time ``t0``.
        regression_id : str
            Identifier used to persist the regression result.

        Returns
        -------
        self : GeodesicRegression
            Fitted estimator.
        """
        config = self._make_config()

        result = GeodesicRegressionResult(
            regression_id,
            self.metric.dir_config,
            base_point=base_point,
            points=X,
            times=times,
        )

        fingerprint = config.compute_fingerprint()

        if not self.metric._can_reuse(result, fingerprint):
            pdefo.learning.estimate_geodesic_regression(
                source=base_point.as_vtk_path(),
                targets=[point.as_vtk_path() for point in X],
                times=times,
                output_dir=result.dirname,
                config=config,
            )

            result.write(
                cache={
                    "fingerprint": fingerprint,
                    "params": config.build_cache_params(),
                }
            )

        self.result_ = result
        return self


class SplineRegression:
    """LDDMM spline regression for longitudinal shape data.

    Fit a time-dependent deformation trajectory through longitudinal shape
    observations while allowing non-geodesic deviations through external
    forces.

    Parameters
    ----------
    metric : LddmmMetric
        LDDMM metric defining the deformation model.
    t0 : float
        Reference time of the spline trajectory.
    geodesic_weight : float
        Relative weight of the geodesic component with respect to the
        external-force component.

    Attributes
    ----------
    metric : LddmmMetric
        LDDMM metric defining the deformation model.
    t0 : float
        Reference time of the spline trajectory.
    geodesic_weight : float
        Relative weight of the geodesic component with respect to the
        external-force component.
    result_ : SplineRegressionResult
    """

    def __init__(
        self,
        metric,
        geodesic_weight=0.1,
        t0=0.0,
    ):
        self.metric = metric

        self.t0 = t0
        self.geodesic_weight = geodesic_weight

        self.result_ = None

    @property
    def tangent_vec_(self):
        """Initial tangent vector of the fitted spline."""
        if self.result_ is None:
            raise RuntimeError("Estimator has not been fitted yet.")

        return self.result_.tangent_vec

    @property
    def external_forces_(self):
        """Estimated time-discretized external forces."""
        if self.result_ is None:
            raise RuntimeError("Estimator has not been fitted yet.")

        return self.result_.external_forces

    def _make_config(self):
        config = pdefo.config.SplineRegressionConfig.from_registration_config(
            self.metric.config.get_registration_config()
        )
        config.model.set_regression(t0=self.t0, geodesic_weight=self.geodesic_weight)
        return config

    def fit(
        self,
        X,
        times,
        base_point,
        regression_id,
        weights=None,
    ):
        """Fit a regression to longitudinal shapes.

        Parameters
        ----------
        X : sequence of Point
            Observed shapes.
        times : array-like, shape (n_samples,)
            Observation times.
        base_point : Point
            Source shape representing the trajectory at reference time
            ``config.t0``.
        regression_id : str
            Identifier used to persist the regression result.
        weights : array-like, shape (n_samples,)
            Weights applied to the observations in the regression objective.

        Returns
        -------
        self : SplineRegression
            Fitted estimator.
        """
        config = self._make_config()

        result = SplineRegressionResult(
            regression_id,
            self.metric.dir_config,
            base_point=base_point,
            points=X,
            times=times,
        )

        fingerprint = config.compute_fingerprint()

        if not self.metric._can_reuse(result, fingerprint):
            pdefo.learning.estimate_spline_regression(
                source=base_point.as_vtk_path(),
                targets=[point.as_vtk_path() for point in X],
                times=times,
                output_dir=result.dirname,
                config=config,
                weights=weights,
            )

            result.write(
                cache={
                    "fingerprint": fingerprint,
                    "params": config.build_cache_params(),
                }
            )

        self.result_ = result
        return self
