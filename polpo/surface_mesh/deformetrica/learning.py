from in_out.array_readers_and_writers import write_3D_array

import polpo.ext.deformetrica as pdefo

from .core import DeterministicAtlasResult


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
    initial_step_size : float
        Initial step size used by the atlas optimizer.

    Attributes
    ----------
    metric : LddmmMetric
        LDDMM metric used for atlas estimation.
    config : DeterministicAtlasConfig
        Configuration used for deterministic atlas estimation.
    estimate_ : point or None
        Estimated Fréchet mean after fitting.
    """

    def __init__(self, metric, initial_step_size=1e-4):
        self.metric = metric
        self.initial_step_size = initial_step_size

        self.estimate_ = None

    @property
    def config(self):
        """Return the atlas configuration derived from the current metric."""
        return pdefo.config.DeterministicAtlasConfig.from_registration_config(
            self.metric.config.get_registration_config()
        ).set_optimization(
            initial_step_size=self.initial_step_size,
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

        Returns
        -------
        estimator : FrechetMean
            This fitted estimator.
        """
        self.estimate_ = None

        result = DeterministicAtlasResult(atlas_id, self.metric.dir_config, points=X)

        config = self.config
        fingerprint = config.fingerprint()

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
                    write_3D_array(momenta_, result.dirname, filename)
            else:
                result.write_mesh()

            result.write(
                cache={
                    "fingerprint": fingerprint,
                    "params": config.cache_params(),
                }
            )

        self.estimate_ = result.template

        return self
