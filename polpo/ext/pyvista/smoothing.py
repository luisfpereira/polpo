from polpo.pipeline.base import PreprocessingStep
from polpo.utils import params_to_kwargs


class PvSmoothTaubin(PreprocessingStep):
    """Smooth a PolyData DataSet with Taubin smoothing.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.polydatafilters.smooth_taubin#pyvista.PolyDataFilters.smooth_taubin
    """

    def __init__(
        self,
        n_iter=20,
        pass_band=0.1,
        edge_angle=15.0,
        feature_angle=45.0,
        boundary_smoothing=True,
        feature_smoothing=False,
        non_manifold_smoothing=False,
        normalize_coordinates=False,
        inplace=False,
        progress_bar=False,
    ):
        self.n_iter = n_iter
        self.pass_band = pass_band
        self.edge_angle = edge_angle
        self.feature_angle = feature_angle
        self.boundary_smoothing = boundary_smoothing
        self.feature_smoothing = feature_smoothing
        self.non_manifold_smoothing = non_manifold_smoothing
        self.normalize_coordinates = normalize_coordinates
        self.inplace = inplace
        self.progress_bar = progress_bar

    def __call__(self, poly_data):
        """Apply step."""
        return poly_data.smooth_taubin(**params_to_kwargs(self))
