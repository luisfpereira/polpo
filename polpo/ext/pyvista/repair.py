from polpo.utils import params_to_kwargs


class PvExtractLargest:
    """Extract largest connected set in mesh.

    https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.extract_largest#pyvista.DataSetFilters.extract_largest
    """

    def __init__(self, inplace=False, progress_bar=False):
        super().__init__()
        self.inplace = inplace
        self.progress_bar = progress_bar

    def __call__(self, poly_data):
        """Apply step.

        Parameters
        ----------
        poly_data : pv.PolyData
            Mesh.

        Returns
        -------
        mesh : pv.PolyData
            Largest mesh component.
        """
        return poly_data.extract_largest(**params_to_kwargs(self))
