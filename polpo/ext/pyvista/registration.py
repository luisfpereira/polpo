from polpo.registration.base import BaseRegistration
from polpo.utils import params_to_kwargs


class IcpRegistration(BaseRegistration):
    """Rigid registration using PyVista's ICP alignment.

    This class adapts :meth:`pyvista.DataSetFilters.align` to the
    :class:`BaseRegistration` interface. The rigid transformation is
    estimated using iterative closest point (ICP) registration and stored
    as a homogeneous transformation matrix in ``transform_``.

    Parameters
    ----------
    max_landmarks : int
        Maximum number of landmarks used by the ICP algorithm.
    max_mean_distance : float
        Maximum mean distance between matched points used as a convergence
        criterion.
    max_iterations : int
        Maximum number of ICP iterations.
    check_mean_distance : bool
        If True, stop when the mean distance between matched points satisfies
        the convergence criterion.
    start_by_matching_centroids : bool
        If True, initialize the alignment by matching the source and target
        centroids.

    References
    ----------
    PyVista ``DataSetFilters.align`` documentation:
    https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.align
    """

    def __init__(
        self,
        max_landmarks=100,
        max_mean_distance=1e-05,
        max_iterations=500,
        check_mean_distance=True,
        start_by_matching_centroids=True,
    ):
        self.max_landmarks = max_landmarks
        self.max_mean_distance = max_mean_distance
        self.max_iterations = max_iterations
        self.check_mean_distance = check_mean_distance
        self.start_by_matching_centroids = start_by_matching_centroids

    def fit(self, source, target):
        """Estimate the rigid transformation from source to target.

        Parameters
        ----------
        source : pyvista.DataSet
            Source dataset.
        target : pyvista.DataSet
            Target dataset.

        Returns
        -------
        self : PvAlign
            Fitted registration.
        """
        _, self.transform_ = source.align(
            target,
            **params_to_kwargs(self),
            return_matrix=True,
        )
        return self

    def transform(self, source):
        """Apply the estimated rigid transformation.

        Parameters
        ----------
        source : pyvista.DataSet
            Dataset to transform.

        Returns
        -------
        transformed : pyvista.DataSet
            Transformed dataset. The returned dataset has the same PyVista
            dataset type as the input.
        """
        return source.transform(self.transform_, inplace=False)

    def fit_transform(self, source, target):
        """Estimate the transformation and align the source to the target.

        Parameters
        ----------
        source : pyvista.DataSet
            Source dataset.
        target : pyvista.DataSet
            Target dataset.

        Returns
        -------
        transformed : pyvista.DataSet
            Source dataset rigidly aligned to the target.
        """
        transformed, self.transform_ = source.align(
            target,
            **params_to_kwargs(self, ignore=("transform_",)),
            return_matrix=True,
        )
        return transformed
