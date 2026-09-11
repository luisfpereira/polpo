"""Extensions for sklearn decomposition estimators.

This module provides utilities for reusing fitted decomposition models,
including lower-rank truncated views of fitted PCA estimators.
"""

from sklearn.decomposition import PCA


class TruncatedPCA(PCA):
    """PCA obtained by truncating an already fitted PCA.

    ``TruncatedPCA`` is a standalone fitted PCA object.
    Its fitted component-dependent attributes, such as
    ``components_``, ``explained_variance_``, and ``singular_values_``, are
    truncated to the requested number of components.

    Instances are typically constructed with :meth:`from_fitted`, allowing a
    larger PCA to be fitted once and reused to obtain fitted PCA objects for
    several smaller numbers of components.
    """

    @classmethod
    def from_fitted(cls, pca, n_components):
        """Create a fitted PCA by truncating another fitted PCA.

        Parameters
        ----------
        pca : PCA
            Fitted PCA providing the principal components and fitted
            attributes.
        n_components : int
            Number of leading principal components to retain.

        Returns
        -------
        truncated : TruncatedPCA
            Standalone fitted PCA containing the first ``n_components``
            principal components of ``pca``.

        Notes
        -----
        Component-dependent fitted attributes are truncated consistently to
        ``n_components``, while fitted attributes that do not depend on the
        retained rank, such as ``mean_`` and ``n_features_in_``, are copied
        from ``pca``.

        ``noise_variance_`` should not simply be copied from ``pca``, since
        truncating the PCA changes the variance assigned to the discarded
        subspace.
        """
        if n_components > pca.n_components_:
            raise ValueError(
                "n_components cannot exceed the number of fitted components."
            )

        params = pca.get_params()
        params["n_components"] = n_components

        out = cls(**params)

        out.components_ = pca.components_[:n_components].copy()
        out.explained_variance_ = pca.explained_variance_[:n_components].copy()
        out.explained_variance_ratio_ = pca.explained_variance_ratio_[
            :n_components
        ].copy()
        out.singular_values_ = pca.singular_values_[:n_components].copy()

        out.mean_ = pca.mean_.copy()
        out.n_components_ = n_components
        out.n_samples_ = pca.n_samples_
        out.n_features_in_ = pca.n_features_in_
        out._fit_svd_solver = pca._fit_svd_solver

        return out
