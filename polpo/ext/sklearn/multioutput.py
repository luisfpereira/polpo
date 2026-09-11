"""Extensions for sklearn multi-output estimators.

This module provides utilities for working with fitted multi-output models,
including truncation to a subset of leading output estimators.
"""

from sklearn.multioutput import MultiOutputRegressor


class TruncatedMultiOutputRegressor(MultiOutputRegressor):
    """Multi-output regressor obtained by truncating an already fitted model.

    ``TruncatedMultiOutputRegressor`` is a standalone fitted multi-output
    regressor retaining only the first requested output estimators.

    Instances are typically constructed with :meth:`from_fitted`, allowing a
    larger multi-output model to be fitted once and reused for several smaller
    numbers of outputs.
    """

    @classmethod
    def from_fitted(cls, regressor, n_outputs):
        """Create a fitted regressor by truncating another fitted regressor.

        Parameters
        ----------
        regressor : MultiOutputRegressor
            Fitted multi-output regressor providing the output estimators.
        n_outputs : int
            Number of leading output estimators to retain.

        Returns
        -------
        truncated : TruncatedMultiOutputRegressor
            Fitted multi-output regressor containing the first ``n_outputs``
            estimators of ``regressor``.
        """
        if n_outputs > len(regressor.estimators_):
            raise ValueError(
                "n_outputs cannot exceed the number of fitted output estimators."
            )

        out = cls(**regressor.get_params(deep=False))
        out.estimators_ = list(regressor.estimators_[:n_outputs])

        if hasattr(regressor, "n_features_in_"):
            out.n_features_in_ = regressor.n_features_in_

        if hasattr(regressor, "feature_names_in_"):
            out.feature_names_in_ = regressor.feature_names_in_.copy()

        return out
