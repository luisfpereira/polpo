import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, RegressorMixin


class MixedLMRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-compatible wrapper around statsmodels ``MixedLM``.

    Parameters
    ----------
    formula : str
        Mixed-effects model formula using statsmodels formula syntax.
    groups : str
        Name of the column identifying the grouping variable.
    re_formula : str, optional
        One-sided formula defining the random-effects structure.
    target_name : str, default="Y"
        Name assigned to the response variable in the statsmodels data.
    model_kwargs : dict, optional
        Additional keyword arguments passed to
        ``statsmodels.formula.api.mixedlm``.
    fit_kwargs : dict, optional
        Additional keyword arguments passed to ``MixedLM.fit``.
    """

    def __init__(
        self,
        formula,
        groups,
        re_formula=None,
        target_name="Y",
        model_kwargs=None,
        fit_kwargs=None,
    ):
        self.formula = formula
        self.groups = groups
        self.re_formula = re_formula
        self.target_name = target_name
        self.model_kwargs = model_kwargs
        self.fit_kwargs = fit_kwargs

    def fit(self, X, y):
        """Fit the mixed-effects model.

        Parameters
        ----------
        X : pandas.DataFrame
            Training covariates and mixed-effects metadata. It must contain
            the variables referenced by ``formula`` and ``re_formula``, and
            the grouping column specified by ``groups``.
        y : array-like of shape (n_samples,)
            Response values.

        Returns
        -------
        self
            Fitted estimator.
        """
        data = X.copy()
        data[self.target_name] = y

        model_kwargs = {} if self.model_kwargs is None else self.model_kwargs
        fit_kwargs = {} if self.fit_kwargs is None else self.fit_kwargs

        self.model_ = smf.mixedlm(
            self.formula,
            data=data,
            groups=self.groups,
            re_formula=self.re_formula,
            **model_kwargs,
        )
        self.result_ = self.model_.fit(**fit_kwargs)

        return self

    def predict(self, X):
        """Predict the fixed-effects mean response.

        Parameters
        ----------
        X : pandas.DataFrame
            Prediction covariates. It must contain the variables referenced by
            the fixed-effects part of ``formula``. The grouping column used
            during fitting is not required.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted response values based on the fitted fixed effects.

        Notes
        -----
        Predictions do not include subject-specific estimated random effects.
        """
        return self.result_.predict(X)
