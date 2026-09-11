import abc

from sklearn.base import clone

from polpo.logging import logger


class BaseMessagePrinter(abc.ABC):
    def __call__(self, method, model, *args):
        return getattr(self, method)(model, *args)


class ModelBasicMessagePrinter(BaseMessagePrinter):
    def predict(self, model, X, y):
        return f"X: {X} -> y: {y}"


class TelemeteredModel:
    """Model telemetry.

    Wraps a model to log info.

    Parameters
    ----------
    model : sklearn.BaseEstimator
        Estimator being telemetered.
    msg_printer : BaseMessagePrinter
        Handles message to be logged.
    """

    def __init__(self, model, msg_printer=None):
        if msg_printer is None:
            msg_printer = ModelBasicMessagePrinter()

        self.model = model
        self.msg_printer = msg_printer

    def __getattr__(self, name):
        """Get attribute.

        It is only called when ``__getattribute__`` fails.
        Delegates attribute calling to model.
        """
        return getattr(self.model, name)

    def __sklearn_clone__(self):
        return TelemeteredModel(model=clone(self.model), msg_printer=self.msg_printer)

    def fit(self, X, y=None):
        self.model = self.model.fit(X, y)
        return self

    def predict(self, X):
        y = self.model.predict(X)
        logger.info(self.msg_printer("predict", self.model, X, y))
        return y
