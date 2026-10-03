import abc
import logging
import warnings

import numpy as np


class ShiftBased1dRegistration(abc.ABC):
    def apply_transformation(self, curve, transformation):
        return np.roll(curve, -transformation)

    def register(self, moving_curve, target_curve):
        """Register curve.

        Parameters
        ----------
        moving_signal : array-like
            The signal to be registered
        target_signal : array-like
            The target signal to register to

        Returns
        -------
        registered_signal : array-like
            The registered version of moving_signal
        shift : int
            The shift applied to achieve registration
        """
        # TODO: improve docstrings
        return self.apply_transformation(
            moving_curve, self.find_transformation(moving_curve, target_curve)
        )

    @abc.abstractmethod
    def find_transformation(self, moving_curve, target_curve):
        pass


class CorrelationBased1dRegistration(ShiftBased1dRegistration):
    """Register a 1D signal to a target signal using cross-correlation."""

    # inspired by Adele Myers's code
    def find_transformation(self, moving_curve, target_curve):
        """Find transformation.

        Parameters
        ----------
        moving_signal : array-like
            The signal to be registered
        target_signal : array-like
            The target signal to register to

        Returns
        -------
        registered_signal : array-like
            The registered version of moving_signal
        shift : int
            The shift applied to achieve registration
        """
        correlation = np.correlate(target_curve, moving_curve, mode="full")

        # Find the shift that gives maximum correlation
        return np.argmax(correlation) - (len(moving_curve) - 1)


class MseBased1dRegistration(ShiftBased1dRegistration):
    """Register a 1D signal to a target signal using cross-correlation."""

    # inspired by Adele Myers' code
    def find_transformation(self, moving_curve, target_curve):
        """Find transformation.

        Parameters
        ----------
        moving_signal : array-like
            The signal to be registered
        target_signal : array-like
            The target signal to register to

        Returns
        -------
        registered_signal : array-like
            The registered version of moving_signal
        shift : int
            The shift applied to achieve registration
        """
        mse = np.zeros(len(moving_curve))
        for shift in range(len(moving_curve)):
            shifted_signal = np.roll(moving_curve, -shift)
            mse[shift] = np.mean((target_curve - shifted_signal) ** 2)

        return np.argmin(mse)


class CurveCollectionRegistration:
    # TODO: new template func
    # TODO: stopping criteria?

    def __init__(
        self, registration=None, initialization=None, threshold=1e-6, max_iter=20
    ):
        if registration is None:
            registration = MseBased1dRegistration()

        if initialization is None:
            initialization = lambda x: np.mean(x, axis=0)

        self.registration = registration
        self.initialization = initialization
        self.threshold = threshold
        self.max_iter = max_iter

    def register(self, curves, template=None):
        if template is None:
            template = self.initialization(curves)

        for iter_ in range(self.max_iter):
            # Register all signals to current mean
            registered_curves = np.stack(
                [self.registration.register(curve, template) for curve in curves]
            )

            # Compute new mean
            new_template = np.mean(registered_curves, axis=0)

            # Check convergence
            mean_diff = np.mean(np.abs(new_template - template))
            if mean_diff < self.threshold:
                break

            template = new_template
        else:
            warnings.warn(f"Reached maximum number of iterations: {self.max_iter}")
            return registered_curves

        # TODO: add history instead?
        logging.info(f"Registration converged after {iter_} iterations")

        # TODO: store template?
        return registered_curves
