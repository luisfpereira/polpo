from abc import ABC, abstractmethod


class BaseRegistration(ABC):
    """Base class for registration algorithms."""

    @abstractmethod
    def fit(self, source, target):
        """Estimate a transformation from source to target.

        Parameters
        ----------
        source
            Source object.
        target
            Target object.

        Returns
        -------
        self : BaseRegistration
            Fitted registration.
        """

    @abstractmethod
    def transform(self, source):
        """Apply the estimated transformation.

        Parameters
        ----------
        source
            Object to transform.

        Returns
        -------
        transformed
            Transformed object.
        """

    def fit_transform(self, source, target):
        """Estimate and apply the transformation to the source.

        Parameters
        ----------
        source
            Source object.
        target
            Target object.

        Returns
        -------
        transformed
            Source transformed toward the target.
        """
        return self.fit(source, target).transform(source)

    def __call__(self, source, target):
        """Estimate and apply the transformation to the source."""
        return self.fit_transform(source, target)

    def against_same_target(self, target):
        """Create a registration against a fixed target.

        Parameters
        ----------
        target
            Common registration target.

        Returns
        -------
        registration : CommonTargetRegistration
            Registration callable with the target fixed.
        """
        return CommonTargetRegistration(self, target)


class CommonTargetRegistration:
    """Register multiple sources against a common target.

    The same pairwise registration algorithm is fitted independently for
    each source against a fixed target.

    Parameters
    ----------
    registration : BaseRegistration
        Pairwise registration algorithm.
    target
        Common registration target.
    """

    def __init__(self, registration, target):
        self.registration = registration
        self.target = target

    def __call__(self, sources):
        """Register multiple sources against the common target.

        Parameters
        ----------
        sources : iterable
            Source objects to register.

        Returns
        -------
        transformed : list
            Sources registered independently against the common target.
        """
        return [self.registration(source, self.target) for source in sources]
