import abc


class PreprocessingStep(abc.ABC):
    """Preprocessing step."""

    # TODO: just call it step

    @abc.abstractmethod
    def __call__(self, data=None):
        """Apply step."""

    def __add__(self, other):
        if other is None:
            return self

        if isinstance(other, list):
            other = Pipeline(other)

        if isinstance(other, Pipeline):
            return other.__radd__(self)

        return Pipeline([self, other])

    def __radd__(self, other):
        if other is None:
            return self

        if isinstance(other, list):
            other = Pipeline(other)

        if isinstance(other, Pipeline):
            return other + self

        return Pipeline([other, self])


class IdentityStep(PreprocessingStep):
    def __call__(self, data=None):
        return data


class Pipeline(PreprocessingStep):
    def __init__(self, steps, data=None):
        super().__init__()
        self.steps = steps
        self.data = data

    def __call__(self, data=None):
        if self.data is not None:
            data = self.data

        out = data
        for step in self.steps:
            try:
                out = step(out)
            except Exception as e:
                e.args = (f"Failed in step '{step}':\n{e.args[0]}",)
                raise

        return out

    def load(self):
        return self.__call__()

    def _ignore_other(self, other):
        if other is None or isinstance(other, IdentityStep):
            return True

        return False

    def __add__(self, other):
        steps = self.steps.copy()
        if hasattr(other, "steps"):
            steps.extend(other.steps)
        elif isinstance(other, list):
            steps.extend(other)
        else:
            if not self._ignore_other(other):
                steps.append(other)

        return Pipeline(steps)

    def __radd__(self, other):
        steps = self.steps.copy()
        if hasattr(other, "steps"):
            steps = other.steps + steps
        elif isinstance(other, list):
            steps = other + steps
        else:
            if not self._ignore_other(other):
                steps = [other] + steps

        return Pipeline(steps)
