class VarDef:
    def __init__(
        self,
        id_,
        min_value=None,
        max_value=None,
        default_value=None,
        name="",
        unit="",
    ):
        # TODO: add notion of short label?

        self.id = id_
        self.name = name
        self.unit = unit

        self.min_value = min_value
        self.max_value = max_value
        self._default_value = default_value

    @property
    def label(self):
        return f"{self.name} {self.unit}"

    @property
    def default_value(self):
        return self._default_value or self.min_value

    @default_value.setter
    def default_value(self, value):
        self._default_value = value

    def __repr__(self):
        return f"VarDef({self.id})"


class Var:
    # TODO: overkill
    def __init__(self, value):
        self.value = value
