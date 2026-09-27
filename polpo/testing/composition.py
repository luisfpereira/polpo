class CompositeGeometricCaseData:
    def __init__(self, *components):
        self.components = []

        for component in components:
            self.add_component(component)

    def __add__(self, other):
        return CompositeGeometricCaseData(self, other)

    def add_component(self, component):
        if isinstance(component, CompositeGeometricCaseData):
            for component_ in component.components:
                self.add_component(component_)
            return

        self.components.append(component)

    def propagate(self, name, value):
        matched = False

        for component in self.components:
            if hasattr(component, name):
                setattr(component, name, value)
                matched = True

        if not matched:
            raise AttributeError(f"No component has attribute {name!r}.")

    def get_decorators(self):
        decorators = []
        seen = set()

        for component in self.components:
            for decorator in component.get_decorators():
                if decorator in seen:
                    continue

                seen.add(decorator)
                decorators.append(decorator)

        return decorators

    def get_data_methods(self):
        return self._merge_methods(
            component.get_data_methods() for component in self.components
        )

    def get_vectorization_data_methods(self):
        return self._merge_methods(
            component.get_vectorization_data_methods() for component in self.components
        )

    @staticmethod
    def _merge_methods(method_groups):
        methods = {}

        for group in method_groups:
            overlap = methods.keys() & group.keys()
            if overlap:
                raise ValueError(f"Duplicate test data methods: {sorted(overlap)}.")

            methods.update(group)

        return methods


class MarkedTestData:
    def __init__(self, testing_data, marks):
        self.testing_data = testing_data
        self.marks = marks

    def __getattr__(self, name):
        return getattr(self.testing_data, name)

    def __setattr__(self, name, value):
        if name in {"testing_data", "marks"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.testing_data, name, value)

    def get_data_methods(self):
        data_methods = self.testing_data.get_data_methods()

        return {name: self._mark(method) for name, method in data_methods.items()}

    def _mark(self, method):
        func = method.__func__

        for mark in self.marks:
            func = mark(func)

        return func.__get__(method.__self__, type(method.__self__))


class MarkedGeometricTestData(MarkedTestData):
    def get_vectorization_data_methods(self):
        return self._mark_methods(self.testing_data.get_vectorization_data_methods())
