class PrefixDispatcher:
    def __init__(self):
        self.registry = {}

    def register(self, prefix):
        def decorator(func):
            self.registry[prefix] = func
            return func

        return decorator

    def dispatch(self, value):
        matches = [
            (prefix, func)
            for prefix, func in self.registry.items()
            if value.startswith(prefix)
        ]

        if not matches:
            raise ValueError(f"No implementation registered for {value!r}.")

        prefix, func = max(matches, key=lambda item: len(item[0]))
        return func

    def __call__(self, *args, **kwargs):
        value = kwargs[self.arg_name]
        return self.dispatch(value)(*args, **kwargs)
