import functools
import inspect


class PrefixDispatcher:
    """Dispatch function calls according to string prefixes.

    Parameters
    ----------
    func : callable
        Default function used when no registered prefix matches.
    dispatch_on : str
        Name of the argument whose value determines dispatch.

    Notes
    -----
    Registered implementations are selected by matching prefixes against
    the dispatch argument. If multiple prefixes match, the longest matching
    prefix is used.
    """

    def __init__(self, func, dispatch_on):
        self.func = func
        self.dispatch_on = dispatch_on
        self.signature = inspect.signature(func)
        self.registry = {}

        functools.update_wrapper(self, func)

    def register(self, prefix):
        """Register an implementation for a prefix.

        Parameters
        ----------
        prefix : str
            Prefix used to select the implementation.

        Returns
        -------
        decorator : callable
            Decorator registering a function for ``prefix``.
        """

        def decorator(func):
            self.registry[prefix] = func
            return func

        return decorator

    def dispatch(self, value):
        """Return the implementation matching a value.

        Parameters
        ----------
        value : str
            Value used for prefix matching.

        Returns
        -------
        func : callable
            Registered implementation with the longest matching prefix,
            or the default function if no prefix matches.
        """
        matches = [
            (prefix, func)
            for prefix, func in self.registry.items()
            if value.startswith(prefix)
        ]

        if not matches:
            return self.func

        return max(matches, key=lambda item: len(item[0]))[1]

    def __call__(self, *args, **kwargs):
        """Dispatch and call the matching implementation.

        The dispatch argument is consumed by the dispatcher and is not
        forwarded to registered implementations.
        """
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()

        value = bound.arguments[self.dispatch_on]
        func = self.dispatch(value)

        if func is self.func:
            return func(*args, **kwargs)

        del bound.arguments[self.dispatch_on]
        return func(*bound.args, **bound.kwargs)


def prefixdispatch(dispatch_on):
    """Create a prefix-based function dispatcher.

    Parameters
    ----------
    dispatch_on : str
        Name of the function argument used for prefix-based dispatch.

    Returns
    -------
    decorator : callable
        Decorator converting a function into a ``PrefixDispatcher``.
    """

    def decorator(func):
        return PrefixDispatcher(func, dispatch_on)

    return decorator
