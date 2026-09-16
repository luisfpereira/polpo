from contextvars import ContextVar

_CURRENT_EXECUTION_KEY = ContextVar(
    "_CURRENT_EXECUTION_KEY",
    default=(0, 1),
)


def get_execution_key():
    return _CURRENT_EXECUTION_KEY.get()


def set_execution_key(key):
    _CURRENT_EXECUTION_KEY.set(key)
