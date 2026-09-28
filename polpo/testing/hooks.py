import pytest

from .execution import set_execution_key


def _make_hashable(value):
    try:
        hash(value)
    except TypeError:
        return id(value)

    return value


def _get_execution_key_from_item(item):
    callspec = getattr(item, "callspec", None)

    fixture_params = ()
    if callspec is not None:
        fixture_params = tuple(
            (name, _make_hashable(callspec.params[name]))
            for name, scope in callspec._arg2scope.items()
            if scope.value != "function"
        )

    repeat = 0
    if callspec is not None:
        repeat = callspec.params.get(
            "__pytest_repeat_step_number",
            0,
        )

    rerun = getattr(item, "execution_count", 1)

    return fixture_params, repeat, rerun


def pytest_runtest_call(item):
    set_execution_key(_get_execution_key_from_item(item))


def _has_ignore_marker(markers):
    for marker in markers:
        if marker.name == "ignore":
            return True

    return False


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    selected = []
    deselected = []
    for item in items:
        if _has_ignore_marker(item.own_markers):
            deselected.append(item)
        else:
            selected.append(item)

    config.hook.pytest_deselected(items=deselected)

    items[:] = selected
