import pytest

from polpo.utils import merge_dicts, nest_dict, unnest_dict


def test_nest_dict():
    flat = {"a/b/x": 1, "a/b/y": 2, "c/d/z": 3}
    expected = {"a": {"b": {"x": 1, "y": 2}}, "c": {"d": {"z": 3}}}

    assert nest_dict(flat) == expected
    assert nest_dict({}) == {}


def test_unnest_dict():
    nested = {"a": {"b": {"x": 1, "y": 2}}}

    assert unnest_dict(nested) == {"a/b/x": 1, "a/b/y": 2}
    assert unnest_dict(nested, sep=None) == {
        ("a", "b", "x"): 1,
        ("a", "b", "y"): 2,
    }


def test_merge_dicts():
    dicts = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]

    assert merge_dicts(dicts) == {"a": 1, "b": 3, "c": 4}

    with pytest.raises(ValueError, match="Duplicate keys"):
        merge_dicts(dicts, check_duplicates=True)
