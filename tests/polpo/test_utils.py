from types import SimpleNamespace

import pytest

from polpo.utils import (
    expand_path_names,
    get_first,
    has_package,
    index_to_letters,
    merge_dicts,
    nest_dict,
    params_to_kwargs,
    unnest_dict,
    unnest_list,
)


def test_get_first():
    assert get_first([1, 2, 3]) == 1
    assert get_first(iter([1, 2, 3])) == 1
    assert get_first({"a": 1, "b": 2}) == 1


def test_unnest_list():
    data = [[1, 2], [], [3, [4, 5]]]

    assert unnest_list(data) == [1, 2, 3, [4, 5]]


def test_params_to_kwargs():
    obj = SimpleNamespace(a=1, b=2, _cache=3)

    assert params_to_kwargs(obj) == {"a": 1, "b": 2, "_cache": 3}

    assert params_to_kwargs(
        obj,
        ignore=("b",),
        renamings={"a": "x"},
        ignore_private=True,
    ) == {"x": 1}

    assert params_to_kwargs(obj, func=lambda a, b: None) == {"a": 1, "b": 2}

    assert obj.__dict__ == {"a": 1, "b": 2, "_cache": 3}


def test_expand_path_names(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    missing = tmp_path / "missing.csv"

    a.touch()
    b.touch()

    result = expand_path_names(
        [
            str(tmp_path / "*.txt"),
            str(a),
            str(missing),
            str(tmp_path / "*.nii"),
        ]
    )

    assert set(result) == {a, b, missing}
    assert len(result) == 3
    assert result[-1] == missing


def test_has_package():
    assert has_package("os")
    assert not has_package("nonexistent_package_12345")


def test_index_to_letters():
    assert index_to_letters(0) == "A"
    assert index_to_letters(25) == "Z"
    assert index_to_letters(26) == "AA"
    assert index_to_letters(27) == "AB"
    assert index_to_letters(701) == "ZZ"
    assert index_to_letters(702) == "AAA"


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


def test_nest_dict_tuple_keys():
    flat = {("a", "b", "x"): 1, ("a", "b", "y"): 2}
    expected = {"a": {"b": {"x": 1, "y": 2}}}

    assert nest_dict(flat) == expected
    assert nest_dict(unnest_dict(expected, sep=None)) == expected


def test_merge_dicts():
    dicts = [{"a": 1, "b": 2}, {"b": 3, "c": 4}]

    assert merge_dicts(dicts) == {"a": 1, "b": 3, "c": 4}

    with pytest.raises(ValueError, match="Duplicate keys"):
        merge_dicts(dicts, check_duplicates=True)
