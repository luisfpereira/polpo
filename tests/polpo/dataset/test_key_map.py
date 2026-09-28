import pytest

from polpo.dataset._key_map import NestedKeyMap


@pytest.fixture
def key_map():
    return NestedKeyMap(
        outer={"a": "A", "b": "B"},
        inner={
            "a": {"x": 0, "y": 1},
            "b": {"z": 0},
        },
    )


def test_map(key_map):
    assert key_map("a", "x") == ("A", 0)


def test_map_outer(key_map):
    assert key_map.map_outer("a") == "A"


def test_map_inner(key_map):
    assert key_map.map_inner("a", "y") == 1


def test_map_keys(key_map):
    keys = {
        "a": ["x", "y"],
        "b": ["z"],
    }

    result = key_map.map_keys(keys)

    assert result == {
        "A": [0, 1],
        "B": [0],
    }


def test_domain_keys(key_map):
    assert key_map.domain_keys() == {
        "a": ("x", "y"),
        "b": ("z",),
    }


def test_image_keys(key_map):
    assert key_map.image_keys() == {
        "A": (0, 1),
        "B": (0,),
    }


def test_invert(key_map):
    inverse = key_map.invert()

    assert inverse("A", 0) == ("a", "x")
    assert inverse("A", 1) == ("a", "y")
    assert inverse("B", 0) == ("b", "z")


def test_invert_is_involution(key_map):
    inverse = key_map.invert().invert()

    assert inverse.outer == key_map.outer
    assert inverse.inner == key_map.inner


def test_chain_with():
    first = NestedKeyMap(
        outer={"a": "A"},
        inner={"a": {"x": 0}},
    )
    second = NestedKeyMap(
        outer={"A": "alpha"},
        inner={"A": {0: "i"}},
    )

    chained = first.chain_with(second)

    assert chained("a", "x") == ("alpha", "i")


def test_chain_with_incompatible_domains():
    first = NestedKeyMap(
        outer={"a": "A"},
        inner={"a": {"x": 0}},
    )
    second = NestedKeyMap(
        outer={"B": "beta"},
        inner={"B": {0: "i"}},
    )

    with pytest.raises(ValueError):
        first.chain_with(second)


def test_with_outer():
    first = NestedKeyMap(
        outer={"a": "A"},
        inner={"a": {"x": 0}},
    )
    second = NestedKeyMap(
        outer={"a": "B"},
        inner={"a": {"x": 1}},
    )

    result = first.with_outer(second)

    assert result.outer == {"a": "B"}
    assert result.inner == {"a": {"x": 0}}


def test_with_inner():
    first = NestedKeyMap(
        outer={"a": "A"},
        inner={"a": {"x": 0}},
    )
    second = NestedKeyMap(
        outer={"a": "B"},
        inner={"a": {"x": 1}},
    )

    result = first.with_inner(second)

    assert result.outer == {"a": "A"}
    assert result.inner == {"a": {"x": 1}}


def test_from_dataset():
    dataset = {
        "subject_1": {"t0": None, "t1": None},
        "subject_2": {"t0": None},
    }

    key_map = NestedKeyMap.from_dataset(dataset)

    assert key_map.outer == {
        "subject_1": "A",
        "subject_2": "B",
    }
    assert key_map.inner == {
        "subject_1": {"t0": 0, "t1": 1},
        "subject_2": {"t0": 0},
    }


def test_from_dataset_with_maps():
    dataset = {
        "subject_1": {"t0": None},
    }

    key_map = NestedKeyMap.from_dataset(
        dataset,
        outer_map=lambda index, key: f"outer_{key}",
        inner_map=lambda index, outer_key, inner_key: f"{outer_key}_{inner_key}",
    )

    assert key_map("subject_1", "t0") == (
        "outer_subject_1",
        "subject_1_t0",
    )


def test_from_inner_key_map():
    inner = {
        "a": {"x": 0},
        "b": {"y": 1},
    }

    key_map = NestedKeyMap.from_inner_key_map(inner)

    assert key_map.outer == {"a": "a", "b": "b"}
    assert key_map.inner == inner
