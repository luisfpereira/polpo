from types import SimpleNamespace

import pytest

from polpo.dataset import Dataset, KeyMap, MappedView, NestedKeyEncoder, NestedKeyMap


@pytest.fixture
def key_map():
    return KeyMap({"a": "A", "b": "B"})


@pytest.fixture
def nested_key_map():
    return NestedKeyMap(
        outer={"a": "A", "b": "B"},
        inner={
            "a": {"x": 0, "y": 1},
            "b": {"z": 0},
        },
    )


class TestKeyMap:
    def test_map(self, key_map):
        assert key_map("a") == "A"

    def test_map_keys(self, key_map):
        assert key_map.map_keys(["a", "b"]) == ["A", "B"]

    def test_domain_keys(self, key_map):
        assert key_map.domain_keys() == ("a", "b")

    def test_image_keys(self, key_map):
        assert key_map.image_keys() == ("A", "B")

    def test_invert(self, key_map):
        inverse = key_map.invert()

        assert inverse("A") == "a"
        assert inverse("B") == "b"

    def test_invert_is_involution(self, key_map):
        inverse = key_map.invert().invert()

        assert inverse.mapping == key_map.mapping

    def test_chain_with(self):
        first = KeyMap({"a": "A"})
        second = KeyMap({"A": 1})

        chained = first.chain_with(second)

        assert chained("a") == 1


class TestNestedKeyMap:
    def test_map(self, nested_key_map):
        assert nested_key_map("a", "x") == ("A", 0)

    def test_map_outer(self, nested_key_map):
        assert nested_key_map.map_outer("a") == "A"

    def test_map_inner(self, nested_key_map):
        assert nested_key_map.map_inner("a", "y") == 1

    def test_map_keys(self, nested_key_map):
        keys = {
            "a": ["x", "y"],
            "b": ["z"],
        }

        result = nested_key_map.map_keys(keys)

        assert result == {
            "A": [0, 1],
            "B": [0],
        }

    def test_domain_keys(self, nested_key_map):
        assert nested_key_map.domain_keys() == {
            "a": ("x", "y"),
            "b": ("z",),
        }

    def test_image_keys(self, nested_key_map):
        assert nested_key_map.image_keys() == {
            "A": (0, 1),
            "B": (0,),
        }

    def test_invert(self, nested_key_map):
        inverse = nested_key_map.invert()

        assert inverse("A", 0) == ("a", "x")
        assert inverse("A", 1) == ("a", "y")
        assert inverse("B", 0) == ("b", "z")

    def test_invert_is_involution(self, nested_key_map):
        inverse = nested_key_map.invert().invert()

        assert inverse.outer == nested_key_map.outer
        assert inverse.inner == nested_key_map.inner

    def test_chain_with(self):
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

    def test_chain_with_incompatible_domains(self):
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

    def test_with_outer(self):
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

    def test_with_inner(self):
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

    def test_from_inner_key_map(self):
        inner = {
            "a": {"x": 0},
            "b": {"y": 1},
        }

        key_map = NestedKeyMap.from_inner_key_map(inner)

        assert key_map.outer == {"a": "a", "b": "b"}
        assert key_map.inner == inner

    def test_complete(self):
        key_map = NestedKeyMap(
            outer={"a": "A"},
            inner={"a": {"x": 0}},
        )

        result = key_map.complete(
            {
                "a": ["x", "y"],
                "b": ["z"],
            }
        )

        assert result.outer == {
            "a": "A",
            "b": "b",
        }
        assert result.inner == {
            "a": {
                "x": 0,
                "y": "y",
            },
            "b": {
                "z": "z",
            },
        }

    def test_flatten(self, nested_key_map):
        result = nested_key_map.flatten()

        assert result.mapping == {
            ("a", "x"): ("A", 0),
            ("a", "y"): ("A", 1),
            ("b", "z"): ("B", 0),
        }


def test_nested_key_encoder_default():
    nested_keys = {
        "subject_1": ["week_1", "week_2"],
        "subject_2": ["week_3"],
    }

    key_map = NestedKeyEncoder()(nested_keys)

    assert key_map.outer == {
        "subject_1": "A",
        "subject_2": "B",
    }
    assert key_map.inner == {
        "subject_1": {
            "week_1": 0,
            "week_2": 1,
        },
        "subject_2": {
            "week_3": 0,
        },
    }


class TestMappedView:
    def test_mapped_view(self):
        dataset = Dataset({"a": 1, "b": 2})
        obj = SimpleNamespace(
            dataset=dataset,
            get_dataset=lambda: dataset,
        )

        key_map = KeyMap({"a": "A", "b": "B"})
        view = MappedView(obj, key_map)

        assert view.dataset["A"] == dataset["a"]
        assert view.get_dataset()["B"] == dataset["b"]

    def test_mapped_view_with_key_map(self):
        dataset = Dataset({"a": 1, "b": 2})
        obj = SimpleNamespace(dataset=dataset)

        view = MappedView(obj, KeyMap({"a": "A", "b": "B"}))
        view = view.with_key_map(KeyMap({"A": 0, "B": 1}))

        assert view.dataset[0] == dataset["a"]
        assert view.dataset[1] == dataset["b"]
