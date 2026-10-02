from polpo.utils import index_to_letters


class NestedKeyMap:
    """Encode and keys of a nested dataset.

    A nested key codec stores reversible mappings for both levels of a
    nested dataset. Outer keys are mapped globally, while inner-key
    encodings may depend on the corresponding outer key.

    Parameters
    ----------
    outer : dict
        Mapping from outer keys to mapped outer keys.
    inner : dict
        Mapping from outer keys to mappings from inner keys to mapped
        inner keys.
    """

    def __init__(self, outer, inner):
        self.outer = outer
        self.inner = inner

    def invert(self):
        """Return the inverse mapping."""
        outer = {code: key for key, code in self.outer.items()}

        inner = {
            self.outer[outer_key]: {code: key for key, code in inner_map.items()}
            for outer_key, inner_map in self.inner.items()
        }

        return self.__class__(outer=outer, inner=inner)

    def chain_with(self, other):
        """Return the key map obtained by applying this map then ``other``."""
        intermediate = self.invert()
        domain = intermediate._common_domain(other)

        outer = {}
        inner = {}

        for outer_key, middle_outer_key in self.outer.items():
            if middle_outer_key not in domain.outer:
                continue

            inner_map = {
                inner_key: other.inner[middle_outer_key][middle_inner_key]
                for inner_key, middle_inner_key in self.inner[outer_key].items()
                if middle_inner_key in other.inner[middle_outer_key]
            }

            if inner_map:
                outer[outer_key] = other.outer[middle_outer_key]
                inner[outer_key] = inner_map

        return type(self)(
            outer=outer,
            inner=inner,
        )

    def _domain_is_subset(self, other):
        if not self.outer.keys() <= other.outer.keys():
            return False

        return all(
            self.inner[outer_key].keys() <= other.inner[outer_key].keys()
            for outer_key in self.outer
        )

    def _common_domain(self, other):
        if self._domain_is_subset(other):
            return self

        if other._domain_is_subset(self):
            return other

        raise ValueError("Key map domains are incompatible.")

    def with_inner(self, other):
        """Return a key map using this outer mapping and another inner mapping."""
        domain = self._common_domain(other)

        return self.__class__(
            outer={outer_key: self.outer[outer_key] for outer_key in domain.outer},
            inner={
                outer_key: {
                    inner_key: other.inner[outer_key][inner_key]
                    for inner_key in domain.inner[outer_key]
                }
                for outer_key in domain.outer
            },
        )

    def with_outer(self, other):
        """Return a key map using another outer mapping and this inner mapping."""
        domain = self._common_domain(other)

        return self.__class__(
            outer={outer_key: other.outer[outer_key] for outer_key in domain.outer},
            inner={
                outer_key: {
                    inner_key: self.inner[outer_key][inner_key]
                    for inner_key in domain.inner[outer_key]
                }
                for outer_key in domain.outer
            },
        )

    @classmethod
    def from_dataset(
        cls,
        nested_dataset,
        outer_map=None,
        inner_map=None,
    ):
        """Create a codec from the keys of a nested dataset.

        Parameters
        ----------
        nested_dataset : mapping
            Nested mapping whose outer and inner keys are mapped.
        outer_map : callable
            Function mapping ``(index, outer_key)`` to an mapped outer key.
            By default, outer keys are mapped as uppercase letters.
        inner_map : callable
            Function mapping ``(index, outer_key, inner_key)`` to an mapped
            inner key. By default, inner keys are mapped by their index.

        Returns
        -------
        NestedKeyCodec
            Codec built from the keys of ``nested_dataset``.
        """
        if outer_map is None:
            outer_map = lambda index, outer_key: index_to_letters(index)

        if inner_map is None:
            inner_map = lambda index, outer_key, inner_key: index

        outer = {
            outer_key: outer_map(index, outer_key)
            for index, outer_key in enumerate(nested_dataset)
        }

        inner = {
            outer_key: {
                inner_key: inner_map(index, outer_key, inner_key)
                for index, inner_key in enumerate(inner_dict)
            }
            for outer_key, inner_dict in nested_dataset.items()
        }

        return cls(outer, inner)

    @classmethod
    def from_inner_key_map(cls, inner):
        """Create a codec that only encodes inner keys.

        Outer keys are mapped to themselves.

        Parameters
        ----------
        inner : dict
            Mapping from outer keys to mappings from inner keys to mapped
            inner keys.

        Returns
        -------
        NestedKeyCodec
            Codec with identity encoding for outer keys.
        """
        outer = {outer_key: outer_key for outer_key in inner}

        return cls(
            outer=outer,
            inner=inner,
        )

    def map_outer(self, outer_key):
        """Encode an outer key.

        Parameters
        ----------
        outer_key : hashable
            Outer key to encode.

        Returns
        -------
        hashable
            mapped outer key.
        """
        return self.outer[outer_key]

    def map_inner(self, outer_key, inner_key):
        """Encode an inner key within an outer key.

        Parameters
        ----------
        outer_key : hashable
            Outer key identifying the inner key map.
        inner_key : hashable
            Inner key to encode.

        Returns
        -------
        hashable
            mapped inner key.
        """
        return self.inner[outer_key][inner_key]

    def map(self, outer_key, inner_key):
        """Encode a nested key.

        Parameters
        ----------
        outer_key : hashable
            Outer key to encode.
        inner_key : hashable
            Inner key to encode.

        Returns
        -------
        tuple
            mapped ``(outer_key, inner_key)`` pair.
        """
        return (
            self.map_outer(outer_key),
            self.map_inner(outer_key, inner_key),
        )

    def map_keys(self, nested_keys):
        """Encode a collection of nested keys.

        Parameters
        ----------
        nested_keys : mapping
            Mapping from outer keys to iterables of inner keys.

        Returns
        -------
        dict
            Mapping from mapped outer keys to mapped inner keys.
        """
        return {
            self.map_outer(outer_key): [
                self.map_inner(outer_key, inner_key) for inner_key in inner_keys
            ]
            for outer_key, inner_keys in nested_keys.items()
        }

    def __call__(self, outer_key, inner_key):
        """Encode a nested key."""
        return self.map(outer_key, inner_key)

    def to_dict(self):
        """Return the codec mappings as a dictionary.

        Returns
        -------
        dict
            Dictionary containing the outer and inner key mappings.
        """
        return {
            "outer": self.outer,
            "inner": self.inner,
        }

    def domain_keys(self):
        """Return the nested keys represented by the codec.

        Returns
        -------
        dict
            Mapping from outer keys to tuples of corresponding inner keys.
        """
        return {
            outer_key: tuple(inner_map) for outer_key, inner_map in self.inner.items()
        }

    def image_keys(self):
        """Return the nested keys represented by the codec.

        Returns
        -------
        dict
            Mapping from outer keys to tuples of corresponding inner keys.
        """
        return {
            self.map_outer(outer_key): tuple(inner_map.values())
            for outer_key, inner_map in self.inner.items()
        }


class MappedView:
    def __init__(self, obj, key_map, include=None, exclude=None):
        self._obj = obj
        self.key_map = key_map
        self._include = include
        self._exclude = exclude or set()

    def __getattr__(self, name):
        attr = getattr(self._obj, name)

        should_map = (
            self._include is None or name in self._include
        ) and name not in self._exclude

        if callable(attr):

            def wrapped(*args, **kwargs):
                result = attr(*args, **kwargs)
                if should_map and hasattr(result, "map_keys"):
                    return result.map_keys(self.key_map)
                return result

            return wrapped

        if should_map and hasattr(attr, "map_keys"):
            return attr.map_keys(self.key_map)

        return attr

    def with_key_map(self, key_map):
        if self.key_map is not None:
            key_map = self.key_map.chain_with(key_map)
        return type(self)(
            self._obj, key_map, include=self._include, exclude=self._exclude
        )
