import numpy as np

_KEY_ENCODERS = {
    str: str,
    int: str,
    float: repr,
    bool: lambda value: "true" if value else "false",
}

_KEY_DECODERS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": lambda value: value == "true",
}

_KEY_TYPE_NAMES = {key_type: key_type.__name__ for key_type in _KEY_ENCODERS}


def _encode_keys(keys):
    """Encode keys for storage in a NumPy archive.

    Parameters
    ----------
    keys : iterable
        Scalar or tuple keys. All keys must have the same structure and
        component types. Supported component types are ``str``, ``int``,
        ``float``, and ``bool``.

    Returns
    -------
    encoded : dict
        Arrays containing the encoded keys and metadata required to restore
        their original types and structure.
    """
    keys = list(keys)

    if not keys:
        raise ValueError("Cannot save empty keys.")

    keys_are_tuples = isinstance(keys[0], tuple)
    normalized = [key if keys_are_tuples else (key,) for key in keys]

    arity = len(normalized[0])
    key_types = tuple(type(value) for value in normalized[0])

    for key in normalized:
        if len(key) != arity:
            raise ValueError("All tuple keys must have the same length.")

        if tuple(type(value) for value in key) != key_types:
            raise TypeError("All keys must have the same component types.")

    unsupported = [key_type for key_type in key_types if key_type not in _KEY_ENCODERS]
    if unsupported:
        names = ", ".join(key_type.__name__ for key_type in unsupported)
        raise TypeError(f"Unsupported key types: {names}.")

    encoded = [
        [_KEY_ENCODERS[key_type](value) for key_type, value in zip(key_types, key)]
        for key in normalized
    ]

    return {
        "keys": np.asarray(encoded, dtype=str),
        "key_types": np.asarray(
            [_KEY_TYPE_NAMES[key_type] for key_type in key_types],
            dtype=str,
        ),
        "keys_are_tuples": np.asarray(keys_are_tuples),
    }


def _decode_keys(archive):
    """Decode keys stored in a NumPy archive.

    Parameters
    ----------
    archive : mapping
        NumPy archive containing ``keys``, ``key_types``, and
        ``keys_are_tuples``.

    Returns
    -------
    keys : list
        Decoded scalar or tuple keys.
    """
    encoded_keys = archive["keys"].tolist()
    type_names = archive["key_types"].tolist()
    keys_are_tuples = bool(archive["keys_are_tuples"])

    decoders = [_KEY_DECODERS[type_name] for type_name in type_names]

    keys = [
        tuple(decoder(value) for decoder, value in zip(decoders, encoded_key))
        for encoded_key in encoded_keys
    ]

    if not keys_are_tuples:
        keys = [key[0] for key in keys]

    return keys


def save_indexed_array(path, keys, data):
    """Save an array together with keys indexing its first axis.

    Parameters
    ----------
    path : path-like
        Output ``.npz`` path.
    keys : iterable
        Keys identifying entries along the first axis of ``data``. Keys may
        be scalars or fixed-length tuples with supported component types.
    data : array-like
        Data to save. Its first axis is indexed by ``keys``.
    """
    encoded_keys = _encode_keys(keys)

    np.savez_compressed(
        path,
        **encoded_keys,
        data=data,
        representation="condensed",
        format_version=1,
    )


def load_indexed_array(path):
    """Load an array and its index keys from a NumPy archive.

    Parameters
    ----------
    path : path-like
        Input ``.npz`` path.

    Returns
    -------
    keys : list
        Keys indexing the first axis of ``data``.
    data : ndarray
        Stored data array.
    """
    with np.load(path, allow_pickle=False) as data:
        return _decode_keys(data), data["data"]


def save_dict_as_array(path, data):
    """Save a key-indexed mapping as an array and ordered keys.

    Parameters
    ----------
    path : path-like
        Output ``.npz`` path.
    data : mapping
        Mapping from keys to values. All values must have compatible shapes.
        Mapping insertion order determines the array indexing.
    """
    return save_indexed_array(
        path,
        keys=list(data),
        data=np.stack(list(data.values())),
    )


def load_dict(path):
    """Load a key-indexed array as a dictionary.

    Parameters
    ----------
    path : path-like
        Input ``.npz`` path.

    Returns
    -------
    data : dict
        Mapping from the stored keys to entries along the first axis of the
        stored array.
    """
    keys, data = load_indexed_array(path)
    return dict(zip(keys, data))


def save_indexed_arrays(path, keys, data):
    """Save multiple arrays together with keys indexing their first axis.

    Parameters
    ----------
    path : path-like
        Output ``.npz`` path.
    keys : iterable
        Keys identifying entries along the first axis of each array.
    data : dict
        Mapping from names to arrays. The first axis of each array is indexed
        by ``keys``.
    """
    encoded_keys = _encode_keys(keys)

    np.savez_compressed(
        path,
        **encoded_keys,
        **data,
        data_names=np.asarray(list(data), dtype=str),
        representation="condensed_mapping",
        format_version=1,
    )


def load_indexed_arrays(path):
    """Load multiple arrays and their common index keys.

    Parameters
    ----------
    path : path-like
        Input ``.npz`` path.

    Returns
    -------
    keys : list
        Keys indexing the first axis of the stored arrays.
    data : dict
        Mapping from names to stored arrays.
    """
    with np.load(path, allow_pickle=False) as archive:
        keys = _decode_keys(archive)
        data_names = archive["data_names"].tolist()

        data = {name: archive[name] for name in data_names}

    return keys, data
