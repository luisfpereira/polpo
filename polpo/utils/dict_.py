# TODO: move to utils folder

import random


def extract_random_key(data):
    return random.choice(list(data.keys()))


def unnest_dict(nested_dict, sep="/", current_key=None, flat_dict=None):
    if isinstance(sep, str):
        prefix = "" if current_key is None else f"{current_key}{sep}"

        def _update_key(key):
            return f"{prefix}{key}"

    else:
        prefix = () if current_key is None else current_key

        def _update_key(key):
            return prefix + (key,)

    if flat_dict is None:
        flat_dict = {}

    for key, value in nested_dict.items():
        new_key = _update_key(key)

        if not isinstance(value, dict):
            flat_dict[new_key] = value
        else:
            flat_dict = unnest_dict(
                value, sep=sep, current_key=new_key, flat_dict=flat_dict
            )

    return flat_dict


def nest_dict_inner_level(flat_dict, sep="/"):
    nested_dict = {}

    for key, value in flat_dict.items():
        if isinstance(key, str):
            outer_key, inner_key = key.rsplit(sep, maxsplit=1)
        else:
            outer_key, inner_key = key

        inner_dict = nested_dict[outer_key] = nested_dict.get(outer_key, {})
        inner_dict[inner_key] = value

    return nested_dict


def nest_dict(flat_dict, sep="/"):
    while True:
        # TODO: make a nicer recursion

        try:
            flat_dict = nest_dict_inner_level(flat_dict, sep=sep)
        except ValueError:
            # when unpack error is raised
            break

    return flat_dict


def extract_unique_key_nested(data):
    # TODO: rethink function name
    if not isinstance(data, dict):
        return data

    if len(data.keys()) == 1:
        return extract_unique_key_nested(data[next(iter(data))])

    return {key: extract_unique_key_nested(value) for key, value in data.items()}


def rekey_nested_dict(nested_dict, outer_map, inner_maps):
    return {
        outer_map.get(outer_key, outer_key): {
            inner_maps[outer_key].get(inner_key, inner_key): inner_value
            for inner_key, inner_value in inner.items()
        }
        for outer_key, inner in nested_dict.items()
    }


def invert_dict(dict_):
    return dict(zip(dict_.values(), dict_.keys()))


def merge_dicts(dicts, *, check_duplicates=False):
    result = {}

    for dict_ in dicts:
        if check_duplicates:
            duplicates = result.keys() & dict_.keys()
            if duplicates:
                raise ValueError(f"Duplicate keys: {sorted(duplicates)}")

        result.update(dict_)

    return result
