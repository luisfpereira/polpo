import json


def load_json(path):
    with open(path) as file:
        return json.load(file)


def dump_json(path, data, indent=2):
    with open(path, "w") as file:
        json.dump(data, file, indent=indent)


def check_json_serializable(data):
    """Check that an object can be serialized to JSON.

    Parameters
    ----------
    data : object
        Object to validate.

    Raises
    ------
    TypeError
        If `data` is not JSON-serializable.
    """
    try:
        json.dumps(data)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Object must be JSON-serializable: {error}") from error
