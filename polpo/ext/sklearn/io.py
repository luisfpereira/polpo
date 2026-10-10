from joblib import dump, load


def save_estimator(path, estimator):
    """Save an estimator using joblib.

    Parameters
    ----------
    path : path-like
        Destination file path.
    estimator : object
        Estimator to serialize.
    """
    dump(estimator, path)


def load_estimator(path):
    """Load an estimator saved using joblib.

    Parameters
    ----------
    path : path-like
        Path to the serialized estimator.

    Returns
    -------
    estimator : object
        Deserialized estimator.
    """
    return load(path)
