from joblib import dump, load


def save_estimator(path, estimator):
    dump(estimator, path)


def load_estimator(path):
    return load(path)
