import numpy as np
from sklearn.decomposition import PCA

from polpo.ext.sklearn.reconstruction import Reconstructor


def test_reconstructor():
    X = np.array([[0, 0], [1, 2], [2, 4], [3, 6]])
    transformer = PCA(n_components=1)
    reconstructor = Reconstructor(transformer)

    assert reconstructor.fit(X) is reconstructor
    assert reconstructor.transformer_ is not transformer

    np.testing.assert_allclose(reconstructor.predict(X), X, atol=1e-12)
