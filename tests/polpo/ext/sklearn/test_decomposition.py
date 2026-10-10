import numpy as np
import pytest
from sklearn.decomposition import PCA

from polpo.ext.sklearn.decomposition import RandomizedSVD, TruncatedPCA


def test_truncated_pca():
    X = np.random.default_rng(0).normal(size=(20, 5))
    pca = PCA(n_components=5).fit(X)

    truncated = TruncatedPCA.from_fitted(pca, n_components=2)

    scores = pca.transform(X)[:, :2]

    assert truncated.n_components_ == 2
    assert pca.n_components_ == 5
    np.testing.assert_allclose(truncated.transform(X), scores)
    np.testing.assert_allclose(
        truncated.inverse_transform(scores),
        scores @ pca.components_[:2] + pca.mean_,
    )


@pytest.mark.smoke
def test_randomized_svd():
    X = np.random.default_rng(0).normal(size=(10, 5))

    U, s, Vt = RandomizedSVD(n_components=2, random_state=0)(X)

    assert U.shape == (10, 2)
    assert s.shape == (2,)
    assert Vt.shape == (2, 5)
