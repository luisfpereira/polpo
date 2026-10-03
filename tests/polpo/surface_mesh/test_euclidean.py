import numpy as np

from polpo.surface_mesh import Surface
from polpo.surface_mesh.euclidean import EuclideanSurfaces


def test_dist_translation():
    faces = np.array([[0, 1, 2]])
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    translation = np.array([1.0, 2.0, 2.0])

    surface_a = Surface(vertices=vertices, faces=faces)
    surface_b = Surface(vertices=vertices + translation, faces=faces)

    space = EuclideanSurfaces(faces)

    result = space.metric.dist(surface_a, surface_b)

    expected = np.linalg.norm(translation)
    np.testing.assert_allclose(result, expected)
