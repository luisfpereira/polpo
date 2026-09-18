import numpy as np

from polpo.surface_mesh.ops.topology import (
    compute_one_ring_neighbors,
    compute_vertex_adjacency,
)
from polpo.testing.data import DataCase
from polpo.testing.parametrizers import DataBasedParametrizer


class TopologyMethodsTestCase:
    def test_vertex_adjacency(self, faces, expected):
        adjacency = compute_vertex_adjacency(faces)

        np.testing.assert_array_equal(
            adjacency.toarray(),
            expected,
        )

    def test_one_ring_neighbors(self, faces, expected):
        neighbors = compute_one_ring_neighbors(faces)

        assert neighbors == expected


class TopologyMethodsTestData(DataCase):
    def vertex_adjacency_test_data(self):
        faces_0 = np.array([[0, 1, 2]])
        adj_0 = np.array(
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ]
        )

        faces_1 = np.array([[0, 1, 2], [1, 3, 2]])
        adj_1 = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 0],
            ]
        )
        return [
            (faces_0, adj_0),
            (faces_1, adj_1),
        ]

    def one_ring_neighbors_test_data(self):
        faces_0 = np.array([[0, 1, 2], [1, 3, 2]])
        adj_0 = [[1, 2], [0, 2, 3], [0, 1, 3], [1, 2]]
        return [(faces_0, adj_0)]


class TestTopologyMethods(TopologyMethodsTestCase, metaclass=DataBasedParametrizer):
    testing_data = TopologyMethodsTestData()
