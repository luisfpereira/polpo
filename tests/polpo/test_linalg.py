import numpy as np

from polpo.linalg import (
    get_diag_blocks_by_size,
    permute_by_row_norm,
    sym_to_triu_vec,
    triu_vec_to_sym,
)
from polpo.testing.data import CaseData
from polpo.testing.parametrizers import DataBasedParametrizer


class LinalgTestData(CaseData):
    def sym_to_triu_vec_after_triu_vec_to_sym_test_data(self):
        return [
            dict(vec=np.arange(6), includes_diag=includes_diag)
            for includes_diag in (False, True)
        ]


class TestLinalg(metaclass=DataBasedParametrizer):
    testing_data = LinalgTestData()

    def test_sym_to_triu_vec_after_triu_vec_to_sym(self, vec, includes_diag):
        mat = triu_vec_to_sym(vec, includes_diag=includes_diag)
        result = sym_to_triu_vec(mat, k=not includes_diag)

        np.testing.assert_array_equal(result, vec)


def test_get_diag_blocks_by_size():
    mat = np.arange(16).reshape(4, 4)

    blocks = get_diag_blocks_by_size(mat, sizes=[1, 3])

    np.testing.assert_array_equal(blocks[0], [[0]])
    np.testing.assert_array_equal(
        blocks[1],
        [[5, 6, 7], [9, 10, 11], [13, 14, 15]],
    )


def test_permute_by_row_norm():
    mat = np.array(
        [
            [1, 2, 0],
            [2, 0, 0],
            [0, 0, 3],
        ]
    )

    result, indices = permute_by_row_norm(mat)

    np.testing.assert_array_equal(indices, [2, 0, 1])
    np.testing.assert_array_equal(
        result,
        [[3, 0, 0], [0, 1, 2], [0, 2, 0]],
    )
