import numpy as np
from scipy.sparse import diags

from polpo.ext.pymetis import partition_graph


def test_partition_graph():
    adj = diags(
        [np.ones(5), np.ones(5)],
        offsets=[-1, 1],
        shape=(6, 6),
        format="csr",
    )

    n_edge_cuts, labels = partition_graph(adj, n_parts=2, seed=0)

    assert len(labels) == 6
    assert set(labels) == {0, 1}
    assert n_edge_cuts == 1
