import pytest

from polpo.workflow.lddmm_to_global.distances import (
    EuclideanDistances,
    PersistentEvaluator,
    VarifoldDistances,
)

from .fixtures import lddmm_output  # noqa: F401


@pytest.mark.smoke
def test_euclidean_distances(lddmm_output, tmp_path_factory):
    evaluator = PersistentEvaluator(
        EuclideanDistances(lddmm_output.path),
        tmp_path_factory.mktemp("post_dists_eucl"),
    ).run(overwrite=True, continue_on_error=False)

    assert evaluator.manifest.status == "completed"

    res = evaluator.results

    res.local_reconstruction_error()


@pytest.mark.smoke
@pytest.mark.redundant
def test_varifold_distances(lddmm_output, tmp_path_factory):
    evaluator = PersistentEvaluator(
        VarifoldDistances(lddmm_output.path),
        tmp_path_factory.mktemp("post_dists_var"),
    ).run(overwrite=True, continue_on_error=False)

    assert evaluator.manifest.status == "completed"
