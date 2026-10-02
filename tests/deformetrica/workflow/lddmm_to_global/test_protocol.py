import pytest

from polpo.workflow.lddmm_to_global import (
    LddmmToGlobalMultiOutput,
)
from polpo.workflow.lddmm_to_global.distances import (
    EuclideanDistances,
    LddmmDistances,
    MultiDistanceResults,
    PersistentEvaluator,
)


@pytest.fixture(scope="session")
def lddmm_output(tmp_path_factory):
    from tests.generate_lddmm_output import generate_lddmm_output

    tmp_path = tmp_path_factory.mktemp("lddmm")

    return generate_lddmm_output(tmp_path)


@pytest.mark.slow
@pytest.mark.smoke
def test_runs(lddmm_output):
    pass


@pytest.mark.slow
@pytest.mark.smoke
def test_output_view(lddmm_output):
    view = lddmm_output.view

    view.dataset
    view.local_atlases
    view.local_registrations
    view.local_reconstructed_points
    view.registrations_to_global_atlas
    view.global_shoots
    view.global_points
    view.global_deltas
    view.transports
    view.global_atlas
    view.global_atlas_point
    view.global_atlas_flows


@pytest.mark.slow
@pytest.mark.smoke
def test_multi_output_view(lddmm_output):
    lddmm_multi_output = LddmmToGlobalMultiOutput([lddmm_output, lddmm_output])

    view = lddmm_multi_output.view

    view.dataset
    view.local_reconstructed_points
    view.global_points


@pytest.mark.slow
@pytest.mark.smoke
def test_euclidean_distances(lddmm_output):
    evaluator = PersistentEvaluator(
        EuclideanDistances(lddmm_output.path),
        "post_dists_eucl",
    ).run(overwrite=True, continue_on_error=False)

    assert evaluator.manifest.status == "completed"

    res = evaluator.results

    res.local_reconstruction_error()


@pytest.mark.slow
@pytest.mark.smoke
def test_lddmm_distances(lddmm_output):
    evaluator = LddmmDistances(lddmm_output.path)

    evaluator.local_atlas_fit_error()
    evaluator.global_atlas_fit_error()
    evaluator.local_atlas_distance()
    evaluator.global_atlas_distance()


@pytest.mark.slow
@pytest.mark.smoke
@pytest.mark.redundant
def test_lddmm_distances_all(lddmm_output):
    evaluator = PersistentEvaluator(
        LddmmDistances(lddmm_output.path),
        "post_dists_lddmm",
    ).run(overwrite=True, continue_on_error=False)

    assert evaluator.manifest.status == "completed"


@pytest.mark.slow
@pytest.mark.smoke
def test_multi_distances(lddmm_output):
    evaluator = EuclideanDistances(lddmm_output.path)

    multi = MultiDistanceResults({"left": evaluator, "right": evaluator})

    multi.local_reconstruction_error()
    multi.global_pairwise()


@pytest.mark.slow
@pytest.mark.smoke
def test_multi_distances_with_persisted(lddmm_output):
    evaluator = PersistentEvaluator(
        EuclideanDistances(lddmm_output.path),
        "post_dists_eucl",
    ).run(overwrite=True, continue_on_error=False)

    res = evaluator.results

    multi = MultiDistanceResults({"left": res, "right": res})

    multi.local_reconstruction_error()
    multi.global_pairwise()
