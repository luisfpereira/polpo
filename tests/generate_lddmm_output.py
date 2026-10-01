import string
from pathlib import Path

from polpo.dataset import NestedDataset
from polpo.surface_mesh.core import Surface
from polpo.surface_mesh.generation.blob import create_blob
from polpo.workflow.lddmm_to_global import LddmmToGlobal, LddmmToGlobalOutput


def generate_lddmm_output(path):
    """Generate an LDDMM-to-global output for testing."""
    data = {}
    for subj_index, (n_meshes, bump_amp, n_bumps) in enumerate(
        zip((3, 2, 4), (0.2, 0.3, 0.4), (3, 5, 6))
    ):
        data[string.ascii_uppercase[subj_index + 3]] = {
            index + 2: Surface.from_polydata(
                create_blob(
                    resolution=10,
                    bump_amp=bump_amp,
                    n_bumps=n_bumps,
                    smoothing_iter=10,
                )
            )
            for index in range(n_meshes)
        }

    dataset = NestedDataset(data)

    atlas_keys = {
        "D": [2, 3],
        "E": [2],
        "F": [2, 3],
    }
    atlas_only_keys = {"D": [3]}

    protocol = LddmmToGlobal(
        known_correspondences=True,
        results_dir=path,
    )
    protocol.run(
        dataset,
        atlas_keys=atlas_keys,
        atlas_only_keys=atlas_only_keys,
    )

    return LddmmToGlobalOutput(path)


if __name__ == "__main__":
    outputs_dir = Path(__file__).parent / ".test_data" / "lddmm"

    generate_lddmm_output(outputs_dir)
