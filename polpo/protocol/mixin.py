import polpo.utils as putils
from polpo.distmat import PairwiseDistances
from polpo.ext.pyvista.surface_mesh import PvSurface
from polpo.pipeline.mesh.registration import RigidAlignment


# TODO: fully remove
class RigidAlignmentMixin:
    def preprocess_meshes(self, data):
        # data : polpo.dataset.Dataset

        # rigidly aligns all the meshes against a randomly chosen target
        self.timer.start("prep")

        key = putils.extract_random_key(data)
        aligner = RigidAlignment(
            target=data[key],
            known_correspondences=self.known_correspondences,
        )

        data_ = data.transform(aligner).map_values(PvSurface)

        self.timer.stop("prep")

        self.params_["rigid_alignment"] = {
            "known_correspondences": self.known_correspondences,
        }
        self.results_["rigid_alignment"] = {
            "key": key,
        }

        return data_


class PairwiseDistancesMixin:
    def compute_pairwise_dists(self, meshes, metric):
        # data : polpo.dataset.Dataset

        self.timer.start("dists")

        dists = PairwiseDistances(
            meshes.keys_list(),
            putils.pairwise_dists_par(
                meshes.values_list(),
                metric.dist,
                as_matrix=False,
                n_jobs=self.n_jobs,
            ),
        )

        self.params_["dists"] = {
            "n_jobs": self.n_jobs,
        }
        self.timer.stop("dists")

        return dists
