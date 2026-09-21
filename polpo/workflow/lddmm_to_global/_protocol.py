# TODO: create script to check registration time with no decimation

import json
import traceback
from datetime import datetime, timezone

import numpy as np

from polpo.dataset import Dataset, NestedDataset
from polpo.ext.pyvista.surface_mesh import PvSurface
from polpo.pipeline.mesh.registration import RigidAlignment
from polpo.surface_mesh.deformetrica import FrechetMean, LddmmMetric, Point
from polpo.surface_mesh.varifold.tuning.geometry_based import SigmaFromLengths
from polpo.time import Timer

# TODO: add restart


class LddmmToGlobal:
    PROTOCOL_VERSION = "0.2.0"

    DEFAULT_REGISTRATION_KWARGS = {
        "regularisation": 1.0,
        "max_iter": 2000,
        "freeze_control_points": False,
        "metric": "varifold",
        "tol": 1e-16,
    }

    DEFAULT_FRECHET_MEAN_KWARGS = {
        "initial_step_size": 1e-1,
    }

    def __init__(
        self,
        known_correspondences,
        results_dir,
        ratio_kernel=1.5,
        ratio_charlen_mesh=2.0,
        ratio_charlen=0.25,
        registration_kwargs=None,
        frechet_mean_kwargs=None,
        random_state=None,
        metadata=None,
    ):
        self.timer = Timer()

        self.known_correspondences = known_correspondences
        self.results_dir = results_dir

        self.ratio_kernel = ratio_kernel
        self.ratio_charlen = ratio_charlen
        self.ratio_charlen_mesh = ratio_charlen_mesh

        self.registration_kwargs = {
            **self.DEFAULT_REGISTRATION_KWARGS,
            **(registration_kwargs or {}),
        }
        self.frechet_mean_kwargs = {
            **self.DEFAULT_FRECHET_MEAN_KWARGS,
            **(frechet_mean_kwargs or {}),
        }

        self.metadata = metadata or {}
        self.random_state = random_state

        self.reset()

    def reset(self):
        self.timer.reset()

        seed_sequence = np.random.SeedSequence(self.random_state)
        self.rng_ = np.random.default_rng(seed_sequence)

        self.params_ = {
            "version": self.PROTOCOL_VERSION,
            "metadata": self.metadata,
            "random_state": self.random_state,
            "rigid_alignment": {
                "known_correspondences": self.known_correspondences,
            },
            "kernel_tuning": {
                "ratio_kernel": self.ratio_kernel,
                "ratio_charlen_mesh": self.ratio_charlen_mesh,
                "ratio_charlen": self.ratio_charlen,
            },
            "registration": self.registration_kwargs,
            "frechet_mean": self.frechet_mean_kwargs,
        }

        self.results_ = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "random_state": int(seed_sequence.entropy),
        }

    def preprocess_meshes(self, data):  # TODO: update it in varifold
        # data : polpo.dataset.Dataset

        # rigidly aligns all the meshes against a randomly chosen target
        with self.timer("prep"):
            selected_mesh = data.sample(random_state=self.rng_)
            aligner = RigidAlignment(
                target=selected_mesh.values_list()[0],
                known_correspondences=self.known_correspondences,
            )

            data_ = data.transform(aligner).map_values(PvSurface)

        self.results_["rigid_alignment"] = {
            "key": selected_mesh.keys_list()[0],
        }

        return data_

    def tune_kernel(self, nested_meshes):
        # select varifold kernel using a randomly selected mesh per subject
        with self.timer("tuning"):
            sigma_search = SigmaFromLengths(
                ratio_charlen_mesh=self.ratio_charlen_mesh,
                ratio_charlen=self.ratio_charlen,
            )

            selected_meshes = nested_meshes.sample_inner(random_state=self.rng_)
            sigma_search.fit(selected_meshes.flatten().values_list())

        sigma_var = sigma_search.sigma_
        sigma_vel = self.ratio_kernel * sigma_var

        self.results_["kernel_tuning"] = {
            "sigma_vel": sigma_vel,
            "sigma_var": sigma_var,
            "meshes": selected_meshes.flatten().keys_list(),
        }

        return sigma_vel, sigma_var

    def instantiate_metric(self, sigma_vel, sigma_var):
        kwargs = {
            **self.registration_kwargs,
            "kernel_width": sigma_vel,
            "attachment_kernel_width": sigma_var,
        }

        metric = LddmmMetric(self.results_dir, **kwargs)

        self.params_["dirs"] = metric.dir_config.to_dict()

        return metric

    def meshes_as_points(self, nested_meshes, metric):
        return nested_meshes.map_items(
            lambda outer_key, inner_key, mesh: Point(
                id_=f"{outer_key}-{inner_key}",
                pv_surface=mesh,
                dirname=metric.dir_config.meshes_dir,
            )
        )

    def build_local_atlases(self, nested_points, metric, atlas_keys):
        estimator = FrechetMean(metric, **self.frechet_mean_kwargs)

        with self.timer("local_atlases"):
            atlases = {}
            for outer_key, points in nested_points.items():
                subset = Dataset(points).select(atlas_keys[outer_key]).values_list()
                estimator.fit(subset, atlas_id=outer_key)
                atlas = estimator.estimate_

                atlases[outer_key] = atlas

        return Dataset(atlases)

    def build_global_atlas(self, local_atlases, metric):
        estimator = FrechetMean(metric, **self.frechet_mean_kwargs)

        with self.timer("global_atlas"):
            estimator.fit(local_atlases.values_list(), "gl")

        return estimator.estimate_

    def register_and_transport(self, nested_points, metric, atlas, local_atlases):
        self.timer.start("register_and_transport")

        global_reprs = {}
        point_a = atlas
        for outer_key, points in nested_points.items():
            global_reprs[outer_key] = reprs = {}
            point_b = local_atlases[outer_key]

            vec_ba = metric.log(point_a, point_b)

            for inner_key, point_c in points.items():
                vec_bc = metric.log(point_c, point_b)

                trans_vec_bc = metric.parallel_transport(
                    vec_bc, point_b, direction=vec_ba
                )

                reprs[inner_key] = metric.exp(trans_vec_bc, point_a)

        self.timer.stop("register_and_transport")

        return NestedDataset(global_reprs)

    def write(self):
        with open(self.results_dir / "params.json", "w") as file:
            json.dump(self.params_, file, indent=2)

        with open(self.results_dir / "results.json", "w") as file:
            json.dump(self.results_, file, indent=2)

        with open(self.results_dir / "time.json", "w") as file:
            json.dump(self.timer.as_dict(), file, indent=2)

    def _record_failure(self, error):
        self.results_.update(
            {
                "status": "failed",
                "failed_stage": self.current_stage_,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )

    def run(self, nested_meshes, atlas_keys):
        # nested_meshes: dict or polpo.dataset.NestedDataset
        if isinstance(nested_meshes, dict):
            nested_meshes = NestedDataset(nested_meshes)

        self.reset()

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_["status"] = "running"

        try:
            with self.timer("run"):
                self.current_stage_ = "preprocessing"
                nested_meshes_ = self.preprocess_meshes(nested_meshes.flatten()).nest()

                self.current_stage_ = "metric_instantiation"
                sigma_vel, sigma_var = self.tune_kernel(nested_meshes_)
                metric = self.instantiate_metric(sigma_vel, sigma_var)

                nested_points = self.meshes_as_points(nested_meshes_, metric)

                self.current_stage_ = "local_atlases"
                local_atlases = self.build_local_atlases(
                    nested_points, metric, atlas_keys
                )

                self.current_stage_ = "global_atlas"
                atlas = self.build_global_atlas(local_atlases, metric)

                self.current_stage_ = "registration_and_transport"
                global_reprs = self.register_and_transport(
                    nested_points,
                    metric,
                    atlas,
                    local_atlases,
                )

                self.current_stage_ = "completed"
        except Exception as error:
            self._record_failure(error)
            self.write()
            raise

        self.results_["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.results_["status"] = "completed"

        self.write()

        self.metric_ = metric
        self.local_atlases_ = local_atlases
        self.atlas_ = atlas
        self.global_reprs_ = global_reprs

        return self
