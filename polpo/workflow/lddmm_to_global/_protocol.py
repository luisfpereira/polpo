import traceback

import numpy as np

from polpo.dataset import Dataset, NestedDataset
from polpo.io.json import check_json_serializable, dump_json
from polpo.surface_mesh.deformetrica import FrechetMean, Point
from polpo.surface_mesh.registration import SurfaceRigidRegistration
from polpo.surface_mesh.varifold.tuning import SigmaFromScale
from polpo.time import Timer, utc_now

from ._metric import instantiate_lddmm_metric


class LddmmToGlobal:
    """Run the LDDMM-to-global shape representation protocol.

    Parameters
    ----------
    known_correspondences : bool
        Whether meshes have known vertex correspondences.
    results_dir : path-like
        Directory where protocol outputs and intermediate results are stored.
    ratio_kernel : float
        Ratio between the deformation-kernel width and the tuned attachment-kernel width.
    discretization_ratio : float
        Ratio controlling the attachment-kernel scale relative to mesh discretization.
    object_ratio : float
        Ratio controlling the attachment-kernel scale relative to object size.
    regularization : float
        Regularization parameter controlling the relative weight of the attachment term.
    max_iter : int
        Maximum number of registration optimization iterations.
    tol : float
        Convergence tolerance for registration optimization.
    random_state : int
        Seed controlling random operations in the protocol.
    metadata : dict
        Metadata persisted with the protocol results.
    """

    PROTOCOL_VERSION = "0.4.0"

    def __init__(
        self,
        known_correspondences,
        results_dir,
        ratio_kernel=1.5,
        discretization_ratio=2.0,
        object_ratio=0.25,
        regularization=1.0,
        max_iter=500,
        tol=1e-5,
        random_state=None,
        metadata=None,
    ):
        self.timer = Timer()

        self.known_correspondences = known_correspondences
        self.results_dir = results_dir

        self.ratio_kernel = ratio_kernel
        self.object_ratio = object_ratio
        self.discretization_ratio = discretization_ratio

        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol

        self.metadata = dict(metadata or {})
        self.random_state = random_state

        check_json_serializable(self.metadata)

        self._reset()

    def _reset(self):
        """Reset runtime state for a new protocol run."""
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
                "discretization_ratio": self.discretization_ratio,
                "object_ratio": self.object_ratio,
            },
            "metric": {
                "regularization": self.regularization,
                "max_iter": self.max_iter,
                "tol": self.tol,
            },
        }

        self.results_ = {
            "started_at": utc_now(),
            "random_state": int(seed_sequence.entropy),
        }

    def preprocess_meshes(self, data, target_data=None):
        """Rigidly align meshes to a randomly selected target.

        Parameters
        ----------
        data : Dataset
            Surface meshes to align.
        target_data : Dataset or None
            Surface meshes from which to select the alignment target.

        Returns
        -------
        aligned : Dataset
            Rigidly aligned surface meshes.
        """
        target_data = target_data or data

        with self.timer("prep"):
            target_mesh = target_data.sample(random_state=self.rng_)

            registration = SurfaceRigidRegistration(
                known_correspondences=self.known_correspondences
            ).against_same_target(target_mesh.values_list()[0])

            data_ = data.transform(registration)

        self.results_["rigid_alignment"] = {
            "key": target_mesh.keys_list()[0],
        }

        return data_

    def tune_kernel(self, meshes):
        """Estimate kernel widths using atlas meshes.

        Parameters
        ----------
        meshes : Dataset
            Dataset of aligned surface meshes.

        Returns
        -------
        kernel_width : float
            Width of the deformation kernel.
        attachment_kernel_width : float
            Width of the varifold attachment kernel.
        """
        with self.timer("tuning"):
            sigma_search = SigmaFromScale(
                discretization_ratio=self.discretization_ratio,
                object_ratio=self.object_ratio,
            )

            attachment_kernel_width = sigma_search(meshes.values_list())

        kernel_width = self.ratio_kernel * attachment_kernel_width

        self.results_["kernel_tuning"] = {
            "kernel_width": kernel_width,
            "attachment_kernel_width": attachment_kernel_width,
        }

        return kernel_width, attachment_kernel_width

    def instantiate_metric(self, kernel_width, attachment_kernel_width):
        """Instantiate the LDDMM metric used by the protocol.

        Parameters
        ----------
        kernel_width : float
            Width of the deformation kernel.
        attachment_kernel_width : float
            Width of the varifold attachment kernel.

        Returns
        -------
        metric : LddmmMetric
            Configured LDDMM metric.
        """
        metric = instantiate_lddmm_metric(
            self.results_dir,
            kernel_width=kernel_width,
            attachment_kernel_width=attachment_kernel_width,
            regularization=self.regularization,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        metric.config.set_attachment(
            noise_std=self.regularization,
        )
        metric.config.set_optimization(
            max_iter=self.max_iter,
            tol=self.tol,
        )

        self.params_["dirs"] = metric.dir_config.to_dict()

        return metric

    def meshes_as_points(self, nested_meshes, metric):
        """Convert surface meshes to LDDMM points.

        Parameters
        ----------
        nested_meshes : NestedDataset
            Nested dataset of surface meshes.
        metric : LddmmMetric
            Metric providing the storage configuration.

        Returns
        -------
        points : NestedDataset
            Nested dataset of LDDMM points.
        """
        return nested_meshes.map_items(
            lambda outer_key, inner_key, mesh: Point(
                id_=f"{outer_key}-{inner_key}",
                surface=mesh,
                dirname=metric.dir_config.meshes,
            )
        )

    def build_local_atlases(self, nested_points, metric):
        """Estimate one local atlas per outer dataset key.

        Parameters
        ----------
        nested_points : NestedDataset
            Nested dataset of LDDMM points.
        metric : LddmmMetric
            Metric used for atlas estimation.

        Returns
        -------
        atlases : Dataset
            Local atlas for each outer key.
        """
        estimator = FrechetMean(metric)

        with self.timer("local_atlases"):
            atlases = {}
            for outer_key, points in nested_points.items():
                estimator.fit(Dataset(points).values_list(), atlas_id=outer_key)
                atlases[outer_key] = estimator.estimate_

        return Dataset(atlases)

    def build_global_atlas(self, local_atlases, metric):
        """Estimate a global atlas from the local atlases.

        Parameters
        ----------
        local_atlases : Dataset
            Local atlases.
        metric : LddmmMetric
            Metric used for atlas estimation.

        Returns
        -------
        atlas : Point
            Global atlas.
        """
        estimator = FrechetMean(metric)

        with self.timer("global_atlas"):
            estimator.fit(local_atlases.values_list(), "gl")

        return estimator.estimate_

    def register_and_transport(self, nested_points, metric, atlas, local_atlases):
        """Map shapes to the global atlas through parallel transport.

        For each shape, compute its tangent representation at the corresponding
        local atlas, parallel transport it to the global atlas, and exponentiate
        the transported tangent vector.

        Parameters
        ----------
        nested_points : NestedDataset
            Nested dataset of LDDMM points.
        metric : LddmmMetric
            Metric used for registration, transport, and shooting.
        atlas : Point
            Global atlas.
        local_atlases : Dataset
            Local atlas for each outer key.

        Returns
        -------
        global_reprs : NestedDataset
            Shapes represented relative to the global atlas.
        """
        with self.timer("register_and_transport"):
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

        return NestedDataset(global_reprs)

    def _write(self):
        """Write protocol parameters, results, and timings to disk."""
        dump_json(self.results_dir / "params.json", self.params_)
        dump_json(self.results_dir / "results.json", self.results_)
        dump_json(self.results_dir / "time.json", self.timer.as_dict())

    def _record_failure(self, error):
        """Record information about a protocol failure."""
        self.results_.update(
            {
                "status": "failed",
                "failed_stage": self.current_stage_,
                "finished_at": utc_now(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )

    def _validate_keys(self, nested_meshes, atlas_keys, atlas_only_keys):
        nested_keys = nested_meshes.nested_keys()

        missing_outer = set(nested_keys) - set(atlas_keys)
        if missing_outer:
            raise ValueError(
                f"Missing atlas keys for outer keys: {sorted(missing_outer)}."
            )

        _validate_nested_keys(atlas_keys, nested_keys, "atlas_keys")
        _validate_nested_keys(atlas_only_keys, atlas_keys, "atlas_only_keys")

    def run(self, nested_meshes, atlas_keys, atlas_only_keys=None):
        """Run the LDDMM-to-global shape representation protocol.

        Parameters
        ----------
        nested_meshes : NestedDataset or dict
            Nested dataset of input surface meshes.
        atlas_keys : mapping
            Inner keys used to estimate each local atlas.
        atlas_only_keys: mapping
            Inner keys used for atlas estimation but excluded from global
            representations.

        Returns
        -------
        self : LddmmToGlobal
            Fitted protocol with computed atlases and global representations.
        """
        if isinstance(nested_meshes, dict):
            nested_meshes = NestedDataset(nested_meshes)

        atlas_only_keys = atlas_only_keys or {}

        self._reset()
        self._validate_keys(nested_meshes, atlas_keys, atlas_only_keys)

        self.params_["keys"] = nested_meshes.nested_keys()
        self.params_["atlas_keys"] = atlas_keys
        self.params_["atlas_only_keys"] = atlas_only_keys

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_["status"] = "running"

        try:
            with self.timer():
                self.current_stage_ = "preprocessing"
                nested_meshes_ = self.preprocess_meshes(
                    nested_meshes.flatten(),
                    nested_meshes.drop_inner(atlas_only_keys).flatten(),
                ).nest()

                self.current_stage_ = "metric_instantiation"
                atlas_meshes = nested_meshes_.select_inner(atlas_keys)
                kernel_width, attachment_kernel_width = self.tune_kernel(
                    atlas_meshes.flatten()
                )
                metric = self.instantiate_metric(kernel_width, attachment_kernel_width)

                nested_points = self.meshes_as_points(nested_meshes_, metric)
                atlas_points = nested_points.select_inner(atlas_keys)

                self.current_stage_ = "local_atlases"
                local_atlases = self.build_local_atlases(atlas_points, metric)

                self.current_stage_ = "global_atlas"
                atlas = self.build_global_atlas(local_atlases, metric)

                nested_points_ = nested_points.drop_inner(atlas_only_keys)

                self.current_stage_ = "registration_and_transport"
                global_reprs = self.register_and_transport(
                    nested_points_,
                    metric,
                    atlas,
                    local_atlases,
                )

                self.current_stage_ = "completed"
        except Exception as error:
            self._record_failure(error)
            self._write()
            raise

        self.results_["finished_at"] = utc_now()
        self.results_["status"] = "completed"

        self._write()

        self.metric_ = metric
        self.local_atlases_ = local_atlases
        self.atlas_ = atlas
        self.global_reprs_ = global_reprs

        return self


def _validate_nested_keys(keys, reference, name):
    """Validate nested keys against a reference mapping.

    Parameters
    ----------
    keys : mapping
        Mapping from outer keys to collections of inner keys to validate.
    reference : mapping
        Mapping from outer keys to the allowed inner keys for each outer key.
    name : str
        Name used to identify `keys` in validation errors.

    Raises
    ------
    ValueError
        If `keys` contains an outer key not present in `reference`, or if any
        inner key is not contained in `reference` for the corresponding outer
        key.

    Notes
    -----
    The validation enforces

        keys[outer_key] <= reference[outer_key]

    for every outer key in `keys`, and requires

        set(keys) <= set(reference).
    """
    unknown_outer = set(keys) - set(reference)
    if unknown_outer:
        raise ValueError(f"Unknown outer keys in {name}: {sorted(unknown_outer)}.")

    for outer_key, inner_keys in keys.items():
        unknown_inner = set(inner_keys) - set(reference[outer_key])
        if unknown_inner:
            raise ValueError(
                f"Unknown inner keys in {name}[{outer_key!r}]: "
                f"{sorted(unknown_inner)}."
            )
