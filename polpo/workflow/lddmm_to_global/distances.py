from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from polpo.dataset import Dataset
from polpo.distmat import PairwiseDistances
from polpo.ext.numpy.io import load_dict, save_dict_as_array
from polpo.io.json import load_json
from polpo.surface_mesh.deformetrica.geometry import LddmmMetric
from polpo.surface_mesh.euclidean import EuclideanSurfaces
from polpo.surface_mesh.varifold.geometry import VarifoldMetric
from polpo.utils.dict_ import merge_dicts
from polpo.utils.np import pairwise_dists
from polpo.workflow.task import TaskRunner

from .output import LddmmToGlobalOutput


@dataclass
class TaskSpec:
    kind: str
    filename: str
    load: callable
    save: callable = None


def _save_pairwise_distances(path, result):
    result.save(path)


def _load_distances(path):
    return Dataset(load_dict(path))


def _task(key, filename=None, *, pairwise=False):
    filename = f"{key}.npz" if filename is None else filename
    kind = "pairwise" if pairwise else "dataset"

    save = _save_pairwise_distances if pairwise else save_dict_as_array
    load = PairwiseDistances.load if pairwise else _load_distances

    return key, TaskSpec(
        filename=filename,
        kind=kind,
        save=save,
        load=load,
    )


LOADERS_BY_KIND = {
    "pairwise": PairwiseDistances.load,
    "dataset": _load_distances,
}

REGISTRATION_TASKS = dict(
    [
        _task("local_atlas_to_reconstructed"),
        _task("global_atlas_to_global"),
    ]
)

DISTANCE_TASKS = dict(
    [
        _task("local_reconstruction_error"),
        _task("local_atlas_fit_error"),
        _task("global_atlas_fit_error"),
        _task("local_to_global_reconstruction_error"),
        _task("local_to_global_transport_error"),
        _task("local_pairwise", pairwise=True),
        _task("local_reconstructed_pairwise", pairwise=True),
        _task("global_pairwise", pairwise=True),
    ]
)


def varifold_metric_from_results(data, engine="auto"):
    sigma = data["kernel_tuning"]["sigma_var"]
    return VarifoldMetric(sigma=sigma, engine=engine)


def _reconstruction_error(registration_res, dist_fnc):
    return dist_fnc(
        registration_res.point.as_surface(),
        registration_res.reconstructed.as_surface(),
    )


def _atlas_reconstruction_error(atlas_res, dist_fnc):
    def _id_to_key(id_):
        if "-" in id_:
            return tuple(id_.split("-"))

        return id_

    return {
        _id_to_key(point.id): dist_fnc(point.as_surface(), cmp_point.as_surface())
        for point, cmp_point in zip(atlas_res.points, atlas_res.reconstructed)
    }


def _parallel_transport_res_error(transport_res, atlas, dist_fnc):
    return dist_fnc(
        transport_res.reconstructed.as_surface(),
        atlas,
    )


class DistanceEvaluator:
    def __init__(self, source, metric):
        self.source = source
        self.metric = metric

    def local_reconstruction_error(self):
        # compares original against reconstructed after registration
        return self.source.mapped_view.local_registrations.map_values(
            _reconstruction_error,
            dist_fnc=self.metric.dist,
        ).flatten()

    def local_atlas_fit_error(self):
        # compares original against reconstructed during deterministic atlas
        errors = self.source.mapped_view.local_atlases.map_values(
            _atlas_reconstruction_error,
            dist_fnc=self.metric.dist,
        )
        return Dataset(merge_dicts(errors.values_list()))

    def global_atlas_fit_error(self):
        # compares local atlas against reconstructed local atlas during deterministic atlas
        return _atlas_reconstruction_error(
            self.source.global_atlas,
            dist_fnc=self.metric.dist,
        )

    def local_to_global_reconstruction_error(self):
        # compares global against registration from local
        # establishes transport direction
        return self.source.mapped_view.registrations_to_global_atlas.map_values(
            _reconstruction_error,
            dist_fnc=self.metric.dist,
        )

    def local_to_global_transport_error(self):
        # error induced by transport direction
        # only collecting one per outer due to the nature of the algorithm
        # must compare with local_to_global_reconstruction_error

        selected = {}
        for outer_key, inner in self.source.mapped_view.transports.items():
            try:
                selected[outer_key] = next(
                    result for result in inner.values() if result.method == "fanning"
                )
            except StopIteration:
                raise ValueError(
                    f"No fanning transport found for {outer_key!r}."
                ) from None

        trans_res = Dataset(selected)

        # NB: only fan has reconstructed

        return trans_res.map_values(
            _parallel_transport_res_error,
            atlas=self.source.global_atlas_point.as_surface(),
            dist_fnc=self.metric.dist,
        )

    def local_pairwise(self):
        return self._pairwise(self.source.mapped_view.dataset.flatten())

    def local_reconstructed_pairwise(self):
        return self._pairwise(
            self.source.mapped_view.local_reconstructed_points.flatten()
        )

    def global_pairwise(self):
        return self._pairwise(self.source.mapped_view.global_points.flatten())

    def _pairwise(self, data):
        surfaces = data.map_values(lambda point: point.as_surface())

        return PairwiseDistances(
            surfaces.keys_list(),
            pairwise_dists(
                surfaces.values_list(),
                self.metric.dist,
                as_matrix=False,
            ),
        )


class VarifoldDistances(DistanceEvaluator):
    def __init__(self, experiment_dir, engine="auto"):
        source = LddmmToGlobalOutput(experiment_dir)

        metric = varifold_metric_from_results(
            source.results,
            engine=engine,
        )

        super().__init__(source, metric)


class EuclideanDistances(DistanceEvaluator):
    def __init__(self, experiment_dir):
        source = LddmmToGlobalOutput(experiment_dir)

        metric = EuclideanSurfaces(
            faces=source.global_atlas_point.as_surface().faces
        ).metric

        super().__init__(source, metric)


class LddmmDistances:
    def __init__(self, experiment_dir):
        source = LddmmToGlobalOutput(experiment_dir)
        metric = LddmmMetric(
            experiment_dir,
            kernel_width=source.results["kernel_tuning"]["sigma_vel"],
        )

        self.source = source
        self.metric = metric

    def local_atlas_to_reconstructed(self):
        # distance from local atlas to reconstructed
        return self.source.mapped_view.local_registrations.map_values(
            # TODO: add norm
            lambda x: self.metric.norm(x.tangent_vec),
        ).flatten()

    def global_atlas_to_global(self):
        # distance from local atlas to reconstructed
        # NB: parallel transport preserves distance
        return self.source.mapped_view.global_shoots.map_values(
            lambda x: self.metric.norm(x.tangent_vec),
        ).flatten()


class PersistentEvaluator(TaskRunner):
    def __init__(self, evaluator, results_dir="post_dists", task_specs=None):
        results_dir = Path(results_dir)
        if not results_dir.is_absolute():
            results_dir = evaluator.source.path / results_dir

        if task_specs is None:
            if isinstance(evaluator, DistanceEvaluator):
                task_specs = DISTANCE_TASKS
            elif isinstance(evaluator, LddmmDistances):
                task_specs = REGISTRATION_TASKS
            else:
                raise ValueError(
                    f"No default task_specs for {type(evaluator).__name__}"
                )

        self.evaluator = evaluator

        self.task_specs = task_specs
        self.results_dir = results_dir

        super().__init__(results_dir)

    def tasks(self):
        return {name: self._make_task(name) for name in self.task_specs}

    def _make_task(self, name):
        def run():
            result = getattr(self.evaluator, name)()
            spec = self.task_specs[name]

            spec.save(self.results_dir / spec.filename, result)

        return run

    def _mark_completed(self, task):
        spec = self.task_specs[task]
        self.manifest_["tasks"][task] = {
            "status": "completed",
            "kind": spec.kind,
            "filename": spec.filename,
            **self.timer.as_dict(task),
        }

    @property
    def results(self):
        return DistanceResults.from_dir(self.results_dir)


class _AttrAccessMixin:
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class DistanceResults(_AttrAccessMixin):
    def __init__(self, results_dir, task_specs):
        self.results_dir = Path(results_dir)
        self.task_specs = task_specs

        self._cache = {}

    @classmethod
    def from_dir(cls, results_dir):
        results_dir = Path(results_dir)
        manifest = load_json(results_dir / "manifest.json")
        task_specs = {
            task: TaskSpec(
                kind=info["kind"],
                filename=info["filename"],
                load=LOADERS_BY_KIND[info["kind"]],
            )
            for task, info in manifest["tasks"].items()
            if info["status"] == "completed"
        }
        return cls(results_dir, task_specs=task_specs)

    @property
    def manifest_path(self):
        return self.results_dir / "manifest.json"

    @cached_property
    def manifest(self):
        return load_json(self.manifest_path)

    def __getitem__(self, task):
        if task not in self.task_specs:
            raise KeyError(task)

        if not self.is_available(task):
            raise KeyError(f"Distance result {task!r} is not available.")

        if task not in self._cache:
            spec = self.task_specs[task]
            self._cache[task] = spec.load(self.results_dir / spec.filename)

        return lambda: self._cache[task]

    def __iter__(self):
        return (task for task in self.task_specs if self.is_available(task))

    def __len__(self):
        return sum(self.is_available(task) for task in self.task_specs)

    def __contains__(self, task):
        return self.is_available(task)

    def is_available(self, task):
        if task not in self.task_specs:
            return False

        task_info = self.manifest.get("tasks", {}).get(task, {})
        return task_info.get("status") == "completed"

    def clear_cache(self, task=None):
        if task is None:
            self._cache.clear()
            return

        self._cache.pop(task, None)

    def refresh(self):
        self.clear_cache()
        self.__dict__.pop("manifest", None)

    @classmethod
    def combine(cls, results):
        return MultiDistanceResults(results)


class MultiDistanceResults(_AttrAccessMixin):
    def __init__(self, results, prefix_key=None):
        self._results = results
        if prefix_key is None:

            def prefix_key(label, key):
                if isinstance(key, str):
                    return f"{label}_{key}"

                return (label,) + key

        self._prefix_key = prefix_key

    def __getitem__(self, task):
        values = [
            self._adapt(
                label,
                res[task](),
            )
            for label, res in self._results.items()
        ]
        return lambda: self._merge(values)

    def _adapt(self, label, value):
        if isinstance(value, Dataset):
            return value.map_keys(lambda key: self._prefix_key(label, key))

        if isinstance(value, PairwiseDistances):
            return value.map_labels(lambda key: self._prefix_key(label, key))

        return value

    def _merge(self, values):
        return type(values[0]).merge(values)
