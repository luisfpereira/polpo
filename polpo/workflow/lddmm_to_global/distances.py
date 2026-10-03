from dataclasses import dataclass
from pathlib import Path

from polpo.dataset import Dataset
from polpo.distmat import PairwiseDistances
from polpo.ext.numpy.io import load_dict, save_dict_as_array
from polpo.io.json import load_json
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
        _task("local_atlas_distance"),
        _task("global_atlas_distance"),
    ]
)


def _atlas_reconstruction_error(atlas_res, dist_fnc):
    def _id_to_key(id_):
        if "-" in id_:
            return tuple(id_.split("-"))

        return id_

    return {
        _id_to_key(point.id): dist_fnc(point, reconstructed)
        for point, reconstructed in zip(
            atlas_res.points,
            atlas_res.reconstructed,
        )
    }


class DistanceEvaluator:
    """Evaluate geometric discrepancies for an LDDMM-to-global output.

    Parameters
    ----------
    source : LddmmToGlobalOutput
        Protocol output providing persisted shapes and registration artifacts.
    metric : object
        Metric exposing a ``dist`` method used to evaluate geometric
        discrepancies.
    """

    def __init__(self, source, metric):
        self.source = source
        self.metric = metric

    def _dist(self, point_a, point_b):
        return self.metric.dist(
            point_a.as_surface(),
            point_b.as_surface(),
        )

    def local_reconstruction_error(self):
        """Compute distances between aligned shapes and local reconstructions."""
        return self.source.local_registrations.map_values(
            lambda registration: self._dist(
                registration.point,
                registration.reconstructed,
            )
        ).flatten()

    def local_atlas_fit_error(self):
        """Compute distances between atlas input shapes and their local-atlas reconstructions."""
        errors = self.source.local_atlases.map_values(
            _atlas_reconstruction_error,
            dist_fnc=self._dist,
        )
        return Dataset(merge_dicts(errors.values_list()))

    def global_atlas_fit_error(self):
        """Compute distances between local atlases and their global-atlas reconstructions."""
        return Dataset(
            _atlas_reconstruction_error(
                self.source.global_atlas,
                dist_fnc=self._dist,
            )
        )

    def local_to_global_reconstruction_error(self):
        """Compute distances between the global atlas and local-to-global registration reconstructions.

        This reflects numerical error in establishing the local-to-global geodesic.
        """
        return self.source.registrations_to_global_atlas.map_values(
            lambda registration: self._dist(
                registration.point,
                registration.reconstructed,
            )
        )

    def local_to_global_transport_error(self):
        """Compute distances between the global atlas and local-to-global transport reconstructions.

        This reflects numerical error in constructing the local-to-global geodesic
        direction used for parallel transport.
        Since this direction is defined once
        for each local atlas, only one transport reconstruction is evaluated per
        outer key.
        """
        selected = {}
        for outer_key, inner in self.source.transports.items():
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
            lambda transport: self._dist(
                self.source.global_atlas_point,
                transport.reconstructed,
            )
        )

    def local_pairwise(self):
        """Compute pairwise distances between rigidly aligned input shapes."""
        return self._pairwise(self.source.dataset.flatten())

    def local_reconstructed_pairwise(self):
        """Compute pairwise distances between shapes reconstructed from local registrations."""
        return self._pairwise(self.source.local_reconstructed_points.flatten())

    def global_pairwise(self):
        """Compute pairwise distances between shapes represented at the global atlas."""
        return self._pairwise(self.source.global_points.flatten())

    def local_atlas_distance(self):
        """Compute distances between local atlases and locally reconstructed shapes."""
        local_atlases = self.source.local_atlases_points

        return self.source.local_reconstructed_points.map_items(
            lambda outer_key, _, point: self._dist(
                local_atlases[outer_key],
                point,
            )
        ).flatten()

    def global_atlas_distance(self):
        """Compute distances between the global atlas and globally represented shapes."""
        return self.source.global_points.map_values(
            lambda point: self._dist(
                self.source.global_atlas_point,
                point,
            )
        ).flatten()

    def _pairwise(self, data):
        """Compute pairwise distances between surface-valued points.

        Parameters
        ----------
        data : Dataset
            Dataset of points convertible to surfaces.

        Returns
        -------
        distances : PairwiseDistances
            Pairwise distances indexed by the input dataset keys.
        """
        return PairwiseDistances(
            data.keys_list(),
            pairwise_dists(
                data.values_list(),
                self._dist,
                as_matrix=False,
            ),
        )


class VarifoldDistances(DistanceEvaluator):
    """Evaluate varifold distances for an LDDMM-to-global output."""

    def __init__(self, experiment_dir, engine="auto"):
        source = LddmmToGlobalOutput(experiment_dir)

        super().__init__(source, source.instantiate_varifold_metric(engine=engine))


class EuclideanDistances(DistanceEvaluator):
    """Evaluate Euclidean distances for an LDDMM-to-global output."""

    def __init__(self, experiment_dir):
        source = LddmmToGlobalOutput(experiment_dir)

        super().__init__(source, source.instantiate_euclidean_metric())


class LddmmDistances(DistanceEvaluator):
    """Evaluate LDDMM distances for an LDDMM-to-global output."""

    def __init__(self, experiment_dir):
        source = LddmmToGlobalOutput(experiment_dir)
        self.source = source

        super().__init__(source, source.instantiate_lddmm_metric())

    def _dist(self, point_a, point_b):
        return self.metric.dist(point_a, point_b)

    def local_atlas_fit_error(self):
        """Compute LDDMM distances from local atlases to their input shapes."""
        errors = self.source.local_atlases.map_values(
            lambda atlas: {
                point.id: self.metric.norm(tangent_vec)
                for point, tangent_vec in zip(
                    atlas.points,
                    atlas.tangent_vecs,
                )
            }
        )
        return Dataset(merge_dicts(errors.values_list()))

    def global_atlas_fit_error(self):
        """Compute LDDMM distances from the global atlas to the local atlases."""
        atlas = self.source.global_atlas

        return Dataset(
            {
                point.id: self.metric.norm(tangent_vec)
                for point, tangent_vec in zip(
                    atlas.points,
                    atlas.tangent_vecs,
                )
            }
        )

    def local_atlas_distance(self):
        """Compute distances between local atlases and locally reconstructed shapes."""
        return self.source.local_registrations.map_values(
            lambda registration: self.metric.norm(registration.tangent_vec),
        ).flatten()

    def global_atlas_distance(self):
        """Compute distances between the global atlas and globally represented shapes."""
        return self.source.global_shoots.map_values(
            lambda shoot: self.metric.norm(shoot.tangent_vec),
        ).flatten()


class PersistentEvaluator(TaskRunner):
    """Evaluate distance tasks and persist their results to disk.

    Parameters
    ----------
    evaluator : DistanceEvaluator
        Distance evaluator providing the task computations.
    results_dir : path-like
        Directory where computed distance results are stored.
    task_specs : mapping or None
        Specifications of the distance tasks to execute and persist.
    """

    def __init__(self, evaluator, results_dir="post_dists", task_specs=None):
        results_dir = Path(results_dir)
        if not results_dir.is_absolute():
            results_dir = evaluator.source.path / results_dir

        if task_specs is None:
            task_specs = DISTANCE_TASKS

        self.evaluator = evaluator

        self.task_specs = task_specs
        self.results_dir = results_dir

        super().__init__(results_dir)

    def tasks(self):
        """Return the distance tasks to execute."""
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
        """Return lazy access to the persisted distance results."""
        return DistanceResults.from_dir(self.results_dir)


class DistanceResults:
    """Lazy access to persisted distance results.

    Completed tasks are exposed as callable attributes and loaded from disk
    on first access.

    Parameters
    ----------
    results_dir : path-like
        Directory containing persisted distance results.
    task_specs : mapping
        Specifications for the available persisted tasks.
    """

    def __init__(self, results_dir, task_specs):
        self.results_dir = Path(results_dir)
        self.task_specs = task_specs
        self._cache = {}

    @classmethod
    def from_dir(cls, results_dir):
        """Create distance results from a persisted task manifest.

        Parameters
        ----------
        results_dir : path-like
            Directory containing the task manifest and persisted results.

        Returns
        -------
        results : DistanceResults
            Lazy accessor for completed distance tasks.
        """
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

    def __getattr__(self, task):
        """Return a lazy loader for a persisted task result."""
        if task not in self.task_specs:
            raise AttributeError(task)

        def _load():
            if task not in self._cache:
                spec = self.task_specs[task]
                self._cache[task] = spec.load(self.results_dir / spec.filename)

            return self._cache[task]

        return _load


class MultiDistanceResults:
    """Combine corresponding distance results across labeled sources.

    Task attributes are evaluated for each source and merged into a single
    result. Keys or labels are prefixed by the corresponding source label to
    preserve their origin.

    Parameters
    ----------
    results : mapping
        Mapping from source labels to distance evaluators or persisted distance
        results exposing the same task interface.
    prefix_key : callable or None
        Function mapping a source label and result key to a combined key.
    """

    def __init__(self, results, prefix_key=None):
        self._results = results
        if prefix_key is None:

            def prefix_key(label, key):
                if isinstance(key, str):
                    return f"{label}_{key}"

                return (label,) + key

        self._prefix_key = prefix_key

    def __getattr__(self, task):
        """Return a callable combining a task across all sources."""
        if not all(hasattr(res, task) for res in self._results.values()):
            raise AttributeError(task)

        def _compute():
            values = [
                self._adapt(label, getattr(res, task)())
                for label, res in self._results.items()
            ]
            return type(values[0]).merge_many(values)

        return _compute

    def _adapt(self, label, value):
        """Prefix keys or labels of a result with its source label."""
        if isinstance(value, Dataset):
            return value.map_keys(lambda key: self._prefix_key(label, key))

        if isinstance(value, PairwiseDistances):
            return value.map_labels(lambda key: self._prefix_key(label, key))

        raise TypeError(
            f"Cannot adapt value of type {type(value).__name__!r} for label {label!r}. "
            f"Expected Dataset or PairwiseDistances."
        )
