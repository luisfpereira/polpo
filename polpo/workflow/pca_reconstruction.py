from functools import partial

import numpy as np

from polpo.ext.numpy.io import save_indexed_array
from polpo.ext.sklearn.decomposition import PCA, TruncatedPCA
from polpo.ext.sklearn.io import load_estimator, save_estimator
from polpo.transform import CompositeTransform
from polpo.workflow.task import TaskRunner, task


class PCAReconstructionEvaluator(TaskRunner):
    """Evaluate PCA mesh reconstruction across component counts.

    Fits PCA to a vector representation of a collection of meshes, reconstructs
    the meshes using different numbers of principal components, and evaluates
    the reconstructions with one or more metric callables.

    Each metric is exposed as an independent task and its reconstruction
    dissimilarities are persisted separately.

    Parameters
    ----------
    dataset : Dataset
        Dataset of meshes used to fit PCA and evaluate reconstruction. Dataset
        keys are used to index persisted reconstruction errors.
    transform : invertible transform
        Transform mapping meshes to the vector representation on which PCA is
        fitted.
    metrics : dict
        Mapping from task names to callables measuring dissimilarity between
        two meshes.
    ks : array-like
        Numbers of principal components used for reconstruction.
    results_dir : path-like
        Directory where fitted estimators and evaluation results are stored.
    state_dir : path-like
        Directory where task execution state is stored. If ``None``,
        ``results_dir`` is used.
    """

    def __init__(
        self,
        results_dir,
        dataset,
        transform,
        metrics,
        ks,
        state_dir=None,
    ):
        if state_dir is None:
            state_dir = results_dir

        super().__init__(
            state_dir,
            metadata={
                "ks": ks,
                "metrics": list(metrics),
                "transform": transform.__class__.__name__,
            },
        )

        self.dataset = dataset
        self.transform = transform
        self.metrics = metrics

        self.ks = ks
        self.results_dir = results_dir

    def tasks(self):
        tasks = super().tasks()

        tasks.update(
            {
                name: partial(self._run_metric, name, metric)
                for name, metric in self.metrics.items()
            }
        )

        return tasks

    @property
    def pca_path(self):
        return self.results_dir / "pca.joblib"

    @task
    def fit(self):
        X = self.transform(self.dataset.values_list())

        pca = PCA(
            n_components=int(np.max(self.ks)),
        ).fit(X)

        save_estimator(self.pca_path, pca)

    def _run_metric(self, name, metric):
        """Evaluate and persist reconstruction dissimilarities for one metric.

        Parameters
        ----------
        name : str
            Name used to identify the metric task and its persisted results.
        metric : callable
            Function measuring dissimilarity between an original and a
            reconstructed mesh.
        """
        pca = load_estimator(self.pca_path)

        meshes = self.dataset.values_list()

        dissimilarities = []

        for k in self.ks:
            pca_k = TruncatedPCA.from_fitted(pca, n_components=k)

            transform = CompositeTransform([self.transform, pca_k])

            rec_meshes = transform.inverse(transform(meshes))

            dissimilarities.append(
                [
                    metric(mesh, rec_mesh)
                    for mesh, rec_mesh in zip(
                        meshes,
                        rec_meshes,
                    )
                ]
            )

        # (n_meshes, n_ks), so keys index axis 0
        dissimilarities = np.asarray(dissimilarities).T

        save_indexed_array(
            self.results_dir / f"{name}.npz",
            keys=self.dataset.keys_list(),
            data=dissimilarities,
        )
