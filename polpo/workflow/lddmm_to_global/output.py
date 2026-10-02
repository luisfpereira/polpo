from functools import cached_property
from pathlib import Path

import pyvista as pv

from polpo.dataset import Dataset, NestedDataset, NestedKeyMap
from polpo.ext.pyvista.surface_mesh import PvSurface
from polpo.io.json import load_json
from polpo.surface_mesh.deformetrica.paths import LddmmPaths

from ._collect import (
    collect_atlases,
    collect_dataset,
    collect_global_shoots,
    collect_local_registrations,
    collect_registrations_to_global_atlas,
    collect_transports,
    get_global_atlas,
)


class _OutputView:
    """View of protocol outputs under a nested key transformation."""

    def __init__(self, output, key_map=None):
        self._output = output
        self.key_map = key_map

    def _transform(self, data):
        if self.key_map is not None:
            data = data.map_keys(self.key_map)

        return data

    def with_key_map(self, key_map):
        """Return a view with an additional key transformation."""
        if self.key_map is not None:
            key_map = self.key_map.chain_with(key_map)

        return type(self)(self._output, key_map=key_map)


class LddmmToGlobalOutputView(_OutputView):
    """Lazy view over artifacts produced by an LDDMM-to-global run."""

    @cached_property
    def dataset(self):
        """Rigidly aligned input meshes."""
        data = collect_dataset(
            self._output.dir_config,
            self._output.keys,
        )
        return self._transform(data)

    @cached_property
    def local_registrations(self):
        """Registrations from local atlases to observed shapes."""
        data = collect_local_registrations(
            self._output.dir_config,
            self._output.representation_keys,
        )
        return self._transform(data)

    @cached_property
    def local_reconstructed_points(self):
        """Points reconstructed from local registrations."""
        return self.local_registrations.map_values(lambda x: x.reconstructed)

    @cached_property
    def registrations_to_global_atlas(self):
        """Registrations from local atlases to the global atlas."""
        data = collect_registrations_to_global_atlas(
            self._output.dir_config,
            self._output.keys.keys(),
        )
        return self._transform(data)

    @cached_property
    def global_shoots(self):
        """Shoot results representing shapes at the global atlas."""
        data = collect_global_shoots(
            self._output.dir_config,
            self._output.representation_keys,
        )
        return self._transform(data)

    @cached_property
    def global_points(self):
        """Shape representations as points based at the global atlas."""
        return self.global_shoots.map_values(
            lambda x: x.point,
        )

    @cached_property
    def global_deltas(self):
        """Vertex displacements from the global atlas."""
        template_vertices = self._output.global_atlas_point.as_surface().vertices
        return self.global_points.map_values(
            lambda point: point.as_surface().vertices - template_vertices
        )

    @cached_property
    def transports(self):
        """Parallel-transport results from local to global atlases."""
        data = collect_transports(
            self._output.dir_config,
            self._output.representation_keys,
        )
        return self._transform(data)

    @cached_property
    def local_atlases(self):
        """Local atlas estimates."""
        data = collect_atlases(
            self._output.dir_config,
            self._output.keys,
        )
        return self._transform(data)

    @property
    def local_atlases_points(self):
        """Local atlas template points."""
        return self.local_atlases.map_values(lambda x: x.template)

    @property
    def global_atlas(self):
        """Global atlas estimate."""
        return self._output.global_atlas

    @property
    def global_atlas_point(self):
        """Global atlas template point."""
        return self._output.global_atlas_point

    @property
    def global_atlas_flows(self):
        """Flows associated with the global atlas."""
        return self._transform(Dataset(self.global_atlas.flows))


class LddmmToGlobalOutput:
    """Filesystem-backed output of an LDDMM-to-global run.

    Parameters
    ----------
    path : path-like
        Directory containing the protocol outputs.
    """

    def __init__(self, path):
        self.path = Path(path)

    @cached_property
    def mapped_view(self):
        """View using the keys stored by the protocol."""
        return LddmmToGlobalOutputView(self, key_map=None)

    @cached_property
    def view(self):
        """View using the source keys when a key map is available."""
        key_map = self.params["metadata"].get("key_map")
        if key_map is not None:
            key_map = NestedKeyMap(**key_map).invert()

        return LddmmToGlobalOutputView(self, key_map=key_map)

    @cached_property
    def params(self):
        """Protocol parameters loaded from disk."""
        return load_json(self.path / "params.json")

    @cached_property
    def results(self):
        """Protocol results loaded from disk."""
        return load_json(self.path / "results.json")

    @cached_property
    def dir_config(self):
        """Directory configuration for persisted protocol artifacts."""
        return LddmmPaths(
            root=self.path,
            **{key: self.path / value for key, value in self.params["dirs"].items()},
        )

    @cached_property
    def keys(self):
        """Nested keys stored by the protocol."""
        return self.params["keys"]

    @cached_property
    def atlas_only_keys(self):
        """Nested keys used only for atlas estimation."""
        return self.params.get("atlas_only_keys", {})

    @cached_property
    def representation_keys(self):
        """Nested keys with global shape representations."""
        return {
            outer_key: [
                key
                for key in keys
                if key not in self.atlas_only_keys.get(outer_key, ())
            ]
            for outer_key, keys in self.keys.items()
        }

    @cached_property
    def global_atlas(self):
        """Global atlas loaded from persisted outputs."""
        return get_global_atlas(self.dir_config)

    @property
    def global_atlas_point(self):
        """Template point of the global atlas."""
        return self.global_atlas.template

    def instantiate_varifold_metric(self, engine="auto"):
        """Instantiate the varifold metric used by the protocol."""
        from polpo.surface_mesh.varifold.geometry import VarifoldMetric

        return VarifoldMetric(
            sigma=self.results["kernel_tuning"]["attachment_kernel_width"],
            engine=engine,
        )

    def instantiate_euclidean_metric(self):
        """Instantiate the Euclidean metric on the global-atlas mesh."""
        from polpo.surface_mesh.euclidean import EuclideanSurfaces

        return EuclideanSurfaces(
            faces=self.global_atlas_point.as_surface().faces
        ).metric

    def instantiate_lddmm_metric(self):
        """Instantiate the LDDMM metric used by the protocol."""
        from ._metric import instantiate_lddmm_metric

        tuning = self.results["kernel_tuning"]
        config = self.params["metric"]

        return instantiate_lddmm_metric(
            self.path,
            kernel_width=tuning["kernel_width"],
            attachment_kernel_width=tuning["attachment_kernel_width"],
            regularization=config["regularization"],
            max_iter=config["max_iter"],
            tol=config["tol"],
        )


class MultiPoint:
    """Collection of corresponding points from multiple protocol outputs.

    Parameters
    ----------
    points : sequence
        Corresponding points to combine.
    """

    def __init__(self, points):
        self._points = points

    def as_polydata(self):
        """Convert the points to a merged PyVista PolyData."""
        return pv.merge([point.as_surface().to_polydata() for point in self._points])

    def as_pv_surface(self):
        """Convert the points to a merged PyVista surface."""
        return PvSurface(self.as_polydata())


class LddmmToGlobalMultiOutputView(_OutputView):
    """View combining corresponding data from multiple protocol outputs."""

    def _zip_and_transform(self, data):
        data = NestedDataset.zip_many(data, func=lambda points: MultiPoint(points))
        return self._transform(data)

    @cached_property
    def dataset(self):
        """Aligned meshes combined across outputs."""
        return self._zip_and_transform(
            [output.mapped_view.dataset for output in self._output]
        )

    @cached_property
    def local_reconstructed_points(self):
        """Locally reconstructed points combined across outputs."""
        return self._zip_and_transform(
            [output.mapped_view.local_reconstructed_points for output in self._output]
        )

    @cached_property
    def global_points(self):
        """Global shape representations combined across outputs."""
        return self._zip_and_transform(
            [output.mapped_view.global_points for output in self._output]
        )


class LddmmToGlobalMultiOutput:
    """Collection of LDDMM-to-global outputs over a shared key domain.

    Parameters
    ----------
    outputs : sequence of LddmmToGlobalOutput
        Outputs to combine.
    """

    def __init__(self, outputs):
        self.outputs = outputs

    @cached_property
    def mapped_view(self):
        """Combined view using the keys stored by each protocol."""
        return LddmmToGlobalMultiOutputView(self, key_map=None)

    @cached_property
    def view(self):
        """Combined view using source keys when a key map is available."""
        return LddmmToGlobalMultiOutputView(
            self,
            key_map=self[0].view.key_map,
        )

    def __iter__(self):
        """Iterate over the individual protocol outputs."""
        return iter(self.outputs)

    def __getitem__(self, index):
        """Return an individual protocol output."""
        return self.outputs[index]
