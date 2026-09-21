from functools import cached_property
from pathlib import Path

import pyvista as pv

from polpo.dataset import Dataset, NestedDataset, NestedKeyMap
from polpo.ext.pyvista.surface_mesh import PvSurface
from polpo.io.json import load_json
from polpo.surface_mesh.deformetrica.paths import LddmmPaths

from .collect import (
    collect_atlases,
    collect_dataset,
    collect_global_shoots,
    collect_local_registrations,
    collect_registrations_to_global_atlas,
    collect_transports,
    get_global_atlas,
)

# TODO: some code to read errors e.g. missing meshes/failed registrations


class _OutputView:
    def __init__(self, output, key_map=None):
        self._output = output
        self.key_map = key_map

    def _transform(self, data):
        if self.key_map is not None:
            data = data.map_keys(self.key_map)

        return data

    def with_key_map(self, key_map):
        if self.key_map is not None:
            key_map = self.key_map.chain_with(key_map)
        return type(self)(self._output, key_map=key_map)


class LddmmToGlobalOutputView(_OutputView):
    @cached_property
    def dataset(self):
        # original meshes after rigid alignment
        data = collect_dataset(
            self._output.dir_config,
            self._output.encoded_keys,
        )
        return self._transform(data)

    @cached_property
    def local_registrations(self):
        # reconstructed results from local to observations
        data = collect_local_registrations(
            self._output.dir_config,
            self._output.encoded_keys,
        )
        return self._transform(data)

    @cached_property
    def local_reconstructed_points(self):
        return self.local_registrations.map_values(lambda x: x.reconstructed)

    @cached_property
    def registrations_to_global_atlas(self):
        # reconstructed results from local to global atlas
        data = collect_registrations_to_global_atlas(
            self._output.dir_config,
            self._output.encoded_keys.keys(),
        )
        return self._transform(data)

    @cached_property
    def global_shoots(self):
        data = collect_global_shoots(
            self._output.dir_config,
            self._output.encoded_keys,
        )
        return self._transform(data)

    @cached_property
    def global_points(self):
        return self.global_shoots.map_values(
            lambda x: x.point,
        )

    @cached_property
    def global_deltas(self):
        template_vertices = self._output.global_atlas_point.as_polydata().points
        return self.global_points.map_values(
            lambda point: point.as_polydata().points - template_vertices
        )

    @cached_property
    def transports(self):
        data = collect_transports(
            self._output.dir_config,
            self._output.encoded_keys,
        )
        return self._transform(data)

    @cached_property
    def local_atlases(self):
        data = collect_atlases(
            self._output.dir_config,
            self._output.encoded_keys,
        )
        return self._transform(data)

    @property
    def local_atlases_points(self):
        return self.local_atlases.map_values(lambda x: x.template)

    @property
    def global_atlas(self):
        return self._output.global_atlas

    @property
    def global_atlas_point(self):
        return self._output.global_atlas_point

    @property
    def global_atlas_flows(self):
        return self._transform(Dataset(self.global_atlas.flows))


class LddmmToGlobalOutput:
    def __init__(self, path):
        self.path = Path(path)

    @cached_property
    def encoded(self):
        return LddmmToGlobalOutputView(self, key_map=None)

    @cached_property
    def decoded(self):
        return LddmmToGlobalOutputView(self, key_map=self.key_map.invert())

    @cached_property
    def params(self):
        return load_json(self.path / "params.json")

    @cached_property
    def results(self):
        return load_json(self.path / "results.json")

    @cached_property
    def dir_config(self):
        return LddmmPaths(
            outputs_dir=self.path,
            **{key: self.path / value for key, value in self.params["dirs"].items()},
        )

    @cached_property
    def key_map(self):
        return NestedKeyMap.from_dict(self.params["metadata"]["key_map"])

    @cached_property
    def encoded_keys(self):
        return self.key_map.keys(encoded=True)

    @cached_property
    def global_atlas(self):
        return get_global_atlas(self.dir_config)

    @property
    def global_atlas_point(self):
        return self.global_atlas.template


class MultiPoint:
    def __init__(self, points):
        self._points = points

    def as_polydata(self):
        return pv.merge([point.as_polydata() for point in self._points])

    def as_pv_surface(self):
        return PvSurface(self.as_polydata())


class LddmmToGlobalMultiOutputView(_OutputView):
    def _zip_and_transform(self, data):
        data = NestedDataset.zip_many(data, func=lambda points: MultiPoint(points))
        return self._transform(data)

    @cached_property
    def dataset(self):
        return self._zip_and_transform(
            [output.encoded.dataset for output in self._output]
        )

    @cached_property
    def local_reconstructed_points(self):
        return self._zip_and_transform(
            [output.encoded.local_reconstructed_points for output in self._output]
        )

    @cached_property
    def global_points(self):
        return self._zip_and_transform(
            [output.encoded.global_points for output in self._output]
        )


class LddmmToGlobalMultiOutput:
    def __init__(self, outputs):
        self.outputs = outputs

    @cached_property
    def encoded(self):
        return LddmmToGlobalMultiOutputView(self, key_map=None)

    @cached_property
    def decoded(self):
        return LddmmToGlobalMultiOutputView(self, key_map=self.key_map.invert())

    def __iter__(self):
        return iter(self.outputs)

    def __getitem__(self, index):
        return self.outputs[index]

    @property
    def key_map(self):
        return self[0].key_map
