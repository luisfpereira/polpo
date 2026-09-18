"""Test scale estimators on a regular tetrahedron."""

import numpy as np
import pytest

from polpo.ext.pyvista.surface_mesh import PvSurface
from polpo.ext.trimesh.surface_mesh import TrimeshSurface
from polpo.surface_mesh.core import Surface
from polpo.surface_mesh.edge import normalize_edges
from polpo.surface_mesh.generation.blob import create_blob
from polpo.testing.data import DataCase, LazyValue
from polpo.testing.decorators import materialize_lazy_values
from polpo.testing.parametrizers import DataBasedParametrizer

ATOL = 1e-6


def _make_property_test(name, atol=ATOL):
    def test(self, surfaces):
        values = [getattr(surface, name) for surface in surfaces]
        self._assert_all_close(values, atol)

    test.__name__ = f"test_{name}"
    return test


class SurfaceTestCase:
    """Test consistency across surface mesh representations."""

    def _assert_all_close(self, values, atol=ATOL, strict=False):
        reference = values[0]

        for value in values[1:]:
            np.testing.assert_equal(reference.shape, value.shape)
            np.testing.assert_allclose(
                reference,
                value,
                atol=atol,
                strict=strict,
            )

    def test_edges(self, surfaces):
        edges = [normalize_edges(surface.edges) for surface in surfaces]
        self._assert_all_close(edges, atol=0)

    def test_edge_lengths(self, surfaces, atol=ATOL):
        edge_lengths = []

        for surface in surfaces:
            _, permutation = normalize_edges(
                surface.edges,
                return_permutation=True,
            )
            edge_lengths.append(surface.edge_lengths[permutation])

        self._assert_all_close(edge_lengths, atol)


for _name, _atol in {
    "vertices": ATOL,
    "faces": 0,
    "face_centroids": ATOL,
    "face_areas": ATOL,
    "face_normals": ATOL,
    "vertex_centroid": ATOL,
    "surface_centroid": ATOL,
    "bounds": ATOL,
}.items():
    setattr(
        SurfaceTestCase,
        f"test_{_name}",
        _make_property_test(_name, _atol),
    )


class BlobTestData(DataCase):
    """Test data for surface mesh representations."""

    def __init__(self, resolution=10):
        super().__init__()
        self.resolution = resolution

        self.surfaces = [
            (LazyValue(self._make_surfaces),),
        ]

    def _make_surfaces(self):
        # NB: pv returns float32
        polydata = create_blob(resolution=self.resolution)

        pv_surface = PvSurface(polydata)

        reference = Surface(pv_surface.vertices, pv_surface.faces)
        trimesh_mesh = TrimeshSurface.from_data(pv_surface.vertices, pv_surface.faces)
        return [
            reference,
            pv_surface,
            trimesh_mesh,
        ]

    def get_decorators(self):
        return [materialize_lazy_values]

    def get_data_methods(self):
        methods = super().get_data_methods()

        @pytest.mark.random
        def get_surfaces():
            return self.surfaces

        methods.update(
            {
                name: get_surfaces
                for name in (
                    "vertices",
                    "faces",
                    "face_centroids",
                    "face_areas",
                    "face_normals",
                    "vertex_centroid",
                    "surface_centroid",
                    "edges",
                    "edge_lengths",
                    "bounds",
                )
            }
        )

        return methods


class TestSurfaceWithBlob(SurfaceTestCase, metaclass=DataBasedParametrizer):
    """Test surface mesh representations."""

    testing_data = BlobTestData()
