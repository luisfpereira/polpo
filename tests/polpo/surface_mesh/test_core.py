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


class SurfaceTestCase:
    """Test consistency across surface mesh representations."""

    def _assert_all_close(self, values, atol=ATOL, strict=False):
        # NB: strict checks shape and dtype
        reference = values[0]

        for value in values[1:]:
            np.testing.assert_equal(reference.shape, value.shape)
            np.testing.assert_allclose(reference, value, atol=atol, strict=strict)

    def test_vertices(self, surfaces, atol=ATOL):
        vertices = [surface.vertices for surface in surfaces]
        self._assert_all_close(vertices, atol)

    def test_faces(self, surfaces):
        faces = [surface.faces for surface in surfaces]
        self._assert_all_close(faces, atol=0)

    def test_face_centroids(self, surfaces, atol=ATOL):
        face_centroids = [surface.face_centroids for surface in surfaces]
        self._assert_all_close(face_centroids, atol)

    def test_face_areas(self, surfaces, atol=ATOL):
        face_areas = [surface.face_areas for surface in surfaces]
        self._assert_all_close(face_areas, atol)

    def test_face_normals(self, surfaces, atol=ATOL):
        face_normals = [surface.face_normals for surface in surfaces]
        self._assert_all_close(face_normals, atol)

    def test_vertex_centroid(self, surfaces, atol=ATOL):
        vertex_centroids = [surface.vertex_centroid for surface in surfaces]
        self._assert_all_close(vertex_centroids, atol)

    def test_surface_centroid(self, surfaces, atol=ATOL):
        surface_centroids = [surface.surface_centroid for surface in surfaces]
        self._assert_all_close(surface_centroids, atol)

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

    def test_bounds(self, surfaces, atol=ATOL):
        bounds = [surface.bounds for surface in surfaces]
        self._assert_all_close(bounds, atol=atol)


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
