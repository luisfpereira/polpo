import numpy as np
import pytest

from polpo.ext.pyvista.surface_mesh import PvSurface
from polpo.ext.trimesh.surface_mesh import TrimeshSurface
from polpo.surface_mesh.core import Surface
from polpo.surface_mesh.generation.blob import create_blob
from polpo.testing.data import DataCase, LazyValue
from polpo.testing.decorators import materialize_lazy_values
from polpo.testing.parametrizers import DataBasedParametrizer

ATOL = 1e-6


class SurfaceTestCase:
    def _assert_all_close(self, values, atol=ATOL, strict=False):
        # NB: strict checks shape and dtype
        reference = values[0]

        for value in values[1:]:
            np.testing.assert_equal(reference.shape, value.shape)
            np.testing.assert_allclose(reference, value, atol=atol, strict=strict)

    def test_face_centroids(self, surfaces, atol=ATOL):
        face_centroids = [surface.face_centroids for surface in surfaces]
        self._assert_all_close(face_centroids, atol)

    def test_face_areas(self, surfaces, atol=ATOL):
        face_areas = [surface.face_areas for surface in surfaces]
        self._assert_all_close(face_areas, atol)

    def test_face_normals(self, surfaces, atol=ATOL):
        face_normals = [surface.face_normals for surface in surfaces]
        self._assert_all_close(face_normals, atol)


class SurfaceTestData(DataCase):
    def __init__(self, resolution=10):
        super().__init__()
        self.resolution = resolution

        self.surfaces = [
            (LazyValue(self._make_surfaces),),
        ]

    def _make_surfaces(self):
        # NB: pv does float32
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

    @pytest.mark.random
    def face_centroids_test_data(self):
        return self.surfaces

    @pytest.mark.random
    def face_areas_test_data(self):
        return self.surfaces

    @pytest.mark.random
    def face_normals_test_data(self):
        return self.surfaces


class TestSurface(SurfaceTestCase, metaclass=DataBasedParametrizer):
    testing_data = SurfaceTestData()
