import geomstats.backend as gs


class TrimeshSurface:
    # polymorphic to `Surface` and `Trimesh`
    def __init__(self, trimesh):
        # TODO: use notion of native?
        self.trimesh = trimesh

    @property
    def face_centroids(self):
        return gs.from_numpy(self._mesh.triangles_center)

    @property
    def face_areas(self):
        return gs.expand_dims(gs.from_numpy(self._mesh.area_faces), axis=1)

    @property
    def vertex_centroid(self):
        return gs.from_numpy(self._mesh.centroid)
