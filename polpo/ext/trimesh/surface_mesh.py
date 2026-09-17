import geomstats.backend as gs
import trimesh


class TrimeshSurface:
    def __init__(self, mesh):
        self.trimesh = mesh

    @classmethod
    def from_data(cls, vertices, faces, process=False):
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
        )
        return cls(mesh)

    @property
    def face_centroids(self):
        return gs.asarray(self.trimesh.triangles_center)

    @property
    def face_areas(self):
        return gs.asarray(self.trimesh.area_faces)

    @property
    def face_normals(self):
        return gs.asarray(self.trimesh.face_normals)

    @property
    def vertex_centroid(self):
        return self.trimesh.centroid
