import geomstats.backend as gs


class PvSurface:
    def __init__(self, pv_mesh, id_=None):
        # TODO: notion of native?
        self.id_ = id_

    @property
    def vertices(self):
        return gs.asarray(self._mesh.points)

    @property
    def faces(self):
        return gs.asarray(self._mesh.regular_faces)

    @property
    def face_areas(self):
        return gs.expand_dims(
            gs.asarray(self._mesh.compute_cell_sizes()["Area"]), axis=1
        )

    @property
    def face_centroids(self):
        return gs.asarray(
            self._mesh.cell_centers().points, dtype=gs.get_default_dtype()
        )

    @property
    def face_normals(self):
        return gs.asarray(self._mesh.face_normals, dtype=gs.get_default_dtype())

    @property
    def bounds(self):
        return gs.moveaxis(gs.reshape(gs.asarray(self._mesh.bounds), (3, 2)), 0, 1)

    @property
    def vertex_centroid(self):
        return gs.mean(self.vertices, axis=0)

    @property
    def edges(self):
        """Edges of the mesh.

        Returns
        -------
        edges : array-like, shape=[n_edges, 2]
        """
        vind012 = gs.concatenate([self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]])
        vind120 = gs.concatenate([self.faces[:, 1], self.faces[:, 2], self.faces[:, 0]])
        edges = gs.stack(
            [
                gs.concatenate([vind012, vind120]),
                gs.concatenate([vind120, vind012]),
            ],
            axis=-1,
        )
        edges = gs.unique(edges, axis=0)
        return edges[edges[:, 1] > edges[:, 0]]

    @property
    def edge_lengths(self):
        edge_points = self.vertices[self.edges]

        return gs.linalg.norm(edge_points[..., 0, :] - edge_points[..., 1, :], axis=-1)

    def as_pv(self):
        # TODO: rename/remove
        return self._mesh
