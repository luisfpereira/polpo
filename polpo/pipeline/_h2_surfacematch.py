import H2_SurfaceMatch.H2_match
import H2_SurfaceMatch.utils.utils

from polpo.preprocessing.base import PreprocessingStep


class H2MeshDecimator(PreprocessingStep):
    def __init__(self, decimation_factor=10.0):
        super().__init__()
        self.decimation_factor = decimation_factor

    def __call__(self, mesh):
        # TODO: issues using due to open3d (delete?)

        # TODO: make a check for colors?
        vertices, faces, colors = mesh

        n_faces_after_decimation = faces.shape[0] // self.decimation_factor
        (
            vertices_after_decimation,
            faces_after_decimation,
            colors_after_decimation,
        ) = H2_SurfaceMatch.utils.utils.decimate_mesh(
            vertices, faces, n_faces_after_decimation, colors=colors
        )

        return [
            vertices_after_decimation,
            faces_after_decimation,
            colors_after_decimation,
        ]


class H2MeshAligner(PreprocessingStep):
    def __init__(
        self,
        a0=0.01,
        a1=10.0,
        b1=10.0,
        c1=1.0,
        d1=0.0,
        a2=1.0,
        resolutions=0,
        paramlist=(),
    ):
        super().__init__()
        self.a0 = a0
        self.a1 = a1
        self.b1 = b1
        self.c1 = c1
        self.d1 = d1
        self.a2 = a2
        self.resolutions = resolutions
        self.paramlist = paramlist

        # TODO: allow control of device
        self.device = None

    def __call__(self, meshes):
        target_mesh, template_mesh = meshes

        # TODO: upgrade device?
        geod, F0, color0 = H2_SurfaceMatch.H2_match.H2MultiRes(
            source=template_mesh,
            target=target_mesh,
            a0=self.a0,
            a1=self.a1,
            b1=self.b1,
            c1=self.c1,
            d1=self.d1,
            a2=self.a2,
            resolutions=self.resolutions,
            start=None,
            paramlist=self.paramlist,
            device=self.device,
        )

        return geod[-1], F0, color0
