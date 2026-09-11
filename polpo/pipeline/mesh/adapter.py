from polpo.preprocessing import IdentityStep, Map
from polpo.preprocessing.base import Pipeline


class PointCloudAdapter(Pipeline):
    def __init__(self, step, mesh2points, points2mesh):
        super().__init__(steps=[mesh2points, step, points2mesh])

    @classmethod
    def build_pipes(
        cls,
        template_faces,
        mesh2points=None,
        points2points=None,
        data2mesh=None,
        multiple=False,
    ):
        # converters default to pv if None

        # TODO: make template_faces optional/dynamic/setter?

        if mesh2points is None:
            mesh2points = lambda x: x.points

        if points2points is None:
            points2points = IdentityStep()

        if data2mesh is None:
            from polpo.preprocessing.mesh.conversion import PvFromData

            data2mesh = PvFromData()

        points2data = lambda points: (points, template_faces)

        if multiple:
            if not callable(multiple):
                multiple = Map

            points2points = multiple(points2points)
            points2data = multiple(points2data)
            data2mesh = multiple(data2mesh)

        points2mesh = points2points + points2data + data2mesh

        return mesh2points, points2mesh
