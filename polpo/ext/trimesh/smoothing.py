import trimesh


class TrimeshLaplacianSmoothing:
    """
    https://trimesh.org/trimesh.smoothing.html#trimesh.smoothing.filter_laplacian
    """

    def __init__(
        self,
        lamb=0.5,
        iterations=10,
        implicit_time_integration=False,
        volume_constraint=True,
        laplacian_operator=None,
        inplace=True,
    ):
        super().__init__()
        self.lamb = lamb
        self.iterations = iterations
        self.implicit_time_integration = implicit_time_integration
        self.volume_constraint = volume_constraint
        self.laplacian_operator = laplacian_operator
        self.inplace = inplace

    def __call__(self, mesh):
        if not self.inplace:
            mesh = mesh.copy()

        trimesh.smoothing.filter_laplacian(
            mesh,
            lamb=self.lamb,
            iterations=self.iterations,
            implicit_time_integration=self.implicit_time_integration,
            volume_constraint=self.volume_constraint,
            laplacian_operator=self.laplacian_operator,
        )

        return mesh
