import geomstats.backend as gs
from geomfum.convert import P2pFromFmConverter
from geomfum.descriptor.pipeline import (
    DescriptorPipeline,
    L2InnerNormalizer,
)
from geomfum.descriptor.spectral import WaveKernelSignature
from geomfum.functional_map import (
    FactorSum,
    LBCommutativityEnforcing,
    OperatorCommutativityEnforcing,
    SpectralDescriptorPreservation,
)
from geomfum.numerics.optimization import ScipyMinimize
from geomfum.refine import IcpRefiner, Refiner, ZoomOut
from geomfum.shape import TriangleMesh
from geomstats.geometry.stratified.vectorization import vectorize_point


@vectorize_point((0, "point"))
def vertices_to_geomfum(point):
    return [TriangleMesh(point_.vertices, point_.faces) for point_ in point]


class SequentialRefiner(Refiner):
    # TODO: bring to geomfum
    def __init__(self, refiners):
        super().__init__()
        self.refiners = refiners

    def __call__(self, fmap_matrix, basis_a, basis_b):
        for refiner in self.refiners:
            fmap_matrix = refiner(fmap_matrix, basis_a, basis_b)

        return fmap_matrix


class FmAlignerAlgorithm:
    def __init__(
        self,
        use_k=10,
        spectrum_size=30,
        descriptor_pipe=None,
        objective_builder=None,
        optimizer=None,
        p2p_converter=None,
        refiner=None,
    ):
        # TODO: pass objective as
        if descriptor_pipe is None:
            steps = [
                WaveKernelSignature.from_registry(n_domain=10),
                L2InnerNormalizer(),
            ]
            descriptor_pipe = DescriptorPipeline(steps)

        if objective_builder is None:

            def objective_builder(mesh_a, mesh_b, descr_a, descr_b):
                factors = [
                    SpectralDescriptorPreservation(
                        mesh_a.basis.project(descr_a),
                        mesh_b.basis.project(descr_b),
                        weight=1.0,
                    ),
                    LBCommutativityEnforcing.from_bases(
                        mesh_a.basis,
                        mesh_b.basis,
                        weight=1e-2,
                    ),
                    OperatorCommutativityEnforcing.from_multiplication(
                        mesh_a.basis, descr_a, mesh_b.basis, descr_b, weight=1e-1
                    ),
                ]

                return FactorSum(factors)

        if optimizer is None:
            optimizer = ScipyMinimize(
                method="L-BFGS-B",
            )

        if p2p_converter is None:
            p2p_converter = P2pFromFmConverter()

        if refiner is None:
            refiner = SequentialRefiner([IcpRefiner(nit=5), ZoomOut(nit=6, step=2)])

        self.use_k = use_k
        self.spectrum_size = spectrum_size
        self.descriptor_pipe = descriptor_pipe
        self.objective_builder = objective_builder
        self.optimizer = optimizer
        self.p2p_converter = p2p_converter
        self.refiner = refiner

    def align(self, point, base_point):
        """Align point to base point."""
        mesh_a, mesh_b = point, base_point

        mesh_a.laplacian.find_spectrum(
            spectrum_size=self.spectrum_size, set_as_basis=True
        )
        mesh_b.laplacian.find_spectrum(
            spectrum_size=self.spectrum_size, set_as_basis=True
        )

        mesh_a.basis.use_k = self.use_k
        mesh_b.basis.use_k = self.use_k

        descr_a = self.descriptor_pipe.apply(mesh_a)
        descr_b = self.descriptor_pipe.apply(mesh_b)

        objective = self.objective_builder(mesh_a, mesh_b, descr_a, descr_b)

        x0 = gs.zeros((mesh_b.basis.spectrum_size, mesh_a.basis.spectrum_size))

        res = self.optimizer.minimize(
            objective,
            x0,
            fun_jac=objective.gradient,
        )

        fmap = res.x.reshape(x0.shape)

        fmap = self.refiner(fmap, mesh_a.basis, mesh_b.basis)
        p2p = self.p2p_converter(fmap, mesh_a.basis, mesh_b.basis)

        # TODO: review this
        aligned_mesh = TriangleMesh(mesh_a.vertices[p2p], mesh_b.faces)

        return aligned_mesh
