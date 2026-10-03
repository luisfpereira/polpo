import geomstats.backend as gs

from polpo.surface_mesh.deformetrica.core import TangentVector


class LDDMMMetric:
    def __init__(self, kernel_operator):
        self.kernel_operator = kernel_operator

    def _hamiltonian_derivatives(self, control_points, momenta):
        d_control_points = self.kernel_operator.kernel_prod(
            control_points,
            control_points,
            momenta,
        )

        d_momenta = -self.kernel_operator.kernel_grad_prod(
            control_points,
            control_points,
            momenta,
            momenta,
        )

        return d_control_points, d_momenta

    def velocity_at(self, x, tangent_vec):
        control_points = tangent_vec.control_points.as_array()
        momenta = tangent_vec.momenta.as_array()

        return self.kernel_operator.kernel_prod(
            x,
            control_points,
            momenta,
        )

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        # NB: base_point is ignored
        control_points = tangent_vec_a.control_points.as_array()
        momenta = tangent_vec_a.momenta.as_array()

        velocity = self.velocity_at(control_points, tangent_vec_b)

        return gs.sum(momenta * velocity, axis=(-2, -1))

    def squared_norm(self, tangent_vec, base_point=None):
        return self.inner_product(tangent_vec, tangent_vec, base_point)

    def norm(self, tangent_vec, base_point=None):
        return gs.sqrt(self.squared_norm(tangent_vec, base_point))

    def shoot(self, tangent_vec, end_time=1.0):
        control_points = tangent_vec.control_points.as_array()
        momenta = tangent_vec.momenta.as_array()

        control_points, momenta = self.integrator.integrate(
            self._hamiltonian_derivatives,
            control_points,
            momenta,
            end_time=end_time,
        )

        return TangentVector(
            control_points=control_points,
            momenta=momenta,
        )

    def exp(self, tangent_vec, base_point):
        pass
