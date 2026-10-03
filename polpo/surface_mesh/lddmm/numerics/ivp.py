class HamiltonianRK4:
    def __init__(self, n_steps=10):
        self.n_steps = n_steps

    def step(self, rhs, position, momentum, step_size):
        dq1, dp1 = rhs(position, momentum)

        dq2, dp2 = rhs(
            position + step_size / 2 * dq1,
            momentum + step_size / 2 * dp1,
        )

        dq3, dp3 = rhs(
            position + step_size / 2 * dq2,
            momentum + step_size / 2 * dp2,
        )

        dq4, dp4 = rhs(
            position + step_size * dq3,
            momentum + step_size * dp3,
        )

        position = position + step_size / 6 * (dq1 + 2 * dq2 + 2 * dq3 + dq4)
        momentum = momentum + step_size / 6 * (dp1 + 2 * dp2 + 2 * dp3 + dp4)

        return position, momentum

    def integrate(self, rhs, initial_position, initial_momentum, end_time=1.0):
        position = initial_position
        momentum = initial_momentum
        step_size = end_time / self.n_steps

        for _ in range(self.n_steps):
            position, momentum = self.step(
                rhs,
                position,
                momentum,
                step_size,
            )

        return position, momentum
