"""Configuration of Deformetrica optimization algorithms."""

from .base import OptionsConfig


class ScipyLbfgsConfig(OptionsConfig):
    """Configuration for Deformetrica's SciPy L-BFGS optimizer."""

    def __init__(self):
        self.set_optimization(
            max_iter=200,
            tol=1e-5,
            print_every=20,
            memory_length=10,
            max_line_search_iterations=10,
        )

    def set_optimization(
        self,
        max_iter=None,
        tol=None,
        print_every=None,
        memory_length=None,
        max_line_search_iterations=None,
    ):
        """Set optimization-related parameters.

        Parameters
        ----------
        max_iter : int
            Maximum number of optimization iterations.
        print_every : int
            Number of optimization iterations between progress reports.
        tol : float
            Optimization convergence tolerance.
        memory_length : int
            Number of correction vectors retained by L-BFGS.
        max_line_search_iterations : int
            Maximum number of line-search iterations.

        Returns
        -------
        config : ScipyLbfgsConfig
            This configuration.
        """
        if max_iter is not None:
            self.max_iter = max_iter

        if tol is not None:
            self.tol = tol

        if print_every is not None:
            self.print_every = print_every

        if memory_length is not None:
            self.memory_length = memory_length

        if max_line_search_iterations is not None:
            self.max_line_search_iterations = max_line_search_iterations

        return self

    def to_options(self):
        """Convert the configuration to Deformetrica estimator options."""
        return {
            "optimization_method_type": "ScipyLBFGS",
            "individual_RER": {},
            "optimized_log_likelihood": "complete",
            #
            "max_iterations": self.max_iter,
            "convergence_tolerance": self.tol,
            "print_every_n_iters": self.print_every,
            "save_every_n_iters": 100,
            #
            "memory_length": self.memory_length,
            "max_line_search_iterations": self.max_line_search_iterations,
            #
            "verbose": 1,
            "callback": None,
            #
            "state_file": None,
            "load_state_file": False,
        }

    def _get_cache_exclusions(self):
        return {
            "optimization_method_type",
            "individual_RER",
            "optimized_log_likelihood",
            "print_every_n_iters",
            "save_every_n_iters",
            "verbose",
            "callback",
            "state_file",
            "load_state_file",
        }
