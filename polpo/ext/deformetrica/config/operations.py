"""Configuration of LDDMM operations and Deformetrica estimation workflows."""

from .attachment import SurfaceAttachmentConfig
from .base import BaseConfig, OptionsConfig
from .mixins import DeformationConfigMixin, ExecutionConfigMixin
from .models import (
    DeterministicAtlasModelConfig,
    FixedTemplateDeterministicAtlasModelConfig,
    GeodesicRegressionModelConfig,
    SplineRegressionModelConfig,
)
from .optimization import ScipyLbfgsConfig


class LddmmConfig:
    """Configuration shared across LDDMM operations."""

    def __init__(self):
        self._registration = RegistrationConfig()
        self._shoot = ShootConfig()
        self._parallel_transport = ParallelTransportConfig()

    def set_deformation(
        self,
        kernel_width=None,
        use_svf=None,
        preserve_volume=None,
    ):
        """Set deformation parameters across LDDMM operations."""
        for config in (
            self._parallel_transport,
            self._registration.model,
            self._shoot,
        ):
            config.set_deformation(
                kernel_width=kernel_width,
                use_svf=use_svf,
                preserve_volume=preserve_volume,
            )

        return self

    def set_integration(
        self,
        number_of_time_steps=None,
        shooting_method=None,
        flow_method=None,
    ):
        """Set integration parameters across LDDMM operations."""
        transport = self._parallel_transport.get_method_config()
        if self._parallel_transport.method == "fanning":
            transport.set_integration(
                time_steps_per_unit_time=number_of_time_steps,
                post_transport_number_of_time_steps=number_of_time_steps,
                transported_shooting_method=shooting_method,
                flow_method=flow_method,
            )

        elif self._parallel_transport.method == "pole_ladder":
            transport.set_integration(
                time_steps_per_unit_time=number_of_time_steps,
            )

        self._registration.model.set_integration(
            number_of_time_steps=number_of_time_steps,
            shooting_method=shooting_method,
            flow_method=flow_method,
        )
        self._shoot.set_integration(
            time_steps_per_unit_time=number_of_time_steps,
            shooting_method=shooting_method,
            flow_method=flow_method,
        )

        return self

    def set_attachment(self, **kwargs):
        """Set surface attachment parameters for registration."""
        self._registration.attachment.set_attachment(**kwargs)
        return self

    def set_optimization(self, **kwargs):
        """Set optimization parameters for registration."""
        self._registration.optimizer.set_optimization(**kwargs)
        return self

    def set_control_points(self, **kwargs):
        """Set control-point parameters for registration."""
        self._registration.model.set_control_points(**kwargs)
        return self

    def set_execution(
        self,
        kernel_type=None,
        device=None,
    ):
        """Set execution parameters across LDDMM operations."""
        for config in (
            self._registration.model,
            self._shoot,
            self._parallel_transport,
        ):
            config.set_execution(
                kernel_type=kernel_type,
                device=device,
            )

        return self

    def set_transport_method(self, method):
        """Set the parallel transport method."""
        self._parallel_transport.set_transport_method(method)

        transport = self._parallel_transport.get_method_config()

        transport.set_integration(
            time_steps_per_unit_time=self._shoot.time_steps_per_unit_time,
            flow_method=self._shoot.flow_method,
        )

        if method == "fanning":
            transport.set_integration(
                post_transport_number_of_time_steps=self._registration.model.number_of_time_steps,
                transported_shooting_method=self._shoot.shooting_method,
            )

        return self

    def get_registration_config(self):
        """Return the registration-specific configuration.

        Returns
        -------
        config : RegistrationConfig
            Configuration used for Deformetrica registration.
        """
        return self._registration

    def get_shoot_config(self):
        """Return the shooting-specific configuration.

        Returns
        -------
        config : ShootConfig
            Configuration used for Deformetrica geodesic shooting.
        """
        return self._shoot

    def get_parallel_transport_config(self):
        """Return the parallel-transport-specific configuration."""
        return self._parallel_transport


class BaseEstimationConfig(BaseConfig):
    """Base configuration for model-based Deformetrica estimation."""

    def build_model_options(self):
        """Build Deformetrica model options."""
        return self.model.to_options()

    def build_estimator_options(self):
        """Build Deformetrica estimator options."""
        return self.optimizer.to_options()

    def build_template_specifications(self, source):
        """Build Deformetrica template specifications."""
        return self.attachment.build_template_specifications(
            source,
            kernel_type=self.model.kernel_type,
            kernel_device=self.model._get_kernel_device(),
        )

    def build_cache_params(self):
        """Return parameters that determine the estimation result."""
        return {
            "model": self.model.build_cache_params(),
            "attachment": self.attachment.build_cache_params(),
            "optimizer": self.optimizer.build_cache_params(),
        }

    @classmethod
    def from_registration_config(cls, registration_config):
        """Create an estimation configuration from a registration configuration.

        Compatible model settings and surface attachment settings are copied from
        ``registration_config``. Optimizer settings are not copied, so the target
        estimation configuration retains its own optimization defaults.

        Parameters
        ----------
        registration_config : RegistrationConfig
            Registration configuration providing shared settings.

        Returns
        -------
        config : BaseEstimationConfig
            New estimation configuration initialized from the registration settings.
        """
        config = cls()

        config.model.update_from(registration_config.model)
        config.attachment.update_from(registration_config.attachment)

        config.verbosity = registration_config.verbosity

        return config


class RegistrationConfig(BaseEstimationConfig):
    """Configuration for Deformetrica surface registration."""

    def __init__(self):
        super().__init__()

        self.optimizer = ScipyLbfgsConfig().set_optimization(
            max_iter=200,
            print_every=20,
            tol=1e-5,
        )
        self.model = (
            FixedTemplateDeterministicAtlasModelConfig()
            .set_control_points(freeze=False, initial=None)
            .set_integration(
                number_of_time_steps=10,
                shooting_method="euler",
                flow_method="euler",
            )
        )
        self.attachment = SurfaceAttachmentConfig()

        self.verbosity = "WARNING"


class ShootConfig(DeformationConfigMixin, ExecutionConfigMixin, OptionsConfig):
    """Configuration for Deformetrica geodesic shooting."""

    def __init__(self):
        super().__init__()

        self.set_integration(
            time_steps_per_unit_time=10,
            shooting_method="euler",
            flow_method="euler",
        )

        self.write_adjoint_parameters = False

    def set_integration(
        self,
        time_steps_per_unit_time=None,
        shooting_method=None,
        flow_method=None,
    ):
        """Set integration-related shooting parameters.

        Parameters
        ----------
        time_steps_per_unit_time : int
            Number of time steps per unit of time used to discretize the
            geodesic during shooting.
        shooting_method : {"euler", "rk2", "rk4"}
            Numerical integration method used for shooting.
        flow_method : {"euler", "rk2"}
            Numerical integration method used to flow the shape.

        Returns
        -------
        config : ShootConfig
            This configuration.
        """
        if time_steps_per_unit_time is not None:
            self.time_steps_per_unit_time = time_steps_per_unit_time

        if shooting_method is not None:
            self.shooting_method = shooting_method

        if flow_method is not None:
            self.flow_method = flow_method

        self._validate_integration()
        return self

    def _validate_integration(self):
        if self.shooting_method not in {"euler", "rk2", "rk4"}:
            raise ValueError(f"Unknown shooting method {self.shooting_method!r}.")

        if self.flow_method not in {"euler", "rk2"}:
            raise ValueError(f"Unknown flow method {self.flow_method!r}.")

    def to_options(self):
        """Convert the configuration to Deformetrica keyword arguments."""
        return {
            "deformation_kernel_width": self.kernel_width,
            "deformation_kernel_type": self.kernel_type,
            "use_svf": self.use_svf,
            "preserve_volume": self.preserve_volume,
            "concentration_of_time_points": self.time_steps_per_unit_time,
            "use_rk2_for_shoot": self.shooting_method == "rk2",
            "use_rk4_for_shoot": self.shooting_method == "rk4",
            "use_rk2_for_flow": self.flow_method == "rk2",
            "write_adjoint_parameters": self.write_adjoint_parameters,
            "gpu_mode": self._get_gpu_mode(),
        }

    def _get_cache_exclusions(self):
        return {
            "gpu_mode",
            "write_adjoint_parameters",
        }


class FanningConfig(DeformationConfigMixin, ExecutionConfigMixin, OptionsConfig):
    """Configuration for Deformetrica fanning parallel transport."""

    def __init__(self):
        super().__init__()
        self.set_integration(
            time_steps_per_unit_time=10,
            post_transport_number_of_time_steps=10,
            transported_shooting_method="euler",
            flow_method="euler",
        )

    def set_integration(
        self,
        time_steps_per_unit_time=None,
        post_transport_number_of_time_steps=None,
        transported_shooting_method=None,
        flow_method=None,
    ):
        """Set numerical integration parameters for parallel transport.

        Parameters
        ----------
        time_steps_per_unit_time : int
            Number of time steps per unit of time used to discretize the
            reference geodesic along which the momentum is transported.
        post_transport_number_of_time_steps : int
            Number of time steps used to discretize each shooting step generated
            from the transported momentum.
        transported_shooting_method : {"euler", "rk2"}
            Numerical integration method used to evolve control points and
            momenta during post-transport shooting.
            The reference geodesic shooting is hard-coded to RK2 in Deformetrica.
        flow_method : {"euler", "rk2"}
            Numerical integration method used to evolve surface points under the
            deformation flow. The same method is used along the reference
            geodesic and during post-transport shooting.

        Returns
        -------
        config : FanningConfig
            This configuration.
        """
        if time_steps_per_unit_time is not None:
            self.time_steps_per_unit_time = time_steps_per_unit_time

        if post_transport_number_of_time_steps is not None:
            self.post_transport_number_of_time_steps = (
                post_transport_number_of_time_steps
            )

        if transported_shooting_method is not None:
            self.transported_shooting_method = transported_shooting_method

        if flow_method is not None:
            self.flow_method = flow_method

        self._validate_integration()
        return self

    def _validate_deformation(self):
        if self.use_svf:
            raise ValueError(
                "Parallel transport does not support stationary velocity fields."
            )

        if self.preserve_volume:
            raise ValueError(
                "Standard parallel transport does not support "
                "volume-preserving deformations."
            )

    def _validate_integration(self):
        if self.transported_shooting_method not in {"euler", "rk2"}:
            raise ValueError(
                f"Unknown shooting method {self.transported_shooting_method!r}."
            )

        if self.flow_method not in {"euler", "rk2"}:
            raise ValueError(f"Unknown flow method {self.flow_method!r}.")

    def to_options(self):
        """Convert the configuration to Deformetrica keyword arguments."""
        return {
            "deformation_kernel_width": self.kernel_width,
            "deformation_kernel_type": self.kernel_type,
            "concentration_of_time_points": self.time_steps_per_unit_time,
            "number_of_time_points": self.post_transport_number_of_time_steps + 1,
            "use_rk2_for_shoot": self.transported_shooting_method == "rk2",
            "use_rk2_for_flow": self.flow_method == "rk2",
            "gpu_mode": self._get_gpu_mode(),
        }

    def _get_cache_exclusions(self):
        return {"gpu_mode"}


class PoleLadderConfig(DeformationConfigMixin, ExecutionConfigMixin, OptionsConfig):
    """Configuration for Deformetrica pole-ladder transport."""

    def __init__(self):
        super().__init__()

        self.set_integration(
            time_steps_per_unit_time=10,
            flow_method="euler",
        )

    def set_integration(
        self,
        time_steps_per_unit_time=None,
        flow_method=None,
    ):
        """Set numerical integration parameters for pole-ladder transport.

        Parameters
        ----------
        time_steps_per_unit_time : int
            Number of time steps per unit of time used to discretize the
            reference geodesic.
        flow_method : {"euler", "rk2"}
            Numerical integration method used to evolve surface points under the
            deformation flow. The same method is used along the reference
            geodesic and during post-transport shooting.

        Returns
        -------
        config : PoleLadderConfig
            This configuration.
        """
        if time_steps_per_unit_time is not None:
            self.time_steps_per_unit_time = time_steps_per_unit_time

        if flow_method is not None:
            self.flow_method = flow_method

        self._validate_integration()
        return self

    def _validate_deformation(self):
        if self.use_svf:
            raise ValueError(
                "Pole-ladder transport does not support stationary " "velocity fields."
            )

        if self.preserve_volume:
            raise ValueError(
                "Standard pole-ladder transport does not support "
                "volume-preserving deformations."
            )

    def _validate_integration(self):
        if self.flow_method not in {"euler", "rk2"}:
            raise ValueError(f"Unknown flow method {self.flow_method!r}.")

    def to_options(self):
        """Convert the configuration to Deformetrica keyword arguments."""
        return {
            "deformation_kernel_width": self.kernel_width,
            "deformation_kernel_type": self.kernel_type,
            "concentration_of_time_points": self.time_steps_per_unit_time,
            "number_of_time_points": self.time_steps_per_unit_time + 1,
            "use_rk2_for_flow": self.flow_method == "rk2",
            "gpu_mode": self._get_gpu_mode(),
        }

    def _get_cache_exclusions(self):
        return {"gpu_mode"}


class ParallelTransportConfig(OptionsConfig):
    """Configuration for LDDMM parallel transport."""

    _CONFIGS = {
        "fanning": FanningConfig,
        "pole_ladder": PoleLadderConfig,
    }

    def __init__(self):
        super().__init__()
        self._config = None
        self.method = None
        self.set_transport_method("fanning")

    def set_transport_method(self, method):
        """Set the parallel transport method.

        Parameters
        ----------
        method : {"fanning", "pole_ladder"}
            Parallel transport method.

        Returns
        -------
        config : ParallelTransportConfig
            This configuration.
        """
        if method not in self._CONFIGS:
            raise ValueError(f"Unknown transport method {method!r}.")

        if method == self.method:
            return self

        old_config = self._config
        new_config = self._CONFIGS[method]()

        if old_config is not None:
            new_config.set_deformation(
                kernel_width=old_config.kernel_width,
                use_svf=old_config.use_svf,
                preserve_volume=old_config.preserve_volume,
            )

            new_config.set_execution(
                kernel_type=old_config.kernel_type,
                device=old_config.device,
            )

            new_config.set_integration(
                time_steps_per_unit_time=old_config.time_steps_per_unit_time,
                flow_method=old_config.flow_method,
            )

        self._config = new_config
        self.method = method

        return self

    def set_deformation(self, **kwargs):
        """Set deformation parameters for the active transport method."""
        self._config.set_deformation(**kwargs)
        return self

    def set_execution(self, **kwargs):
        """Set execution parameters for the active transport method."""
        self._config.set_execution(**kwargs)
        return self

    def set_integration(self, **kwargs):
        """Set integration parameters for the active transport method."""
        self._config.set_integration(**kwargs)
        return self

    def get_method_config(self):
        """Return the configuration for the active transport method.

        Returns
        -------
        config : FanningConfig or PoleLadderConfig
            Configuration for the active parallel transport method.
        """
        return self._config

    def to_options(self):
        """Convert the active configuration to Deformetrica keyword arguments."""
        return self._config.to_options()

    def build_cache_params(self):
        """Return parameters that determine the parallel transport result."""
        return {
            "method": self.method,
            **self._config.build_cache_params(),
        }


class DeterministicAtlasConfig(BaseEstimationConfig):
    """Configuration for Deformetrica deterministic atlas estimation."""

    def __init__(self):
        self.optimizer = ScipyLbfgsConfig().set_optimization(
            max_iter=200,
            print_every=20,
            tol=1e-5,
            max_line_search_iterations=50,
        )

        self.model = (
            DeterministicAtlasModelConfig()
            .set_template(
                freeze=False,
                use_sobolev_gradient=True,
                sobolev_kernel_width_ratio=1.0,
            )
            .set_control_points(
                freeze=False,
                initial=None,
            )
            .set_integration(
                number_of_time_steps=10,
                shooting_method="euler",
                flow_method="euler",
            )
        )

        self.attachment = SurfaceAttachmentConfig()

        self.verbosity = "WARNING"


class GeodesicRegressionConfig(BaseEstimationConfig):
    """Configuration for Deformetrica geodesic regression."""

    def __init__(self):
        super().__init__()
        self.optimizer = ScipyLbfgsConfig().set_optimization(
            max_iter=200,
            print_every=20,
            tol=1e-5,
        )
        self.model = (
            GeodesicRegressionModelConfig()
            .set_control_points(
                freeze=False,
            )
            .set_integration(
                number_of_time_steps=10,
                shooting_method="euler",
                flow_method="euler",
            )
            .set_regression(t0=0.0)
        )
        self.attachment = SurfaceAttachmentConfig()

        self.verbosity = "WARNING"

    def build_model_options(self, times):
        """Build model options for the observed time points.

        Parameters
        ----------
        times : array-like
            Observation times used to validate the regression reference time.

        Returns
        -------
        options : dict
            Deformetrica geodesic-regression model options.
        """
        self.model.validate(times)
        return self.model.to_options()


class SplineRegressionConfig(BaseEstimationConfig):
    """Configuration for Deformetrica spline regression."""

    def __init__(self):
        super().__init__()

        self.optimizer = ScipyLbfgsConfig().set_optimization(
            max_iter=200,
            print_every=20,
            tol=1e-5,
        )
        self.model = (
            SplineRegressionModelConfig()
            .set_control_points(
                freeze=False,
            )
            .set_integration(
                number_of_time_steps=10,
                flow_method="euler",
            )
            .set_regression(
                t0=0.0,
                geodesic_weight=0.1,
            )
        )
        self.attachment = SurfaceAttachmentConfig()

        self.verbosity = "WARNING"

    def build_model_options(self, times, weights=None):
        """Build model options for the observed time points and weights.

        Parameters
        ----------
        times : array-like
            Observation times used to validate the regression reference time.
        weights : array-like
            Weights associated with the observations. If omitted, all observations
            receive unit weight.

        Returns
        -------
        options : dict
            Deformetrica spline-regression model options.
        """
        self.model.validate(times)

        if weights is None:
            weights = [1.0] * len(times)

        return {
            **self.model.to_options(),
            "target_weights": weights,
        }
