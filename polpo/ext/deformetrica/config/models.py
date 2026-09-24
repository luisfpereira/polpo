"""Configuration of Deformetrica atlas and regression models."""

from .base import OptionsConfig
from .mixins import DeformationConfigMixin, ExecutionConfigMixin


class BaseDeterministicAtlasModelConfig(
    DeformationConfigMixin,
    ExecutionConfigMixin,
    OptionsConfig,
):
    """Configuration for Deformetrica's deterministic atlas model.

    This class contains parameters controlling construction and initialization
    of ``DeterministicAtlas``. Estimator and optimization parameters are kept
    separate from the model configuration.
    """

    def __init__(self):
        super().__init__()

        self.initial_control_points = None
        self.set_control_points(
            freeze=False,
            initial=None,
        )

        self.set_integration(
            number_of_time_steps=10,
            shooting_method="euler",
            flow_method="euler",
        )

        self._dimension = None
        self._dense_mode = False
        self._number_of_processes = 1

        self.freeze_template = False
        self.use_sobolev_gradient = True
        self.sobolev_kernel_width_ratio = 1.0

        self._initial_cp_spacing = None

        self._freeze_momenta = False
        self._initial_momenta = None

        self._shoot_kernel_type = None

        self._random_seed = None

    def set_control_points(
        self,
        freeze=None,
        initial=None,
    ):
        """Set control-point model parameters.

        Parameters
        ----------
        freeze : bool
            Whether to keep control points fixed during estimation.
        initial : path-like
            Path to initial control points.

        Returns
        -------
        config : BaseDeterministicAtlasModelConfig
            This configuration.
        """
        if freeze is not None:
            self.freeze_control_points = freeze

        if initial is not None:
            self.initial_control_points = initial

        return self

    def set_integration(
        self,
        number_of_time_steps=None,
        shooting_method=None,
        flow_method=None,
    ):
        """Set numerical integration parameters.

        Parameters
        ----------
        number_of_time_steps : int
            Number of time steps used to discretize the deformation.
        shooting_method : {"euler", "rk2", "rk4"}
            Numerical method used to evolve control points and momenta.
        flow_method : {"euler", "rk2"}
            Numerical method used to flow template points.

        Returns
        -------
        config : BaseDeterministicAtlasModelConfig
            This configuration.
        """
        if number_of_time_steps is not None:
            self.number_of_time_steps = number_of_time_steps

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

    def _validate_template(self):
        if self.freeze_template and self.use_sobolev_gradient:
            raise ValueError(
                "Sobolev gradients cannot be used when the template is frozen."
            )

        if self.use_sobolev_gradient and self.sobolev_kernel_width_ratio is None:
            raise ValueError(
                "sobolev_kernel_width_ratio is required when "
                "use_sobolev_gradient=True."
            )

        if (
            not self.use_sobolev_gradient
            and self.sobolev_kernel_width_ratio is not None
        ):
            raise ValueError(
                "sobolev_kernel_width_ratio must be None when "
                "use_sobolev_gradient=False."
            )

    def to_options(self):
        """Convert the configuration to Deformetrica model options."""
        options = {
            "dimension": self._dimension,
            "dense_mode": self._dense_mode,
            "number_of_processes": self._number_of_processes,
            #
            "deformation_kernel_type": self.kernel_type,
            "deformation_kernel_width": self.kernel_width,
            "shoot_kernel_type": self._shoot_kernel_type,
            #
            "number_of_time_points": self.number_of_time_steps + 1,
            "use_rk2_for_shoot": self.shooting_method == "rk2",
            "use_rk4_for_shoot": self.shooting_method == "rk4",
            "use_rk2_for_flow": self.flow_method == "rk2",
            #
            "freeze_template": self.freeze_template,
            "use_sobolev_gradient": self.use_sobolev_gradient,
            #
            "initial_control_points": self.initial_control_points,
            "freeze_control_points": self.freeze_control_points,
            "initial_cp_spacing": self._initial_cp_spacing,
            #
            "initial_momenta": self._initial_momenta,
            "freeze_momenta": self._freeze_momenta,
            #
            "use_svf": self.use_svf,
            "preserve_volume": self.preserve_volume,
            #
            "gpu_mode": self._get_gpu_mode(),
            #
            "random_seed": self._random_seed,
        }

        if self.sobolev_kernel_width_ratio is not None:
            options["sobolev_kernel_width_ratio"] = self.sobolev_kernel_width_ratio

        return options

    def _get_cache_exclusions(self):
        return {
            "dimension",
            "dense_mode",
            "number_of_processes",
            "shoot_kernel_type",
            "initial_cp_spacing",
            "initial_momenta",
            "freeze_momenta",
            "gpu_mode",
        }


class DeterministicAtlasModelConfig(BaseDeterministicAtlasModelConfig):
    """Configuration for Deformetrica deterministic atlas estimation.

    This configuration exposes template optimization in addition to the common
    deformation, execution, control-point, and integration settings.
    """

    def set_template(
        self,
        freeze=None,
        use_sobolev_gradient=None,
        sobolev_kernel_width_ratio=None,
    ):
        """Set template-related model parameters.

        Parameters
        ----------
        freeze : bool
            Whether to keep the template fixed during estimation.
        use_sobolev_gradient : bool
            Whether to smooth template gradients with a Sobolev kernel.
        sobolev_kernel_width_ratio : float
            Ratio between the Sobolev smoothing kernel width and the
            deformation kernel width.

        Returns
        -------
        config : DeterministicAtlasModelConfig
            This configuration.
        """
        if freeze is not None:
            self.freeze_template = freeze

        if use_sobolev_gradient is not None:
            self.use_sobolev_gradient = use_sobolev_gradient

            if not self.use_sobolev_gradient:
                self.sobolev_kernel_width_ratio = None

        if sobolev_kernel_width_ratio is not None:
            self.sobolev_kernel_width_ratio = sobolev_kernel_width_ratio

        self._validate_template()
        return self

    def update_from(self, config):
        """Update common model settings from a fixed-template configuration.

        The deformation, execution, control-point, and integration settings are
        copied from ``config``.

        Template settings are not updated. In particular, ``freeze_template``,
        ``use_sobolev_gradient``, and ``sobolev_kernel_width_ratio`` retain their
        current values.

        Parameters
        ----------
        config : FixedTemplateDeterministicAtlasModelConfig
            Configuration providing the settings to copy.

        Returns
        -------
        config : DeterministicAtlasModelConfig
            This configuration.
        """
        if not isinstance(config, FixedTemplateDeterministicAtlasModelConfig):
            raise TypeError(
                "Expected a FixedTemplateDeterministicAtlasModelConfig, "
                f"got {type(config).__name__}."
            )

        self.set_deformation(
            kernel_width=config.kernel_width,
            use_svf=config.use_svf,
            preserve_volume=config.preserve_volume,
        )
        self.set_execution(
            kernel_type=config.kernel_type,
            device=config.device,
        )
        self.set_control_points(
            freeze=config.freeze_control_points,
            initial=config.initial_control_points,
        )
        self.set_integration(
            number_of_time_steps=config.number_of_time_steps,
            shooting_method=config.shooting_method,
            flow_method=config.flow_method,
        )

        return self


class FixedTemplateDeterministicAtlasModelConfig(BaseDeterministicAtlasModelConfig):
    """Configuration for a fixed-template deterministic atlas model.

    This configuration is used directly for registration and provides the
    fixed-template deformation settings reused by regression configurations.
    Template optimization and Sobolev smoothing are disabled.
    """

    def __init__(self):
        super().__init__()

        self.freeze_template = True
        self.use_sobolev_gradient = False
        self.sobolev_kernel_width_ratio = None

    def _get_cache_exclusions(self):
        extra_exclusions = {
            "freeze_template",
            "use_sobolev_gradient",
            "sobolev_kernel_width_ratio",
        }
        return super()._get_cache_exclusions() | extra_exclusions


class BaseRegressionModelConfig(OptionsConfig):
    """Base configuration shared by regression models."""

    def __init__(self):
        self._base = FixedTemplateDeterministicAtlasModelConfig()

        self.set_control_points(
            freeze=False,
            initial=None,
        )
        self.set_integration(number_of_time_steps=10)

        self.set_regression(t0=0.0)

    def set_deformation(self, kernel_width=None):
        """Set deformation parameters.

        Parameters
        ----------
        kernel_width : float
            Width of the deformation kernel.

        Returns
        -------
        config : BaseRegressionModelConfig
            This configuration.
        """
        self._base.set_deformation(
            kernel_width=kernel_width,
            use_svf=False,
            preserve_volume=False,
        )
        return self

    def set_execution(
        self,
        kernel_type=None,
        device=None,
    ):
        """Set execution parameters.

        Parameters
        ----------
        kernel_type : {"torch", "keops"}
            Backend used for kernel computations.
        device : {"auto", "cpu", "cuda"}
            Device used for computation.

        Returns
        -------
        config : BaseRegressionModelConfig
            This configuration.
        """
        self._base.set_execution(
            kernel_type=kernel_type,
            device=device,
        )
        return self

    def set_control_points(
        self,
        freeze=None,
        initial=None,
    ):
        """Set control-point parameters.

        Parameters
        ----------
        freeze : bool
            Whether to keep control points fixed during estimation.
        initial : path-like
            Path to initial control points.

        Returns
        -------
        config : BaseRegressionModelConfig
            This configuration.
        """
        self._base.set_control_points(
            freeze=freeze,
            initial=initial,
        )
        return self

    def set_integration(self, number_of_time_steps=None):
        """Set integration parameters shared by regression models.

        Parameters
        ----------
        number_of_time_steps : int
            Number of time steps used to discretize the trajectory.

        Returns
        -------
        config : BaseRegressionModelConfig
            This configuration.
        """
        self._base.set_integration(
            number_of_time_steps=number_of_time_steps,
        )

        return self

    def set_regression(self, t0=None):
        """Set parameters shared by regression models."""
        if t0 is not None:
            self.t0 = t0

        return self

    def validate(self, times):
        """Validate the configuration against observation times."""
        tmin = min(times)
        tmax = max(times)

        if not tmin <= self.t0 <= tmax:
            raise ValueError(
                f"Expected t0 to lie in the observation interval "
                f"[{tmin}, {tmax}], got {self.t0}."
            )

    def to_options(self):
        """Convert the configuration to Deformetrica model options."""
        return {
            "dense_mode": self._base._dense_mode,
            "dimension": self._base._dimension,
            "number_of_processes": self._base._number_of_processes,
            #
            "deformation_kernel_type": self._base.kernel_type,
            "deformation_kernel_width": self._base.kernel_width,
            "shoot_kernel_type": self._base._shoot_kernel_type,
            #
            "concentration_of_time_points": self._base.number_of_time_steps,
            #
            "freeze_template": self._base.freeze_template,
            "use_sobolev_gradient": self._base.use_sobolev_gradient,
            "sobolev_kernel_width_ratio": (self._base.sobolev_kernel_width_ratio),
            #
            "freeze_control_points": self._base.freeze_control_points,
            "initial_control_points": self._base.initial_control_points,
            "initial_cp_spacing": self._base._initial_cp_spacing,
            #
            "initial_momenta": self._base._initial_momenta,
            #
            "gpu_mode": self._base._get_gpu_mode(),
            #
            "t0": self.t0,
        }

    def _get_cache_exclusions(self):
        """Return parameters shared by regression cache identities."""
        return self._base._get_cache_exclusions()

    @property
    def kernel_type(self):
        """Kernel backend used for deformation computations."""
        return self._base.kernel_type

    def _get_kernel_device(self):
        return self._base._get_kernel_device()


class GeodesicRegressionModelConfig(BaseRegressionModelConfig):
    """Configuration for Deformetrica's geodesic regression model.

    The configuration reuses fixed-template LDDMM model settings while enforcing
    the constraints specific to geodesic regression.

    Geodesic regression does not support stationary velocity fields,
    volume-preserving deformations, or RK4 shooting.
    """

    def __init__(self):
        super().__init__()

        self.set_integration(
            shooting_method="euler",
            flow_method="euler",
        )

    def set_integration(
        self,
        number_of_time_steps=None,
        shooting_method=None,
        flow_method=None,
    ):
        """Set numerical integration parameters.

        Parameters
        ----------
        number_of_time_steps : int
            Number of time steps used to discretize the geodesic.
        shooting_method : {"euler", "rk2"}
            Numerical method used to evolve control points and momenta.
        flow_method : {"euler", "rk2"}
            Numerical method used to flow template points.

        Returns
        -------
        config : GeodesicRegressionModelConfig
            This configuration.
        """
        self._base.set_integration(
            number_of_time_steps=number_of_time_steps,
            shooting_method=shooting_method,
            flow_method=flow_method,
        )

        self._validate_integration()
        return self

    def _validate_integration(self):
        if self._base.shooting_method not in {"euler", "rk2"}:
            raise ValueError(
                f"Unknown shooting method {self._base.shooting_method!r}. "
                "Expected one of {'euler', 'rk2'}."
            )

        if self._base.flow_method not in {"euler", "rk2"}:
            raise ValueError(
                f"Unknown flow method {self._base.flow_method!r}. "
                "Expected one of {'euler', 'rk2'}."
            )

    def to_options(self):
        """Convert the configuration to Deformetrica model options."""
        options = super().to_options()
        options.update(
            {
                "use_rk2_for_shoot": self._base.shooting_method == "rk2",
                "use_rk2_for_flow": self._base.flow_method == "rk2",
            }
        )
        return options

    def update_from(self, config):
        """Update compatible settings from a fixed-template configuration.

        The deformation kernel width, execution settings, control-point settings,
        and integration settings are copied from ``config``.

        Stationary-velocity-field and volume-preservation settings are not copied,
        since geodesic regression does not support them. Regression-specific
        parameters such as ``t0`` are also left unchanged.

        Parameters
        ----------
        config : FixedTemplateDeterministicAtlasModelConfig
            Configuration providing the settings to copy.

        Returns
        -------
        config : GeodesicRegressionModelConfig
            This configuration.
        """
        if not isinstance(config, FixedTemplateDeterministicAtlasModelConfig):
            raise TypeError(
                "Expected a FixedTemplateDeterministicAtlasModelConfig, "
                f"got {type(config).__name__}."
            )

        self.set_deformation(
            kernel_width=config.kernel_width,
        )
        self.set_execution(
            kernel_type=config.kernel_type,
            device=config.device,
        )
        self.set_control_points(
            freeze=config.freeze_control_points,
            initial=config.initial_control_points,
        )
        self.set_integration(
            number_of_time_steps=config.number_of_time_steps,
            shooting_method=config.shooting_method,
            flow_method=config.flow_method,
        )

        return self


class SplineRegressionModelConfig(BaseRegressionModelConfig):
    """Configuration for Deformetrica's spline regression model.

    Spline regression uses fixed-template LDDMM settings and augments geodesic
    regression with external forces and geodesic regularization. Stationary
    velocity fields and volume-preserving deformations are not supported.
    """

    def __init__(self):
        super().__init__()

        self.set_integration(
            flow_method="euler",
        )
        self.set_regression(
            geodesic_weight=0.1,
        )

        self._freeze_external_forces = False

    def set_integration(
        self,
        number_of_time_steps=None,
        flow_method=None,
    ):
        """Set spline-regression integration parameters.

        Parameters
        ----------
        number_of_time_steps : int
            Number of time steps used to discretize the trajectory.
        flow_method : {"euler", "rk2"}
            Numerical method used to flow template points.

        Returns
        -------
        config : SplineRegressionModelConfig
            This configuration.
        """
        super().set_integration(
            number_of_time_steps=number_of_time_steps,
        )

        if flow_method is not None:
            self._base.flow_method = flow_method

        self._validate_integration()
        return self

    def set_regression(
        self,
        t0=None,
        geodesic_weight=None,
    ):
        """Set spline-regression parameters.

        Parameters
        ----------
        t0 : float
            Reference time of the regression trajectory.
        geodesic_weight : float
            Weight of the geodesic regularization term.

        Returns
        -------
        config : SplineRegressionModelConfig
            This configuration.
        """
        super().set_regression(t0=t0)

        if geodesic_weight is not None:
            self.geodesic_weight = geodesic_weight

        return self

    def _validate_integration(self):
        if self._base.flow_method not in {"euler", "rk2"}:
            raise ValueError(f"Unknown flow method {self._base.flow_method!r}.")

    def to_options(self):
        """Convert the configuration to Deformetrica model options."""
        options = super().to_options()
        options.update(
            {
                "use_rk2_for_flow": self._base.flow_method == "rk2",
                "freeze_external_forces": self._freeze_external_forces,
                "geodesic_weight": self.geodesic_weight,
            }
        )
        return options

    def _get_cache_exclusions(self):
        """Return parameters shared by regression cache identities."""
        return super()._get_cache_exclusions() | {"freeze_external_forces"}

    def update_from(self, config):
        """Update compatible settings from a fixed-template configuration.

        The deformation kernel width, execution settings, control-point settings,
        number of integration steps, and flow method are copied from ``config``.

        Stationary-velocity-field and volume-preservation settings are not copied,
        since spline regression does not support them. The shooting method is also
        not copied because it is not configurable for spline regression.
        Regression-specific parameters such as ``t0`` and ``geodesic_weight`` are
        left unchanged.

        Parameters
        ----------
        config : FixedTemplateDeterministicAtlasModelConfig
            Configuration providing the settings to copy.

        Returns
        -------
        config : SplineRegressionModelConfig
            This configuration.
        """
        if not isinstance(config, FixedTemplateDeterministicAtlasModelConfig):
            raise TypeError(
                "Expected a FixedTemplateDeterministicAtlasModelConfig, "
                f"got {type(config).__name__}."
            )

        self.set_deformation(
            kernel_width=config.kernel_width,
        )
        self.set_execution(
            kernel_type=config.kernel_type,
            device=config.device,
        )
        self.set_control_points(
            freeze=config.freeze_control_points,
            initial=config.initial_control_points,
        )
        self.set_integration(
            number_of_time_steps=config.number_of_time_steps,
            flow_method=config.flow_method,
        )

        return self
