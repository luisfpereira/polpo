import hashlib
import json

from support.utilities import GpuMode


class DeformationConfigMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_deformation(
            kernel_width=20.0,
            use_svf=False,
            preserve_volume=False,
        )

    def set_deformation(
        self,
        kernel_width=None,
        use_svf=None,
        preserve_volume=None,
    ):
        """Set deformation-related parameters.

        Parameters
        ----------
        kernel_width : float
            Width of the deformation kernel controlling the spatial scale and
            smoothness of the deformation.
        use_svf : bool
            Whether to use a stationary velocity field instead of a
            time-dependent geodesic deformation.
        preserve_volume : bool
            Whether to use a volume-preserving deformation model.

        Returns
        -------
        config
            This configuration.
        """
        if kernel_width is not None:
            self.kernel_width = kernel_width

        if use_svf is not None:
            self.use_svf = use_svf

        if preserve_volume is not None:
            self.preserve_volume = preserve_volume

        self._validate_deformation()
        return self

    def _validate_deformation(self):
        pass


class ExecutionConfigMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_execution(
            kernel_type="torch",
            device="auto",
        )

    def set_execution(self, kernel_type=None, device=None):
        """Set execution-related parameters.

        Parameters
        ----------
        kernel_type : {"torch", "keops"}
            Backend used for kernel computations.
        device : {"auto", "cpu", "cuda"}
            Device used for computation. ``"auto"`` lets the execution
            backend select an available device.

        Returns
        -------
        config
            This configuration.
        """
        if kernel_type is not None:
            self.kernel_type = kernel_type

        if device is not None:
            self.device = device

        self._validate_execution()
        return self

    def _validate_execution(self):
        if self.kernel_type not in {"torch", "keops"}:
            raise ValueError(f"Unknown kernel type {self.kernel_type!r}.")

        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Unknown device {self.device!r}.")

    def _get_kernel_device(self):
        return {
            "auto": None,
            "cpu": "cpu",
            "cuda": "cuda",
        }[self.device]

    def _get_gpu_mode(self):
        return {
            "auto": GpuMode.AUTO,
            "cpu": GpuMode.NONE,
            "cuda": GpuMode.FULL,
        }[self.device]


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
            self._registration,
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

        self._registration.set_integration(
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
        self._registration.set_attachment(**kwargs)
        return self

    def set_optimization(self, **kwargs):
        self._registration.set_optimization(**kwargs)
        return self

    def set_execution(
        self,
        kernel_type=None,
        device=None,
    ):
        """Set execution parameters across LDDMM operations."""
        for config in (
            self._registration,
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

        if method == "fanning":
            transport.set_integration(
                time_steps_per_unit_time=self._shoot.time_steps_per_unit_time,
                post_transport_number_of_time_steps=self._registration.number_of_time_steps,
                transported_shooting_method=self._shoot.shooting_method,
                flow_method=self._shoot.flow_method,
            )

        elif method == "pole_ladder":
            transport.set_integration(
                time_steps_per_unit_time=self._shoot.time_steps_per_unit_time,
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


class BaseConfig:
    def cache_params(self):
        raise NotImplementedError

    def fingerprint(self):
        """Return a stable fingerprint of cache-relevant parameters."""
        payload = json.dumps(
            self.cache_params(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class RegistrationConfig(DeformationConfigMixin, ExecutionConfigMixin, BaseConfig):
    """Configuration for Deformetrica surface registration."""

    def __init__(self):
        super().__init__()

        self.set_attachment(
            metric="landmark",
            kernel_width=None,
            noise_std=1.0,
        )

        self.initial_control_points = None
        self.set_optimization(
            max_iter=200,
            freeze_control_points=False,
            print_every=20,
            tol=1e-5,
        )

        self.set_integration(
            number_of_time_steps=10,
            shooting_method="euler",
            flow_method="euler",
        )

        self.dimension = 3
        self.verbosity = "WARNING"

    def set_attachment(
        self,
        metric=None,
        kernel_width=None,
        noise_std=None,
    ):
        """Set attachment-related registration parameters.

        Parameters
        ----------
        metric : {"landmark", "current", "varifold"}
            Attachment discrepancy used to compare source and target surfaces.
            ``"landmark"`` uses pointwise correspondence, while ``"current"``
            and ``"varifold"`` provide kernel-based geometric matching.
            Varifold attachment is orientation-independent.
        kernel_width : float
            Spatial scale of the attachment kernel. Relevant for kernel-based
            attachment metrics such as varifold and current, and generally not
            used by landmark attachment.
        noise_std : float
            Standard deviation controlling the weight of the attachment term.
            Smaller values place more emphasis on matching the target, while
            larger values allow greater attachment error.

        Returns
        -------
        config : RegistrationConfig
            This configuration.
        """
        if metric is not None:
            self.metric = metric

            if metric == "landmark":
                self.attachment_kernel_width = None

        if kernel_width is not None:
            self.attachment_kernel_width = kernel_width

        if noise_std is not None:
            self.noise_std = noise_std

        self._validate_attachment()
        return self

    def set_optimization(
        self,
        max_iter=None,
        freeze_control_points=None,
        initial_control_points=None,
        print_every=None,
        tol=None,
    ):
        """Set optimization-related registration parameters.

        Parameters
        ----------
        max_iter : int
            Maximum number of optimization iterations.
        freeze_control_points : bool
            Whether to keep control points fixed during optimization.
        initial_control_points : path-like
            Path to initial control points.
        print_every : int
            Number of optimization iterations between progress reports.
        tol : float
            Optimization convergence tolerance.

        Returns
        -------
        config : RegistrationConfig
            This configuration.

        Notes
        -----
        Freezing the control points is generally preferable when tangent vectors
        from different registrations will be compared or parallel transported.
        Using a common set of control points gives the tangent vectors a shared
        finite-dimensional RKHS representation and avoids the reprojection needed
        when independently optimized control-point sets differ. Such reprojection
        need not preserve the RKHS norm.

        Allowing the control points to vary may improve individual registrations,
        but can therefore introduce an additional representation error in
        downstream geometric operations.
        """
        if max_iter is not None:
            self.max_iter = max_iter

        if freeze_control_points is not None:
            self.freeze_control_points = freeze_control_points

        if initial_control_points is not None:
            self.initial_control_points = initial_control_points

        if print_every is not None:
            self.print_every = print_every

        if tol is not None:
            self.tol = tol

        return self

    def set_integration(
        self,
        number_of_time_steps=None,
        shooting_method=None,
        flow_method=None,
    ):
        """Set integration-related registration parameters.

        Parameters
        ----------
        number_of_time_steps : int
            Number of time steps used to discretize the deformation.
        shooting_method : {"euler", "rk2", "rk4"}
            Numerical integration method used for shooting.
        flow_method : {"euler", "rk2"}
            Numerical integration method used to flow the shape.

        Returns
        -------
        config : RegistrationConfig
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

    def _validate_attachment(self):
        valid_metrics = {"landmark", "current", "varifold"}

        if self.metric not in valid_metrics:
            raise ValueError(
                f"Unknown attachment metric {self.metric!r}. "
                f"Expected one of {valid_metrics}."
            )

        if self.metric == "landmark":
            if self.attachment_kernel_width is not None:
                raise ValueError("Landmark attachment does not use a kernel width.")
            return

        if self.attachment_kernel_width is None:
            raise ValueError(
                f"Attachment metric {self.metric!r} requires a kernel width."
            )

        if self.attachment_kernel_width <= 0:
            raise ValueError("Attachment kernel width must be positive.")

    def _validate_integration(self):
        if self.shooting_method not in {"euler", "rk2", "rk4"}:
            raise ValueError(f"Unknown shooting method {self.shooting_method!r}.")

        if self.flow_method not in {"euler", "rk2"}:
            raise ValueError(f"Unknown flow method {self.flow_method!r}.")

    def estimator_options(self):
        return {
            "max_iterations": self.max_iter,
            "freeze_template": False,
            "freeze_control_points": self.freeze_control_points,
            "freeze_momenta": False,
            "use_sobolev_gradient": True,
            "sobolev_kernel_width_ratio": 1,
            "initial_control_points": self.initial_control_points,
            "initial_cp_spacing": None,
            "initial_momenta": None,
            "dense_mode": False,
            "number_of_threads": 1,
            "print_every_n_iters": self.print_every,
            "downsampling_factor": 1,
            "dimension": self.dimension,
            "optimization_method_type": "ScipyLBFGS",
            "convergence_tolerance": self.tol,
        }

    def model_options(self, output_dir):
        return {
            "deformation_kernel_type": self.kernel_type,
            "deformation_kernel_width": self.kernel_width,
            "deformation_kernel_device": self._get_kernel_device(),
            "use_svf": self.use_svf,
            "preserve_volume": self.preserve_volume,
            "number_of_time_points": self.number_of_time_steps + 1,
            "use_rk2_for_shoot": self.shooting_method == "rk2",
            "use_rk4_for_shoot": self.shooting_method == "rk4",
            "use_rk2_for_flow": self.flow_method == "rk2",
            "freeze_template": False,
            "freeze_control_points": self.freeze_control_points,
            "initial_control_points": self.initial_control_points,
            "dimension": self.dimension,
            "output_dir": output_dir,
        }

    def template_specifications(self, source):
        return {
            "shape": {
                "deformable_object_type": "SurfaceMesh",
                "kernel_type": self.kernel_type,
                "kernel_width": self.attachment_kernel_width,
                "kernel_device": self._get_kernel_device(),
                "noise_std": self.noise_std,
                "filename": source,
                "noise_variance_prior_scale_std": None,
                "noise_variance_prior_normalized_dof": 0.01,
                "attachment_type": self.metric,
            }
        }

    def cache_params(self):
        """Return parameters that determine the registration result."""
        return {
            "kernel_width": float(self.kernel_width),
            "kernel_type": self.kernel_type,
            "use_svf": self.use_svf,
            "preserve_volume": self.preserve_volume,
            "metric": self.metric,
            "attachment_kernel_width": None
            if self.attachment_kernel_width is None
            else float(self.attachment_kernel_width),
            "noise_std": self.noise_std,
            "freeze_control_points": self.freeze_control_points,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "number_of_time_steps": self.number_of_time_steps,
            "shooting_method": self.shooting_method,
            "flow_method": self.flow_method,
            "dimension": self.dimension,
        }


class ShootConfig(DeformationConfigMixin, ExecutionConfigMixin, BaseConfig):
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

    def to_kwargs(self):
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

    def cache_params(self):
        """Return parameters that determine the shooting result."""
        return {
            "kernel_width": float(self.kernel_width),
            "kernel_type": self.kernel_type,
            "use_svf": self.use_svf,
            "preserve_volume": self.preserve_volume,
            "time_steps_per_unit_time": self.time_steps_per_unit_time,
            "shooting_method": self.shooting_method,
            "flow_method": self.flow_method,
        }


class FanningConfig(DeformationConfigMixin, ExecutionConfigMixin):
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

    def to_kwargs(self):
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

    def cache_params(self):
        """Return parameters that determine the fanning transport result."""
        return {
            "kernel_width": float(self.kernel_width),
            "kernel_type": self.kernel_type,
            "time_steps_per_unit_time": self.time_steps_per_unit_time,
            "post_transport_number_of_time_steps": (
                self.post_transport_number_of_time_steps
            ),
            "transported_shooting_method": self.transported_shooting_method,
            "flow_method": self.flow_method,
        }


class PoleLadderConfig(DeformationConfigMixin, ExecutionConfigMixin):
    """Configuration for Deformetrica pole-ladder transport."""

    def __init__(self):
        super().__init__()

        self.set_integration(
            time_steps_per_unit_time=10,
        )

    def set_integration(
        self,
        time_steps_per_unit_time=None,
    ):
        """Set numerical integration parameters for pole-ladder transport.

        Parameters
        ----------
        time_steps_per_unit_time : int
            Number of time steps per unit of time used to discretize the
            reference geodesic.

        Returns
        -------
        config : PoleLadderConfig
            This configuration.
        """
        if time_steps_per_unit_time is not None:
            self.time_steps_per_unit_time = time_steps_per_unit_time

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

    def to_kwargs(self):
        """Convert the configuration to Deformetrica keyword arguments."""
        return {
            "deformation_kernel_width": self.kernel_width,
            "deformation_kernel_type": self.kernel_type,
            "concentration_of_time_points": self.time_steps_per_unit_time,
            "number_of_time_points": self.time_steps_per_unit_time + 1,
            "gpu_mode": self._get_gpu_mode(),
        }

    def cache_params(self):
        """Return parameters that determine the pole-ladder transport result."""
        return {
            "kernel_width": float(self.kernel_width),
            "kernel_type": self.kernel_type,
            "time_steps_per_unit_time": self.time_steps_per_unit_time,
        }


class ParallelTransportConfig(BaseConfig):
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

    def to_kwargs(self):
        """Convert the active configuration to Deformetrica keyword arguments."""
        return self._config.to_kwargs()

    def cache_params(self):
        """Return parameters that determine the parallel transport result."""
        return {
            "method": self.method,
            **self._config.cache_params(),
        }


class DeterministicAtlasConfig(RegistrationConfig):
    """Configuration for Deformetrica deterministic atlas estimation."""

    def __init__(self):
        self.initial_step_size = None
        super().__init__()

        self.set_optimization(initial_step_size=1e-4)
        self.t0 = 0.0

    def set_optimization(
        self,
        max_iter=None,
        freeze_control_points=None,
        initial_control_points=None,
        print_every=None,
        tol=None,
        initial_step_size=None,
    ):
        """Set optimization-related atlas parameters.

        Parameters
        ----------
        max_iter : int
            Maximum number of optimization iterations.
        freeze_control_points : bool
            Whether to keep control points fixed during optimization.
        initial_control_points : path-like
            Path to initial control points.
        print_every : int
            Number of optimization iterations between progress reports.
        tol : float
            Optimization convergence tolerance.
        initial_step_size : float
            Initial step size used by the optimizer.

        Returns
        -------
        config : DeterministicAtlasConfig
            This configuration.
        """
        super().set_optimization(
            max_iter=max_iter,
            freeze_control_points=freeze_control_points,
            initial_control_points=initial_control_points,
            print_every=print_every,
            tol=tol,
        )

        if initial_step_size is not None:
            self.initial_step_size = initial_step_size

        return self

    def model_options(self, output_dir):
        """Convert the configuration to Deformetrica model options."""
        options = super().model_options(output_dir)

        options.update(
            {
                "concentration_of_time_points": self.number_of_time_steps,
                "freeze_momenta": False,
                "freeze_noise_variance": False,
                "use_sobolev_gradient": True,
                "sobolev_kernel_width_ratio": 1,
                "initial_cp_spacing": None,
                "initial_momenta": None,
                "dense_mode": False,
                "number_of_processes": 1,
                "random_seed": None,
                "t0": self.t0,
                "tmin": self.t0,
                "tmax": 1.0,
            }
        )

        return options

    def estimator_options(self):
        """Convert the configuration to Deformetrica estimator options."""
        options = super().estimator_options()

        options.update(
            {
                "max_line_search_iterations": 50,
                "initial_step_size": self.initial_step_size,
            }
        )

        return options

    def cache_params(self):
        """Return parameters that determine the atlas estimation result."""
        return {
            **super().cache_params(),
            "initial_step_size": self.initial_step_size,
            "t0": self.t0,
        }

    @classmethod
    def from_registration_config(cls, registration_config):
        config = cls()

        config.set_deformation(
            kernel_width=registration_config.kernel_width,
            use_svf=registration_config.use_svf,
            preserve_volume=registration_config.preserve_volume,
        )
        config.set_execution(
            kernel_type=registration_config.kernel_type,
            device=registration_config.device,
        )
        config.set_attachment(
            metric=registration_config.metric,
            kernel_width=registration_config.attachment_kernel_width,
            noise_std=registration_config.noise_std,
        )
        config.set_optimization(
            max_iter=registration_config.max_iter,
            freeze_control_points=registration_config.freeze_control_points,
            initial_control_points=registration_config.initial_control_points,
            print_every=registration_config.print_every,
            tol=registration_config.tol,
        )
        config.set_integration(
            number_of_time_steps=registration_config.number_of_time_steps,
            shooting_method=registration_config.shooting_method,
            flow_method=registration_config.flow_method,
        )

        config.dimension = registration_config.dimension
        config.verbosity = registration_config.verbosity

        return config
