"""Shared deformation and execution configuration mixins."""

from deformetrica.support.utilities import GpuMode


class DeformationConfigMixin:
    """Mixin providing deformation-kernel configuration."""

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
    """Mixin providing kernel backend and device configuration."""

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
