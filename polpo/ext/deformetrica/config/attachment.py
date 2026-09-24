"""Configuration of surface attachment terms for Deformetrica."""

from .base import BaseConfig


class SurfaceAttachmentConfig(BaseConfig):
    """Configuration for surface attachment terms.

    This class controls how a source surface is compared with a target surface
    and builds the corresponding Deformetrica template specification.
    """

    def __init__(self):
        super().__init__()
        self.set_attachment(
            metric="landmark",
            kernel_width=None,
            noise_std=1.0,
        )

    def set_attachment(
        self,
        metric=None,
        kernel_width=None,
        noise_std=None,
    ):
        """Set surface attachment parameters.

        Parameters
        ----------
        metric : {"landmark", "current", "varifold"}
            Attachment discrepancy used to compare surfaces.
        kernel_width : float
            Spatial scale of the attachment kernel. Required for current and
            varifold attachment and unused for landmark attachment.
        noise_std : float
            Standard deviation controlling the attachment weight.

        Returns
        -------
        config : SurfaceAttachmentConfig
            This configuration.
        """
        if metric is not None:
            self.metric = metric

            if metric == "landmark":
                self.kernel_width = None

        if kernel_width is not None:
            self.kernel_width = kernel_width

        if noise_std is not None:
            self.noise_std = noise_std

        self._validate_attachment()
        return self

    def _validate_attachment(self):
        valid_metrics = {"landmark", "current", "varifold"}

        if self.metric not in valid_metrics:
            raise ValueError(
                f"Unknown attachment metric {self.metric!r}. "
                f"Expected one of {valid_metrics}."
            )

        if self.metric == "landmark":
            if self.kernel_width is not None:
                raise ValueError("Landmark attachment does not use a kernel width.")
            return

        if self.kernel_width is None:
            raise ValueError(
                f"Attachment metric {self.metric!r} requires a kernel width."
            )

    def build_template_specifications(
        self,
        source,
        kernel_type,
        kernel_device=None,
    ):
        """Build Deformetrica template specifications.

        Parameters
        ----------
        source : path-like
            Path to the source surface.
        kernel_type : str
            Backend used by kernel-based attachment metrics.
        kernel_device : str
            Device used by the attachment kernel.

        Returns
        -------
        specifications : dict
            Deformetrica template specifications.
        """
        return {
            "shape": {
                "deformable_object_type": "SurfaceMesh",
                "kernel_type": kernel_type,
                "kernel_width": self.kernel_width,
                "kernel_device": kernel_device,
                "noise_std": self.noise_std,
                "filename": source,
                "noise_variance_prior_scale_std": None,
                "noise_variance_prior_normalized_dof": 0.01,
                "attachment_type": self.metric,
            }
        }

    def build_cache_params(self):
        """Return parameters that determine the attachment term."""
        return {
            "metric": self.metric,
            "kernel_width": (
                None if self.kernel_width is None else float(self.kernel_width)
            ),
            "noise_std": self.noise_std,
        }

    def update_from(self, config):
        """Update attachment settings from another attachment configuration.

        The attachment metric, kernel width, and noise standard deviation are
        copied from ``config``.

        Parameters
        ----------
        config : SurfaceAttachmentConfig
            Configuration providing the attachment settings to copy.

        Returns
        -------
        config : SurfaceAttachmentConfig
            This configuration.
        """
        if not isinstance(config, SurfaceAttachmentConfig):
            raise TypeError(
                "Expected a SurfaceAttachmentConfig, " f"got {type(config).__name__}."
            )

        return self.set_attachment(
            metric=config.metric,
            kernel_width=config.kernel_width,
            noise_std=config.noise_std,
        )
