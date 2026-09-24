"""Base configuration classes and cache fingerprinting utilities."""

import hashlib
import json

import numpy as np


class BaseConfig:
    """Base configuration."""

    def build_cache_params(self):
        """Return cache-relevant params for fingerprint computation."""
        raise NotImplementedError

    def compute_fingerprint(self):
        """Return a stable fingerprint of cache-relevant parameters."""
        payload = json.dumps(
            self.build_cache_params(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class OptionsConfig(BaseConfig):
    """Base configuration represented by Deformetrica options."""

    def to_options(self):
        """Convert the configuration to Deformetrica options."""
        raise NotImplementedError

    def _get_cache_exclusions(self):
        """Return model options excluded from cache identity."""
        return set()

    def build_cache_params(self):
        """Return cache-relevant Deformetrica options."""
        params = self.to_options()

        return {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in params.items()
            if key not in self._get_cache_exclusions()
        }
