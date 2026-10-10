"""Deformetrica-backed LDDMM tools for surface shapes."""

from polpo.utils import has_package

from . import io

HAS_DEFORMETRICA = has_package("deformetrica")

if HAS_DEFORMETRICA:
    # allows using repr without deformetrica
    from . import config, geometry, learning, registration, utils
