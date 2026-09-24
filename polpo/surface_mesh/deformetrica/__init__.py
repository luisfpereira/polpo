import polpo.utils as _putils

from .representations import (
    ControlPoints,
    Flow,
    Momenta,
    Point,
    TangentVector,
    Velocity,
)

HAS_DEFORMETRICA = _putils.has_package("deformetrica")


if HAS_DEFORMETRICA:
    # allows using repr without deformetrica
    from .geometry import LddmmMetric
    from .learning import FrechetMean, GeodesicRegression, SplineRegression
