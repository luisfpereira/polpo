from polpo.utils import has_package

from .representations import (
    ControlPoints,
    Flow,
    Momenta,
    Point,
    TangentVector,
    Velocity,
)

HAS_DEFORMETRICA = has_package("deformetrica")


if HAS_DEFORMETRICA:
    # allows using repr without deformetrica
    from .geometry import LddmmMetric
    from .learning import FrechetMean, GeodesicRegression, SplineRegression
