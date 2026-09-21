try:
    from polpo.pipeline._trimesh import TrimeshDecimator  # noqa:F401
except ImportError:
    pass

try:
    from polpo.pipeline._fast_simplification import (
        FastSimplificationDecimator,  # noqa:F401
    )
except ImportError:
    pass

try:
    from polpo.pipeline._pyvista import PvDecimate  # noqa:F401
except ImportError:
    pass

try:
    from polpo.pipeline._h2_surfacematch import H2MeshDecimator  # noqa:F401
except ImportError:
    pass
