try:
    from polpo.preprocessing._trimesh import TrimeshDecimator  # noqa:F401
except ImportError:
    pass

try:
    from polpo.preprocessing._fast_simplification import (
        FastSimplificationDecimator,  # noqa:F401
    )
except ImportError:
    pass

try:
    from polpo.preprocessing._pyvista import PvDecimate  # noqa:F401
except ImportError:
    pass

try:
    from polpo.preprocessing._h2_surfacematch import H2MeshDecimator  # noqa:F401
except ImportError:
    pass
