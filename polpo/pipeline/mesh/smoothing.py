try:
    from polpo.preprocessing._trimesh import TrimeshLaplacianSmoothing  # noqa:F401
except ImportError:
    pass

try:
    from polpo.preprocessing._pyvista import PvSmoothTaubin  # noqa:F401
except ImportError:
    pass
