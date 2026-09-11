try:
    from polpo.preprocessing._pyvista import (  # noqa:F401
        PvExtractPoints,
        PvSelectSubset,
        PvSubsetSplitter,
    )
except ImportError:
    pass
