import polpo.utils as _putils

HAS_DEFORMETRICA = _putils.has_package("deformetrica")

if HAS_DEFORMETRICA:
    # allows using repr without deformetrica
    from . import config, geometry, learning, registration, utils


from . import io
