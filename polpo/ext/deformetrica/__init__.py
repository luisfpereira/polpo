import polpo.utils as _putils

HAS_DEFORMETRICA = _putils.has_package("deformetrica")

if HAS_DEFORMETRICA:
    # allows using repr without deformetrica
    from . import geometry, learning, registration


from . import io
