import polpo.utils as _putils

HAS_DEFORMETRICA = _putils.has_package("deformetrica")

if HAS_DEFORMETRICA:
    # to allow post without deformetrica
    from ._protocol import LddmmToGlobal
