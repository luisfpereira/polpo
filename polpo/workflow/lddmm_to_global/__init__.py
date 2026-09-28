import polpo.utils as _putils

from .output import LddmmToGlobalMultiOutput, LddmmToGlobalOutput

if _putils.has_package("deformetrica"):
    # to allow post without deformetrica
    from ._protocol import LddmmToGlobal
