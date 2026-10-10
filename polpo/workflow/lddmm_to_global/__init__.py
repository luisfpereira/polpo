from polpo.utils import has_package

from .output import LddmmToGlobalMultiOutput, LddmmToGlobalOutput

if has_package("deformetrica"):
    # to allow post without deformetrica
    from ._protocol import LddmmToGlobal
