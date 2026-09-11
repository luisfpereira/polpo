import abc

from polpo.logging import logger
from polpo.preprocessing.base import PreprocessingStep


class MessageWithPrefixMixin(abc.ABC):
    def __init__(self, *args, msg_prefix="", **kwargs):
        self.msg_prefix = msg_prefix
        super().__init__(*args, **kwargs)


class BaseMeshInfoPrinter(PreprocessingStep, abc.ABC):
    @abc.abstractmethod
    def msg(self, mesh):
        pass

    def __call__(self, mesh):
        logger.info(self.msg(mesh))
        return mesh


class MeshSize(MessageWithPrefixMixin, BaseMeshInfoPrinter):
    def msg(self, mesh):
        return (
            f"{self.msg_prefix}"
            f"Number of vertices: {len(mesh.vertices)}."
            f" Number of faces: {len(mesh.faces)}"
        )
