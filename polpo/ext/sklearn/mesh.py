"""Transformer mixins handling mesh data."""

from sklearn.preprocessing import FunctionTransformer

from polpo.preprocessing import Map
from polpo.preprocessing.mesh.conversion import FromCombinatorialStructure, ToVertices


class BiMeshesToVertices(FunctionTransformer):
    """Function transformer for mesh to vertices conversion.

    During fit selects a particular mesh providing the combinatorial
    structure (i.e. the template).

    Index should not matter as meshes are expected to have
    same combinatorial structure.

    Parameters
    ----------
    index : int
        Mesh providing combinatorial structure.
    """

    def __init__(self, index=0):
        self.index = index

        super().__init__(
            func=Map(ToVertices()),
            inverse_func=Map(FromCombinatorialStructure()),
            check_inverse=False,
        )

    def fit(self, X, y=None):
        # y is ignored
        self.inverse_func.step.mesh = X[self.index]
        return self
