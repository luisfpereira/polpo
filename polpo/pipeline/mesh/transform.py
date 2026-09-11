"""Mesh transformations."""

import numpy as np

from polpo.preprocessing.base import PreprocessingStep

from ._register import VERTICES_ATTR


def _apply_to_attr(func, mesh, attr=None):
    # NB: done in place
    if attr is None:
        attr = VERTICES_ATTR.get(type(mesh), "vertices")
    values = func(getattr(mesh, attr))
    setattr(mesh, attr, values)

    return mesh


class MeshTransformer(PreprocessingStep):
    """Applies function to mesh attribute.

    NB: operations are done in place.

    Parameters
    ----------
    func : callable
        Function to apply to attribute.
    attr : str
        Mesh attribute containing vertex information.
        If None, it uses registered or "vertices".
    """

    def __init__(self, func, attr=None):
        super().__init__()
        self.func = func
        self.attr = attr

    def _build_func(self):
        return self.func

    def __call__(self, mesh):
        """Apply step.

        Parameters
        ----------
        mesh : Mesh
            Mesh to transform.

        Returns
        -------
        transformed_mesh : Mesh
            Transformed mesh.
        """
        if isinstance(mesh, (list, tuple)):
            mesh, *transform_args = mesh
            func = self._build_func(*transform_args)
        else:
            func = self.func

        return _apply_to_attr(func, mesh, self.attr)


class MeshCenterer(MeshTransformer):
    """Center mesh by removing vertex mean.

    Parameters
    ----------
    attr : str
        Mesh attribute containing vertex information.
        If None, it uses registered or "vertices".
    """

    def __init__(self, attr=None):
        super().__init__(
            attr=attr,
            func=lambda vertices: vertices - np.mean(vertices, axis=0),
        )


class MeshScaler(MeshTransformer):
    """Scale mesh.

    Parameters
    ----------
    scaling_factor : float
        Scaling factor.
    attr : str
        Mesh attribute containing vertex information.
        If None, it uses registered or "vertices".
    """

    def __init__(self, scaling_factor=20.0, attr=None):
        super().__init__(
            attr=attr,
            func=self._build_function(scaling_factor),
        )

    def _build_function(self, scaling_factor):
        return lambda vertices: vertices / scaling_factor


class AffineTransformation(MeshTransformer):
    """Apply affine transform to mesh vertices."""

    def __init__(self, transform=None, attr=None):
        super().__init__(
            attr=attr,
            func=self._build_function(transform) if transform is not None else None,
        )

    def _build_function(self, transform):
        rotation_matrix = transform[:3, :3]
        translation = transform[:3, 3]

        return lambda vertices: (rotation_matrix @ vertices.T).T + translation
