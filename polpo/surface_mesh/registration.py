"""Rigid registration utilities for surface meshes."""

from polpo.registration.base import BaseRegistration
from polpo.registration.rigid import RigidRegistration
from polpo.surface_mesh.core import Surface


class BaseSurfaceRegistration(BaseRegistration):
    """Adapter for registration algorithms operating on surface representations.

    Parameters
    ----------
    registration : BaseRegistration
        Registration algorithm operating on the adapted surface data.
    """

    def __init__(self, registration):
        super().__init__()
        self.registration = registration

    def _to_data(self, surface):
        raise NotImplementedError

    def _from_data(self, data, reference):
        raise NotImplementedError

    def fit(self, source, target):
        """Estimate the rigid transformation from source to target.

        Parameters
        ----------
        source : Surface
            Source surface.
        target : Surface
            Target surface.

        Returns
        -------
        self : BaseSurfaceRegistration
            Fitted registration.
        """
        self.registration.fit(
            self._to_data(source),
            self._to_data(target),
        )
        return self

    def transform(self, surface):
        """Apply the estimated rigid transformation to a surface.

        Parameters
        ----------
        surface : Surface
            Surface to transform.

        Returns
        -------
        transformed : Surface
            Transformed surface with unchanged faces.
        """
        data = self.registration.transform(self._to_data(surface))
        return self._from_data(data, surface)

    def fit_transform(self, source, target):
        """Estimate and apply the transformation to the source.

        Parameters
        ----------
        source: Surface
            Source object.
        target: Surface
            Target object.

        Returns
        -------
        transformed: Surface
            Source transformed toward the target.
        """
        return self._from_data(
            self.registration.fit_transform(
                self._to_data(source), self._to_data(target)
            ),
            source,
        )


class CorrespondingSurfaceRigidRegistration(BaseSurfaceRegistration):
    """Rigid registration of surface meshes with vertex correspondences.

    Registration is estimated from corresponding mesh vertices and applied
    to the full surface while preserving its faces.

    Parameters
    ----------
    registration : BaseRegistration
        Registration algorithm operating on corresponding point sets.
    """

    @classmethod
    def from_default(cls):
        """Create a surface registration using the default rigid registration."""
        return cls(RigidRegistration())

    def _to_data(self, surface):
        return surface.vertices

    def _from_data(self, vertices, reference):
        return Surface(vertices, reference.faces)


class SurfaceIcpRegistration(BaseSurfaceRegistration):
    """Rigid ICP registration of surface meshes.

    This adapter converts surfaces to PyVista representations and delegates
    registration to a PyVista-compatible registration algorithm.

    Parameters
    ----------
    registration : BaseRegistration
        Registration algorithm operating on PyVista datasets.
    """

    @classmethod
    def from_default(cls, **kwargs):
        """Create an instance using the default ICP registration.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to `IcpRegistration`.

        Returns
        -------
        registration : SurfaceIcpRegistration
            Surface registration using the default ICP implementation.
        """
        from polpo.ext.pyvista.registration import IcpRegistration

        return cls(IcpRegistration(**kwargs))

    def _to_data(self, surface):
        return surface.to_polydata()

    def _from_data(self, polydata, surface=None):
        return Surface.from_polydata(polydata)


class SurfaceRigidRegistration:
    """Create a rigid registration for surface meshes.

    Parameters
    ----------
    known_correspondences : bool
        Whether source and target meshes have known vertex correspondences.
        If True, use correspondence-based rigid registration. Otherwise,
        use ICP registration.
    **kwargs
        Keyword arguments forwarded to the selected registration.

    Returns
    -------
    registration : BaseSurfaceRegistration
        Selected surface rigid registration.
    """

    def __new__(cls, known_correspondences=True, **kwargs):
        Registration = (
            CorrespondingSurfaceRigidRegistration
            if known_correspondences
            else SurfaceIcpRegistration
        )

        return Registration.from_default(**kwargs)
