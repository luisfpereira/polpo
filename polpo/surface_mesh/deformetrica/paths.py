"""Filesystem layout for persisted LDDMM computations."""

from pathlib import Path


class LddmmPaths:
    """Paths used by an LDDMM workflow.

    Parameters
    ----------
    root : path-like
        Root output directory.
    meshes : path-like
        Meshes directory.
    registrations : path-like
        Registrations directory.
    transports : path-like
        Transports directory.
    shoots : path-like
        Shoots directory.
    atlases : path-like
        Atlases directory.
    regressions : path-like
        Regressions directory.

    Attributes
    ----------
    root : path-like
        Root output directory.
    meshes : path-like
        Resolved meshes directory.
    registrations : path-like
        Resolved registrations directory.
    transports : path-like
        Resolved transports directory.
    shoots : path-like
        Resolved shoots directory.
    atlases : path-like
        Resolved atlases directory.
    regressions : path-like
        Resolved regressions directory.
    """

    def __init__(
        self,
        root,
        meshes=None,
        registrations=None,
        transports=None,
        shoots=None,
        atlases=None,
        regressions=None,
    ):
        self.root = root

        self.meshes = self._resolve(meshes or "meshes")
        self.registrations = self._resolve(registrations or "registrations")
        self.transports = self._resolve(transports or "transports")
        self.shoots = self._resolve(shoots or "shoots")
        self.atlases = self._resolve(atlases or "atlases")
        self.regressions = self._resolve(regressions or "regressions")

    def _resolve(self, path):
        path = Path(path)

        if path.is_absolute():
            return path

        return self.root / path

    def resolve(self, path):
        """Resolve a bundle-relative artifact path."""
        return self._resolve(path)

    def relative(self, path):
        """Convert an artifact path to a bundle-relative path."""
        return path.relative_to(self.root)

    def to_dict(self):
        """Serialize relative workflow paths.

        Returns
        -------
        data : dict
            Serialized workflow paths relative to ``root``.
        """
        return {
            "meshes": self.relative(self.meshes).as_posix(),
            "registrations": self.relative(self.registrations).as_posix(),
            "transports": self.relative(self.transports).as_posix(),
            "shoots": self.relative(self.shoots).as_posix(),
            "atlases": self.relative(self.atlases).as_posix(),
            "regressions": self.relative(self.regressions).as_posix(),
        }

    @classmethod
    def from_dict(cls, root, data):
        """Create workflow paths from serialized data.

        Parameters
        ----------
        root : path-like
            Root output directory.
        data : dict
            Serialized workflow paths.

        Returns
        -------
        paths : LddmmPaths
            Deserialized workflow paths.
        """
        return cls(
            root=root,
            **data,
        )

    def registration(self, id_):
        """Return the registration directory for a point.

        Parameters
        ----------
        id_ : str
            Point identifier.

        Returns
        -------
        path : path-like
            Registration directory associated with the identifier.
        """
        return self.registrations / id_

    def transport(self, id_):
        """Return the transport directory for a point.

        Parameters
        ----------
        id_ : str
            Point identifier.

        Returns
        -------
        path : path-like
            Transport directory associated with the point.
        """
        return self.transports / id_

    def shoot(self, id_):
        """Return the shoot directory for a point.

        Parameters
        ----------
        id_ : str
            Point identifier.

        Returns
        -------
        path : path-like
            Shoot directory associated with the point.
        """
        return self.shoots / id_

    def atlas(self, id_):
        """Return the atlas directory for a point.

        Parameters
        ----------
        id_ : str
            Point identifier.

        Returns
        -------
        path : path-like
            Atlas directory associated with the point.
        """
        return self.atlases / id_

    def regression(self, id_):
        """Return the regression directory for an identifier.

        Parameters
        ----------
        id_ : str
            Regression identifier.

        Returns
        -------
        path : path-like
            Regression directory associated with the identifier.
        """
        return self.regressions / id_
