from pathlib import Path

import numpy as np

import polpo.ext.deformetrica.io as pdefoio
from polpo.surface_mesh.deformetrica.io import read_vtk_polydata, write_vtk_polydata


class Point:
    """Point associated with a surface mesh.

    A point may be initialized from a surface mesh, a VTK file path, or both.
    Missing representations are created lazily when requested. If both are
    provided, they are assumed to represent the same mesh.

    Parameters
    ----------
    id_ : str
        Point identifier.
    surface : surface-like
        Surface mesh associated with the point.
    vtk_path : path-like
        Path to the corresponding VTK surface mesh.
    dirname : path-like
        Directory used to construct ``vtk_path`` as ``<dirname>/<id_>.vtk``
        when ``vtk_path`` is not provided.

    Attributes
    ----------
    id : str
        Point identifier.
    surface : surface-like or None
        Surface mesh associated with the point. Loaded lazily from
        ``vtk_path`` when needed.
    vtk_path : pathlib.Path
        Path to the corresponding VTK surface mesh. The file is created
        lazily from ``surface`` when needed.
    """

    def __init__(self, id_, surface=None, vtk_path=None, dirname=None):
        self.id = id_
        self.surface = surface

        if vtk_path is None:
            if dirname is None:
                raise ValueError("Need to define ``vtk_path`` or ``dirname``.")

            vtk_path = dirname / f"{id_}.vtk"

        self.vtk_path = vtk_path

        if surface is None and not vtk_path.exists():
            raise ValueError("Need an attached surface or an existing VTK file.")

    def as_vtk_path(self):
        """Return the path to the VTK representation.

        The VTK file is created from the attached surface if it does not
        already exist.

        Returns
        -------
        vtk_path : pathlib.Path
            Path to the VTK surface mesh.
        """
        if not self.vtk_path.exists():
            self.vtk_path.parent.mkdir(parents=True, exist_ok=True)
            write_vtk_polydata(self.vtk_path, self.surface)

        return self.vtk_path

    def as_surface(self):
        """Return the surface representation.

        The surface is loaded from ``vtk_path`` if it is not already
        attached to the point.

        Returns
        -------
        surface : Surface
            Surface mesh associated with the point.
        """
        if self.surface is None:
            self.surface = read_vtk_polydata(self.vtk_path)

        return self.surface

    def to_dict(self, *, root_dir=None):
        """Serialize the point to a dictionary.

        Parameters
        ----------
        root_dir : path-like
            Root directory used to store ``vtk_path`` as a relative path.

        Returns
        -------
        data : dict
            Serialized point data.
        """
        vtk_path = self.vtk_path

        if root_dir is not None:
            vtk_path = vtk_path.relative_to(root_dir)

        return {
            "id": self.id,
            "vtk_path": vtk_path.as_posix(),
        }

    @classmethod
    def from_dict(cls, data, root_dir=None):
        """Create a point from serialized data.

        Parameters
        ----------
        data : dict
            Serialized point data.
        root_dir : path-like
            Root directory used to resolve a relative ``vtk_path``.

        Returns
        -------
        point : Point
            Deserialized point.
        """
        vtk_path = Path(data["vtk_path"])

        if root_dir is not None and not vtk_path.is_absolute():
            vtk_path = root_dir / vtk_path

        return cls(
            data["id"],
            vtk_path=vtk_path,
        )


class ControlPoints:
    """Control-point representation of an LDDMM vector field.

    Parameters
    ----------
    array : array-like, shape (n_control_points, dim)
        Control-point coordinates.
    """

    def __init__(self, array):
        self._array = array

    def as_array(self):
        """Return the control points as an array."""
        return self._array

    def as_polydata(self):
        """Return the control points as PyVista point data."""
        import pyvista as pv

        return pv.PolyData(self.as_array())

    @classmethod
    def from_file(cls, filename):
        """Create lazily loaded control points from a file.

        Parameters
        ----------
        filename : path-like
            Path to a Deformetrica control-points file.

        Returns
        -------
        control_points : StoredControlPoints
            Filesystem-backed control points.
        """
        return StoredControlPoints(filename)


class StoredControlPoints(ControlPoints):
    """Filesystem-backed LDDMM control points.

    The array is loaded from disk only when requested.

    Parameters
    ----------
    filename : path-like
        Path to the Deformetrica control-points file.
    """

    def __init__(self, filename):
        self.filename = filename

    def as_path(self):
        """Return the path to the stored control-points file."""
        return self.filename

    def as_array(self):
        """Load and return the control points as an array."""
        return pdefoio.read_array(self.filename)


class Momenta:
    """Momenta associated with LDDMM control points.

    Parameters
    ----------
    array : array-like, shape (n_control_points, dim)
        Control-point momenta.
    """

    def __init__(self, array):
        self._array = array

    def as_array(self):
        """Return the momenta as an array."""
        return self._array

    @classmethod
    def from_file(cls, filename):
        """Create lazily loaded momenta from a file."""
        return StoredMomenta(filename)


class StoredMomenta(Momenta):
    """Filesystem-backed LDDMM momenta.

    Parameters
    ----------
    filename : path-like
        Path to the Deformetrica momenta file.
    """

    def __init__(self, filename):
        self.filename = filename

    def as_path(self):
        """Return the path to the stored momenta file."""
        return self.filename

    def as_array(self):
        """Load and return the momenta as an array."""
        return pdefoio.read_array(self.filename)


class TangentVector:
    """LDDMM tangent vector represented by control points and momenta.

    Parameters
    ----------
    control_points : ControlPoints
        Locations carrying the momenta.
    momenta : Momenta
        Momenta defining the initial velocity field.
    """

    def __init__(self, control_points, momenta):
        self._control_points = control_points
        self._momenta = momenta

    @classmethod
    def from_dir(cls, id_, dirname):
        """Create a filesystem-backed tangent vector.

        Parameters
        ----------
        id_ : str
            Tangent-vector identifier.
        dirname : path-like
            Deformetrica output directory.

        Returns
        -------
        tangent_vec : StoredTangentVector
            Filesystem-backed tangent vector.
        """
        return StoredTangentVector(id_, dirname)

    @classmethod
    def from_dict(cls, data, root_dir=None):
        """Create a filesystem-backed tangent vector from serialized data."""
        return StoredTangentVector.from_dict(data, root_dir=root_dir)

    @property
    def control_points(self):
        """Control points carrying the tangent-vector momenta."""
        return self._control_points

    @property
    def momenta(self):
        """Momenta defining the tangent vector."""
        return self._momenta

    def as_polydata(self):
        """Return control points with momenta attached as point data."""
        polydata = self.control_points.as_polydata()
        polydata["momenta"] = self.momenta.as_array()
        return polydata

    def as_glyphs(self, factor=1.0):
        """Return arrow glyphs representing the tangent vector.

        Parameters
        ----------
        factor : float
            Glyph scale factor.

        Returns
        -------
        glyphs : pyvista.PolyData
            Arrow representation of the momenta.
        """
        return self.as_polydata().glyph(
            orient="momenta",
            scale="momenta",
            factor=factor,
        )


class StoredTangentVector(TangentVector):
    """Filesystem-backed LDDMM tangent vector.

    Parameters
    ----------
    id_ : str
        Tangent-vector identifier.
    dirname : path-like
        Directory containing its control points and momenta.
    """

    def __init__(self, id_, dirname):
        self.id = id_
        self.dirname = dirname

    @property
    def control_points(self):
        """Control points loaded from the associated Deformetrica output."""
        return ControlPoints.from_file(
            pdefoio.find_control_points(self.dirname),
        )

    @property
    def momenta(self):
        """Momenta loaded from the associated Deformetrica output."""
        id_ = self.id.split("_to_")[-1]

        try:
            path = pdefoio.find_subject_momenta(self.dirname, id_)
        except FileNotFoundError:
            path = pdefoio.find_momenta(self.dirname)

        return Momenta.from_file(path)

    def to_dict(self, root_dir=None):
        """Serialize the tangent-vector location to a dictionary."""
        dirname = self.dirname
        if root_dir is not None:
            dirname = dirname.relative_to(root_dir)

        return dict(id=self.id, dirname=dirname.as_posix())

    @classmethod
    def from_dict(cls, data, root_dir=None):
        """Create a stored tangent vector from serialized data."""
        dirname = Path(data["dirname"])

        if root_dir is not None and not dirname.is_absolute():
            dirname = Path(root_dir) / dirname

        return cls(id_=data["id"], dirname=dirname)


class Velocity:
    """Velocity field sampled at a finite set of locations.

    Parameters
    ----------
    locations : array-like, shape (n_points, dim)
        Locations at which the velocity is evaluated.
    values : array-like, shape (n_points, dim)
        Velocity vectors.
    """

    def __init__(self, locations, values):
        self.locations = locations
        self.values = values

    def as_polydata(self):
        """Return locations with velocity vectors attached as point data."""
        import pyvista as pv

        polydata = pv.PolyData(self.locations)
        polydata["velocity"] = self.values
        return polydata

    def as_glyphs(self, factor=1.0):
        """Return arrow glyphs representing the velocity field."""
        return self.as_polydata().glyph(
            orient="velocity",
            scale="velocity",
            factor=factor,
        )


class Flow:
    """Discrete sampling of a curve of surface points.

    Parameters
    ----------
    points : sequence of Point
        Sampled points along the flow.
    times : array-like, shape (n_points,)
        Sampling times. When omitted, points are assumed to be uniformly
        sampled over `[0, 1]`.
    """

    def __init__(self, points, times=None):
        if times is None:
            times = np.linspace(0.0, 1.0, len(points))

        self.points = points
        self.times = times

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]

    @property
    def initial_point(self):
        """First sampled point of the flow."""
        return self.points[0]

    @property
    def end_point(self):
        """Last sampled point of the flow."""
        return self.points[-1]

    def as_surfaces(self):
        """Return the surface representation of each sampled point."""
        return [point.as_surface() for point in self.points]

    def nearest(self, time):
        """Return the point sampled nearest to a given time.

        Parameters
        ----------
        time : float
            Requested time.

        Returns
        -------
        point : Point
            Sampled point whose time is closest to ``time``.
        """
        index = np.argmin(np.abs(self.times - time))
        return self.points[index]

    def at_sampled_time(self, time):
        """Return the point sampled at a given time.

        Parameters
        ----------
        time : float
            Requested sampling time.

        Returns
        -------
        point : Point
            Point sampled at ``time``.

        Raises
        ------
        ValueError
            If the flow has no sample at the requested time.
        """
        indices = np.flatnonzero(np.isclose(self.times, time))
        if len(indices) == 0:
            raise ValueError(f"No flow point sampled at time {time}")
        return self.points[indices[0]]
