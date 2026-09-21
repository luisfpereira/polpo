"""A filesystem-backed adapter around deformetrica."""

from abc import ABC
from pathlib import Path

import numpy as np

import polpo.ext.deformetrica.io as pdefoio
from polpo.auto_all import auto_all
from polpo.io.json import load_json, save_json
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
    def from_dir(cls, id_, dirname, transported=False):
        """Create a filesystem-backed tangent vector.

        Parameters
        ----------
        id_ : str
            Tangent-vector identifier.
        dirname : path-like
            Deformetrica output directory.
        transported : bool
            Whether the directory contains a transported tangent vector.

        Returns
        -------
        tangent_vec : StoredTangentVector
            Filesystem-backed tangent vector.
        """
        if transported:
            return StoredTransportedVector(id_, dirname)
        return StoredTangentVector(id_, dirname)

    @classmethod
    def from_dict(cls, data, root_dir=None, transported=False):
        """Create a filesystem-backed tangent vector from serialized data."""
        if transported:
            return StoredTransportedVector.from_dict(data, root_dir=root_dir)

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
            path = pdefoio.find_atlas_momenta(self.dirname, id_)
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


class StoredTransportedVector(StoredTangentVector):
    """Filesystem-backed tangent vector produced by parallel transport."""

    @property
    def control_points(self):
        """Final transported control points."""
        return ControlPoints.from_file(
            pdefoio.find_transported_control_points(self.dirname)
        )

    @property
    def momenta(self):
        """Final transported momenta."""
        return Momenta.from_file(pdefoio.find_transported_momenta(self.dirname))


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


class _Result(ABC):
    """Base class for filesystem-backed Deformetrica results."""

    @property
    def params_path(self):
        """Path to the serialized result metadata."""
        return self.dirname / "params.json"

    def write(self, cache=None):
        """Write result metadata to disk.

        Parameters
        ----------
        cache : dict
            Cache metadata to include in the result parameters.
        """
        params = self.params

        if cache is not None:
            params["cache"] = cache

        return save_json(self.params_path, params)

    def read_params(self):
        """Read serialized result parameters.

        Returns
        -------
        params : dict
            Stored parameters, or an empty dictionary when absent.
        """
        if not self.params_path.exists():
            return {}

        return load_json(self.params_path)

    def read_cache(self):
        """Read cache metadata from the stored parameters."""
        return self.read_params().get("cache", {})


class RegistrationResult(_Result):
    """Filesystem-backed result of an LDDMM registration.

    Parameters
    ----------
    id_ : str
        Registration identifier.
    dir_config : LddmmPaths
        Output-directory configuration.
    base_point : Point
        Source point of the registration.
    point : Point
        Target point of the registration.
    """

    def __init__(self, id_, dir_config, base_point, point):
        self.id = id_
        self.dir_config = dir_config

        self.base_point = base_point
        self.point = point

    @property
    def dirname(self):
        """Directory containing the result files."""
        return self.dir_config.registration(self.id)

    @classmethod
    def load(cls, id_, dir_config):
        """Load a result from its serialized metadata.

        Parameters
        ----------
        id_ : str
            Result identifier.
        dir_config : LddmmPaths
            Output-directory configuration.

        Returns
        -------
        result : RegistrationResult
            Reconstructed result object.
        """
        data = load_json(dir_config.registration(id_) / "params.json")

        point = Point.from_dict(
            data["point"],
            root_dir=dir_config.root,
        )
        base_point = Point.from_dict(
            data["base_point"],
            root_dir=dir_config.root,
        )

        return cls(id_, dir_config, base_point, point)

    @property
    def params(self):
        """Serializable parameters describing the result."""
        root_dir = self.dir_config.root

        return dict(
            base_point=self.base_point.to_dict(root_dir=root_dir),
            point=self.point.to_dict(root_dir=root_dir),
        )

    @property
    def tangent_vec(self):
        """Initial tangent vector estimated by registration."""
        return TangentVector.from_dir(self.id, self.dirname)

    @property
    def reconstructed(self):
        """Target reconstruction obtained by shooting the estimated vector."""
        path = pdefoio.find_atlas_reconstruction(
            self.dirname,
            self.point.id,
        )

        return Point(
            id_=f"{self.base_point.id}_shoot_{self.dirname.name}",
            vtk_path=path,
        )

    @property
    def flow(self):
        """Discrete geodesic flow from the source toward the target."""
        paths = pdefoio.find_atlas_flow(
            self.dirname,
            self.point.id,
        )

        return Flow(
            [
                Point(
                    f"{self.dirname.name}|tp{index}",
                    vtk_path=path,
                )
                for index, path in enumerate(paths)
            ]
        )


class ShootResult(_Result):
    """Filesystem-backed result of geodesic shooting.

    Parameters
    ----------
    id_ : str
        Shooting identifier.
    dir_config : LddmmPaths
        Output-directory configuration.
    tangent_vec : TangentVector
        Initial tangent vector.
    base_point : Point
        Initial point of the geodesic.
    """

    def __init__(self, id_, dir_config, tangent_vec, base_point):
        self.id = id_
        self.dir_config = dir_config

        self.tangent_vec = tangent_vec
        self.base_point = base_point

    @property
    def dirname(self):
        """Directory containing the result files."""
        return self.dir_config.shoot(self.id)

    @classmethod
    def load(cls, id_, dir_config):
        """Load a result from its serialized metadata.

        Parameters
        ----------
        id_ : str
            Result identifier.
        dir_config : LddmmPaths
            Output-directory configuration.

        Returns
        -------
        result : ShootResult
            Reconstructed result object.
        """
        data = load_json(dir_config.shoot(id_) / "params.json")

        tangent_data = data["tangent_vec"]
        transported = Path(tangent_data["dirname"]).is_relative_to(
            dir_config.relative(dir_config.transports)
        )
        tangent_vec = TangentVector.from_dict(
            tangent_data, root_dir=dir_config.root, transported=transported
        )

        base_point = Point.from_dict(data["base_point"], root_dir=dir_config.root)

        return cls(id_, dir_config, tangent_vec, base_point)

    @property
    def params(self):
        """Serializable parameters describing the result."""
        root_dir = self.dir_config.root

        return dict(
            tangent_vec=self.tangent_vec.to_dict(root_dir=root_dir),
            base_point=self.base_point.to_dict(root_dir=root_dir),
        )

    @property
    def point(self):
        """End point of the shooting flow."""
        path = pdefoio.find_shooting_flow(self.dirname)[-1]
        return Point(self.dirname.name, vtk_path=path)

    @property
    def flow(self):
        """Discrete geodesic flow produced by shooting."""
        paths = pdefoio.find_shooting_flow(self.dirname)

        return Flow(
            [
                Point(f"{self.dirname.name}|tp{index}", vtk_path=path)
                for index, path in enumerate(paths)
            ]
        )


class _BaseDeterministicAtlasResult(_Result):
    """Base class for deterministic-atlas results.

    Parameters
    ----------
    id_ : str
        Atlas identifier.
    dir_config : LddmmPaths
        Output-directory configuration.
    points : sequence of Point
        Observations used to estimate the atlas.
    """

    def __init__(self, id_, dir_config, points):
        self.id = id_
        self.dir_config = dir_config
        self.points = points

    @property
    def dirname(self):
        """Directory containing the result files."""
        return self.dir_config.atlas(self.id)

    @classmethod
    def load(cls, id_, dir_config):
        """Load a result from its serialized metadata.

        Parameters
        ----------
        id_ : str
            Result identifier.
        dir_config : LddmmPaths
            Output-directory configuration.

        Returns
        -------
        result : AtlasResult
            Reconstructed result object.
        """
        data = load_json(dir_config.atlas(id_) / "params.json")

        points = [
            Point.from_dict(data_, root_dir=dir_config.root) for data_ in data["points"]
        ]
        return cls(id_, dir_config, points)

    @property
    def params(self):
        """Serializable parameters describing the result."""
        return dict(
            points=[pt.to_dict(root_dir=self.dir_config.root) for pt in self.points]
        )


class DeterministicAtlasManyResult(_BaseDeterministicAtlasResult):
    """Result of deterministic-atlas estimation from multiple points."""

    # TODO: add to_registrations

    @property
    def template(self):
        """Estimated atlas template."""
        return Point(
            self.id,
            vtk_path=pdefoio.find_template(self.dirname),
        )

    @property
    def control_points(self):
        """Control points shared by the atlas tangent vectors."""
        return ControlPoints.from_file(
            pdefoio.find_control_points(self.dirname),
        )

    @property
    def tangent_vecs(self):
        """Tangent vectors from the atlas template to each observation."""
        return [
            TangentVector.from_dir(f"{self.id}_to_{pt.id}", self.dirname)
            for pt in self.points
        ]

    @property
    def flows(self):
        """Discrete atlas-to-observation flows indexed by point identifier."""
        return {
            point.id: Flow(
                [
                    Point(
                        f"{self.id}_to_{point.id}|tp{index}",
                        vtk_path=path,
                    )
                    for index, path in enumerate(
                        pdefoio.find_atlas_flow(self.dirname, point.id)
                    )
                ]
            )
            for point in self.points
        }

    @property
    def reconstructed(self):
        """Reconstructed observations obtained from the atlas model."""
        return [
            Point(
                id_=f"{self.id}_shoot_{self.id}_to_{point.id}",
                vtk_path=pdefoio.find_atlas_reconstruction(
                    self.dirname,
                    point.id,
                ),
            )
            for point in self.points
        ]


class DeterministicAtlasOneDir(_BaseDeterministicAtlasResult):
    """Degenerate deterministic-atlas result for a single observation."""

    @property
    def template(self):
        """Atlas template, equal to the single input observation."""
        return Point(
            self.id,
            vtk_path=self.dirname / f"{self.id}.vtk",
        )

    @property
    def reconstructed(self):
        """Single reconstructed observation."""
        return [self.template]

    def write_mesh(self):
        """Write the single input point as the atlas template."""
        self.dirname.mkdir(parents=True, exist_ok=True)

        path = self.dirname / f"{self.id}.vtk"
        write_vtk_polydata(path, self.points[0].as_surface())


class DeterministicAtlasResult(_BaseDeterministicAtlasResult):
    """Factory for deterministic-atlas result representations."""

    def __new__(cls, id_, dir_config, points):
        if len(points) == 1:
            return DeterministicAtlasOneDir(id_, dir_config, points)

        return DeterministicAtlasManyResult(id_, dir_config, points)


class _TransportResult(_Result):
    """Base result of LDDMM parallel transport.

    Parameters
    ----------
    id_ : str
        Transport identifier.
    dir_config : LddmmPaths
        Output-directory configuration.
    tangent_vec : TangentVector
        Tangent vector being transported.
    base_point : Point
        Base point of the reference geodesic.
    direction : TangentVector
        Initial direction of the reference geodesic.
    """

    def __init__(self, id_, dir_config, tangent_vec, base_point, direction):
        self.id = id_
        self.dir_config = dir_config

        self.tangent_vec = tangent_vec
        self.base_point = base_point
        self.direction = direction

    @property
    def dirname(self):
        """Directory containing the result files."""
        return self.dir_config.transport(self.id)

    @property
    def params(self):
        """Serializable parameters describing the result."""
        root_dir = self.dir_config.root

        return dict(
            tangent_vec=self.tangent_vec.to_dict(root_dir=root_dir),
            base_point=self.base_point.to_dict(root_dir=root_dir),
            direction=self.direction.to_dict(root_dir=root_dir),
            method=self.method,
        )

    @property
    def transported(self):
        """Transported tangent vector."""
        return TangentVector.from_dir(
            self.dirname.name,
            self.dirname,
            transported=True,
        )


class TransportResultPoleLadder(_TransportResult):
    """Parallel-transport result computed with pole ladder."""

    method = "pole_ladder"


class TransportResultFan(_TransportResult):
    """Parallel-transport result computed with fanning."""

    method = "fanning"

    @property
    def reconstructed(self):
        """End point of the reference geodesic reconstructed by fanning."""
        vtk_path = pdefoio.find_shooting_flow(self.dirname)[-1]

        return Point(
            id_=f"{self.direction.id}_r",
            vtk_path=vtk_path,
        )

    @property
    def reconstructed_shooted(self):
        """End point obtained by shooting the transported tangent vector."""
        vtk_path = pdefoio.find_parallel_shooting_flow(self.dirname)[-1]

        return Point(
            id_=f"{self.direction.id}_rs",
            vtk_path=vtk_path,
        )

    @property
    def reconstructed_flow(self):
        """Discrete reconstructed reference geodesic."""
        paths = pdefoio.find_shooting_flow(self.dirname)

        return Flow(
            [
                Point(f"{self.direction.id}_r|tp{index}", vtk_path=path)
                for index, path in enumerate(paths)
            ]
        )

    @property
    def reconstructed_shooted_flow(self):
        """Discrete flow obtained by shooting the transported vector."""
        paths = pdefoio.find_parallel_shooting_flow(self.dirname)

        return Flow(
            [
                Point(f"{self.direction.id}_rs|tp{index}", vtk_path=path)
                for index, path in enumerate(paths)
            ]
        )


class TransportResultZero(_TransportResult):
    """No-op parallel-transport result for a numerically zero vector."""

    method = "zero"

    def write_data(self):
        """Write zero-transport output in Deformetrica-compatible form."""
        path = self.dirname

        path.mkdir(exist_ok=True, parents=True)

        vec = self.tangent_vec
        pdefoio.write_array(
            path / "final_cp.txt", vec.control_points.as_array(), header=False
        )
        pdefoio.write_array(path / "transported_momenta.txt", vec.momenta.as_array())


class TransportResult:
    """Factory for parallel-transport result representations."""

    def __new__(cls, *args, method="pole_ladder", **kwargs):
        result_cls = {
            "pole_ladder": TransportResultPoleLadder,
            "fanning": TransportResultFan,
            "zero": TransportResultZero,
        }[method]

        return result_cls(*args, **kwargs)

    @classmethod
    def load(cls, id_, dir_config):
        """Load a result from its serialized metadata.

        Parameters
        ----------
        id_ : str
            Result identifier.
        dir_config : LddmmPaths
            Output-directory configuration.

        Returns
        -------
        result : TransportResult
            Reconstructed result object.
        """
        data = load_json(dir_config.transport(id_) / "params.json")

        tangent_vec = TangentVector.from_dict(
            data["tangent_vec"],
            root_dir=dir_config.root,
        )
        base_point = Point.from_dict(
            data["base_point"],
            root_dir=dir_config.root,
        )
        direction = TangentVector.from_dict(
            data["direction"],
            root_dir=dir_config.root,
        )

        return cls(
            id_,
            dir_config,
            tangent_vec,
            base_point,
            direction,
            method=data["method"],
        )


__all__ = auto_all(globals())
