"""A filesystem-backed adapter around deformetrica."""

from abc import ABC

import numpy as np

import polpo.ext.deformetrica.io as pdefoio
from polpo.io.json import load_json, save_json
from polpo.surface_mesh.deformetrica.io import write_vtk_polydata

from .representations import ControlPoints, Flow, Momenta, Point, TangentVector


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
        """Load a regression result from its serialized metadata.

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
        path = pdefoio.find_subject_reconstruction(
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
        paths = pdefoio.find_subject_flow(
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
        tangent_vec = TangentVector.from_dict(
            tangent_data,
            root_dir=dir_config.root,
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
        path = pdefoio.find_geodesic_flow(self.dirname)[-1]
        return Point(self.dirname.name, vtk_path=path)

    @property
    def flow(self):
        """Discrete geodesic flow produced by shooting."""
        paths = pdefoio.find_geodesic_flow(self.dirname)

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
                        pdefoio.find_subject_flow(self.dirname, point.id)
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
                vtk_path=pdefoio.find_subject_reconstruction(
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
        vtk_path = pdefoio.find_geodesic_flow(self.dirname)[-1]

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
        paths = pdefoio.find_geodesic_flow(self.dirname)

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


class RegressionResult(_Result):
    """Filesystem-backed result of regression.

    Parameters
    ----------
    id_ : str
        Regression identifier.
    dir_config : LddmmPaths
        Output-directory configuration.
    base_point : Point
        Fixed base point of the regression trajectory.
    points : sequence of Point
        Observed points used to fit the regression.
    times : array-like, shape (n_points,)
        Observation times.
    """

    def __init__(self, id_, dir_config, base_point, points, times):
        self.id = id_
        self.dir_config = dir_config

        self.base_point = base_point
        self.points = points
        self.times = np.asarray(times)

    @property
    def dirname(self):
        """Directory containing the regression result."""
        return self.dir_config.regression(self.id)

    @classmethod
    def load(cls, id_, dir_config):
        """Load a geodesic-regression result from serialized metadata."""
        data = load_json(dir_config.regression(id_) / "params.json")

        base_point = Point.from_dict(
            data["base_point"],
            root_dir=dir_config.root,
        )
        points = [
            Point.from_dict(data_, root_dir=dir_config.root) for data_ in data["points"]
        ]

        return cls(
            id_,
            dir_config,
            base_point,
            points,
            data["times"],
        )

    @property
    def params(self):
        """Serializable parameters describing the result."""
        root_dir = self.dir_config.root

        return dict(
            base_point=self.base_point.to_dict(root_dir=root_dir),
            points=[point.to_dict(root_dir=root_dir) for point in self.points],
            times=self.times.tolist(),
        )

    @property
    def tangent_vec(self):
        """Initial tangent vector of the fitted geodesic."""
        return TangentVector(
            control_points=ControlPoints.from_file(
                pdefoio.find_control_points(self.dirname),
            ),
            momenta=Momenta.from_file(
                pdefoio.find_momenta(self.dirname),
            ),
        )

    @property
    def reconstructed(self):
        """Reconstructions of the observations along the fitted geodesic."""
        paths = pdefoio.find_reconstructions(self.dirname)

        return [
            Point(
                f"{point.id}_r",
                vtk_path=path,
            )
            for point, path in zip(self.points, paths)
        ]


class GeodesicRegressionResult(RegressionResult):
    """Filesystem-backed result of geodesic regression.

    Parameters
    ----------
    id_ : str
        Regression identifier.
    dir_config : LddmmPaths
        Output-directory configuration.
    base_point : Point
        Fixed base point of the regression trajectory.
    points : sequence of Point
        Observed points used to fit the regression.
    times : array-like, shape (n_points,)
        Observation times.
    """

    @property
    def flow(self):
        """Discrete sampling of the fitted geodesic."""
        paths = pdefoio.find_geodesic_flow(self.dirname)

        return Flow(
            [
                Point(
                    f"{self.id}|tp{index}",
                    vtk_path=path,
                )
                for index, path in enumerate(paths)
            ]
        )


class ExternalForces:
    """Time-discretized external forces for an LDDMM spline.

    Parameters
    ----------
    array : array-like, shape (n_time_points, n_control_points, dim)
        External-force vectors at each spline time point and control point.
    """

    def __init__(self, array):
        self._array = array

    def as_array(self):
        """Return the external forces as an array."""
        return self._array

    @classmethod
    def from_file(cls, filename):
        """Create lazily loaded external forces from a file."""
        return StoredExternalForces(filename)


class StoredExternalForces(ExternalForces):
    """Filesystem-backed external forces."""

    def __init__(self, filename):
        self.filename = filename

    def as_path(self):
        """Return the path to the stored external-forces file."""
        return self.filename

    def as_array(self):
        """Load and return the external forces."""
        return pdefoio.read_array(self.filename)


class SplineRegressionResult(RegressionResult):
    """Filesystem-backed result of spline regression."""

    @property
    def external_forces(self):
        """Estimated time-discretized external forces."""
        return ExternalForces.from_file(pdefoio.find_external_forces(self.dirname))
