"""I/O and output-discovery utilities for Deformetrica."""

import re

import numpy as np


def read_array(path):
    """Read an array written in Deformetrica text format.

    Parameters
    ----------
    path : path-like
        Path to the text file.

    Returns
    -------
    array : ndarray
        Array stored in the file. Multiple stacked arrays are returned with
        shape `(n_arrays, n_points, dim)`.
    """
    with open(path, "r") as file:
        first_line = file.readline()
        second_line = file.readline()

        if second_line != "\n":
            return np.atleast_2d(np.loadtxt(path))

        n_arrays = int(first_line.split(" ")[0])
        array = np.loadtxt(file)
        if n_arrays == 1:
            return np.atleast_2d(array)

        n, dim = array.shape
        return array.reshape((n_arrays, n // n_arrays, dim))


def write_array(path, array, header=True):
    """Write an array in Deformetrica text format.

    Parameters
    ----------
    path : path-like
        Destination file.
    array : ndarray, shape (n_points, dim) or (n_arrays, n_points, dim)
        Array to write.
    header : bool
        Whether to write the Deformetrica array header.
    """
    if not header:
        np.savetxt(path, array)
        return

    if array.ndim == 2:
        n, dim = array.shape
        n_arrays = 1
        flat_array = array

    elif array.ndim == 3:
        n_arrays, n, dim = array.shape
        flat_array = array.reshape(n_arrays * n, dim)

    else:
        raise ValueError(f"Expected a 2D or 3D array, got shape {array.shape}.")

    with open(path, "w") as file:
        file.write(f"{n_arrays} {n} {dim}\n")
        file.write("\n")
        np.savetxt(file, flat_array)


def _find_unique(paths, description):
    """Return the unique path from a collection of candidate paths.

    Parameters
    ----------
    paths : iterable of path-like
        Candidate paths.
    description : str
        Description of the expected file, used in the error message.

    Returns
    -------
    path : pathlib.Path
        Unique matching path.

    Raises
    ------
    FileNotFoundError
        If the number of matching paths is not one.
    """
    paths = list(paths)

    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {description}, found {len(paths)}.")

    return paths[0]


def _time_index(path):
    """Extract the time-point index encoded in a Deformetrica filename.

    Parameters
    ----------
    path : path-like
        Path whose filename contains a `tp_<index>` component.

    Returns
    -------
    index : int
        Time-point index.

    Raises
    ------
    ValueError
        If no time-point index can be extracted.
    """
    match = re.search(r"(?:__|_)tp_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Could not extract time point from {path.name!r}.")

    return int(match.group(1))


def find_template(dirname):
    """Find the estimated template of a deterministic atlas.

    Parameters
    ----------
    dirname : pathlib.Path
        Deformetrica output directory.

    Returns
    -------
    path : pathlib.Path
        Path to the estimated template VTK file.
    """
    return _find_unique(
        dirname.glob("*__EstimatedParameters__Template*.vtk"),
        "deterministic atlas template",
    )


def find_control_points(dirname):
    """Find estimated control points in a Deformetrica output directory.

    Parameters
    ----------
    dirname : pathlib.Path
        Deformetrica output directory.

    Returns
    -------
    path : pathlib.Path
        Path to the control-points file.

    Raises
    ------
    FileNotFoundError
        If no supported control-points file is found.
    """
    paths = list(dirname.glob("*__EstimatedParameters__ControlPoints.txt"))
    if paths:
        return _find_unique(paths, "control-points file")

    path = dirname / "final_cp.txt"
    if path.exists():
        return path

    paths = list(dirname.glob("*ControlPoints*tp_*.txt"))
    if paths:
        return sorted(paths, key=_time_index)[-1]

    raise FileNotFoundError("Could not find control points.")


def find_momenta(dirname):
    """Find the final momenta in a Deformetrica output directory.

    Parameters
    ----------
    dirname : pathlib.Path
        Deformetrica output directory.

    Returns
    -------
    path : pathlib.Path
        Path to the momenta file.

    Raises
    ------
    FileNotFoundError
        If no supported momenta file is found.
    """
    paths = list(dirname.glob("*__EstimatedParameters__Momenta.txt"))
    if paths:
        return _find_unique(paths, "momenta file")

    path = dirname / "transported_momenta.txt"
    if path.exists():
        return path

    paths = list(dirname.glob("*Transported_Momenta*tp_*.txt"))
    if paths:
        return sorted(paths, key=_time_index)[-1]

    raise FileNotFoundError("Could not find momenta.")


def find_subject_momenta(dirname, id_):
    """Find subject-specific momenta in a Deformetrica output directory.

    Parameters
    ----------
    dirname : pathlib.Path
        Deterministic-atlas output directory.
    id_ : str
        Subject identifier encoded in the Deformetrica filename.

    Returns
    -------
    path : pathlib.Path
        Path to the subject-specific momenta file.
    """
    return _find_unique(
        dirname.glob(f"*__Momenta__*__subject_{id_}*.txt"),
        f"momenta for subject {id_!r}",
    )


def find_subject_reconstruction(dirname, id_):
    """Find a subject-specific reconstruction in a Deformetrica output directory.

    Parameters
    ----------
    dirname : pathlib.Path
        Deformetrica output directory.
    id_ : str
        Subject identifier encoded in the Deformetrica filename.

    Returns
    -------
    path : pathlib.Path
        Path to the reconstructed VTK file.
    """
    return _find_unique(
        dirname.glob(f"*__Reconstruction__*__subject_{id_}*.vtk"),
        f"reconstruction for subject {id_!r}",
    )


def find_subject_flow(dirname, id_):
    """Find a subject-specific sampled flow in a Deformetrica output directory.

    Parameters
    ----------
    dirname : pathlib.Path
        Deformetrica output directory.
    id_ : str
        Subject identifier encoded in the Deformetrica filenames.

    Returns
    -------
    paths : list of pathlib.Path
        Flow VTK files ordered by time-point index.
    """
    paths = list(dirname.glob(f"*__flow__*__subject_{id_}*.vtk"))
    return sorted(paths, key=_time_index)


def find_geodesic_flow(dirname):
    """Find the sampled geodesic shooting flow.

    Parameters
    ----------
    dirname : pathlib.Path
        Deformetrica shooting output directory.

    Returns
    -------
    paths : list of pathlib.Path
        Geodesic-flow VTK files ordered by time-point index.
    """
    paths = list(dirname.glob("*__GeodesicFlow__*.vtk"))
    return sorted(paths, key=_time_index)


def find_parallel_shooting_flow(dirname):
    """Find the sampled flow obtained after parallel transport and shooting.

    Parameters
    ----------
    dirname : pathlib.Path
        Parallel-transport output directory.

    Returns
    -------
    paths : list of pathlib.Path
        Parallel-shooting VTK files ordered by time-point index.
    """
    paths = list(dirname.glob("*parallel_curve*.vtk"))
    return sorted(paths, key=_time_index)


def find_reconstructions(dirname):
    """Find reconstructed surfaces in a Deformetrica output directory.

    Parameters
    ----------
    dirname : pathlib.Path
        Deformetrica output directory.

    Returns
    -------
    paths : list of pathlib.Path
        Reconstruction VTK files ordered by time-point index.
    """
    paths = list(dirname.glob("*__Reconstruction__*.vtk"))
    return sorted(paths, key=_time_index)


def find_external_forces(dirname):
    """Find estimated external forces in a Deformetrica output directory.

    Parameters
    ----------
    dirname : pathlib.Path
        Deformetrica output directory.

    Returns
    -------
    path : pathlib.Path
        Path to the external-forces file.
    """
    return _find_unique(
        dirname.glob("*__EstimatedParameters__ExternalForces.txt"),
        "external-forces file",
    )
