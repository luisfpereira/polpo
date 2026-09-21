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
    """Find the final control points in a Deformetrica output directory.

    The deterministic-atlas and generic final-control-point naming
    conventions are both supported.

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
    for name in (
        "DeterministicAtlas__EstimatedParameters__ControlPoints.txt",
        "final_cp.txt",
    ):
        path = dirname / name
        if path.exists():
            return path

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
    for name in (
        "DeterministicAtlas__EstimatedParameters__Momenta.txt",
        "transported_momenta.txt",
    ):
        path = dirname / name
        if path.exists():
            return path

    raise FileNotFoundError("Could not find momenta.")


def find_atlas_momenta(dirname, id_):
    """Find deterministic-atlas momenta for one subject.

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
        (
            path
            for path in dirname.glob("*.txt")
            if "__Momenta__" in path.name and f"__subject_{id_}" in path.name
        ),
        f"deterministic atlas momenta for {id_!r}",
    )


def find_atlas_reconstruction(dirname, id_):
    """Find the deterministic-atlas reconstruction for one subject.

    Parameters
    ----------
    dirname : pathlib.Path
        Deterministic-atlas output directory.
    id_ : str
        Subject identifier encoded in the Deformetrica filename.

    Returns
    -------
    path : pathlib.Path
        Path to the reconstructed VTK file.
    """
    return _find_unique(
        (
            path
            for path in dirname.glob("*.vtk")
            if "__Reconstruction__" in path.name and f"__subject_{id_}" in path.name
        ),
        f"deterministic atlas reconstruction for {id_!r}",
    )


def find_atlas_flow(dirname, id_):
    """Find the sampled deterministic-atlas flow for one subject.

    Parameters
    ----------
    dirname : pathlib.Path
        Deterministic-atlas output directory.
    id_ : str
        Subject identifier encoded in the Deformetrica filenames.

    Returns
    -------
    paths : list of pathlib.Path
        Flow VTK files ordered by time-point index.
    """
    paths = [
        path
        for path in dirname.glob("*.vtk")
        if "__flow__" in path.name and f"__subject_{id_}" in path.name
    ]

    return sorted(paths, key=_time_index)


def find_shooting_flow(dirname):
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
    paths = [path for path in dirname.glob("*.vtk") if "__GeodesicFlow__" in path.name]

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
    paths = [path for path in dirname.glob("*.vtk") if "parallel_curve" in path.name]

    return sorted(paths, key=_time_index)


def find_transported_control_points(dirname):
    """Find the final transported control points.

    The explicit final-control-point file is preferred when present.
    Otherwise, the last control-points file in the sampled flow is returned.

    Parameters
    ----------
    dirname : pathlib.Path
        Parallel-transport output directory.

    Returns
    -------
    path : pathlib.Path
        Path to the final transported control points.

    Raises
    ------
    FileNotFoundError
        If no transported control points can be found.
    """
    path = dirname / "final_cp.txt"
    if path.exists():
        return path

    paths = [path for path in dirname.glob("*.txt") if "ControlPoints" in path.name]

    if not paths:
        raise FileNotFoundError("Could not find transported control points.")

    return sorted(paths, key=_time_index)[-1]


def find_transported_momenta(dirname):
    """Find the final transported momenta.

    The explicit transported-momenta file is preferred when present.
    Otherwise, the last transported momenta file in the sampled flow is
    returned.

    Parameters
    ----------
    dirname : pathlib.Path
        Parallel-transport output directory.

    Returns
    -------
    path : pathlib.Path
        Path to the final transported momenta.

    Raises
    ------
    FileNotFoundError
        If no transported momenta can be found.
    """
    path = dirname / "transported_momenta.txt"
    if path.exists():
        return path

    paths = [
        path
        for path in dirname.glob("*.txt")
        if "Momenta" in path.name and "Transported" in path.name
    ]

    if not paths:
        raise FileNotFoundError("Could not find transported momenta.")

    return sorted(paths, key=_time_index)[-1]
