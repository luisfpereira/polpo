from pathlib import Path

import numpy as np
import pyvista as pv

import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.path as pppath
import polpo.preprocessing.str as ppstr
import polpo.utils as putils
from polpo.preprocessing import FilteredGroupBy, Map, Sorter
from polpo.preprocessing.mesh.conversion import DataFromPv

# TODO: functions need to be renamed

# TODO: deeply simplify this file


def read_array(path):
    with open(path, "r") as file:
        first_line = next(file)
        second_line = next(file)

        if second_line != "\n":
            return np.loadtxt(path)

        n_arrays = int(first_line.split(" ")[0])
        array = np.loadtxt(file)
        if n_arrays == 1:
            return array

        n, dim = array.shape
        return array.reshape((n_arrays, n // n_arrays, dim))


def write_array(path, array, header=True):
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


def load_vtk_mesh(path, as_pv=False):
    pv_mesh = pv.read(path)
    if as_pv:
        return pv_mesh

    # vertices, faces
    return DataFromPv()(pv_mesh)


def _file2mesh(as_pv=False):
    return lambda path: load_vtk_mesh(path, as_pv=as_pv)


def LoadMeshFlow(as_path=False, as_pv=True, extra_rules=()):
    # TODO: index by time bool

    # as_path as precedence to as_pv
    path_to_tp = ppstr.RegexGroupFinder(r"_tp_(\d+)") + int

    rules = [pppath.IsFileType("vtk")] + putils.as_list(extra_rules)

    file_finder = pppath.FileFinder(
        rules=rules,
        as_list=True,
    ) + ppdict.HashWithIncoming(key_step=Map(path_to_tp), key_sorter=lambda x: x)

    if as_path:
        return file_finder

    return file_finder + ppdict.DictMap(_file2mesh(as_pv=as_pv))


def LoadControlPointsFlow(as_path=False, as_array=True, extra_rules=()):
    # as_path as precedence to as_array

    # TODO: check change
    path_to_tp = (lambda x: x.as_posix()) + ppstr.RegexGroupFinder(r"_tp_(\d+)") + int

    rules = [
        pppath.IsFileType("txt"),
        ppstr.Contains("ControlPoints"),
    ] + putils.as_list(extra_rules)

    file_finder = pppath.FileFinder(
        rules=rules,
        as_list=True,
    ) + ppdict.HashWithIncoming(key_step=Map(path_to_tp), key_sorter=lambda x: x)

    if as_path:
        return file_finder

    pipe = file_finder + ppdict.DictMap(read_array)
    if not as_array:
        return pipe

    return pipe + ppdict.DictToValuesList() + np.stack


def LoadMomentaFlow(as_path=False, as_array=True, extra_rules=()):
    # as_path as precedence to as_array

    # TODO: check change
    path_to_tp = (lambda x: x.as_posix()) + ppstr.RegexGroupFinder(r"_tp_(\d+)") + int

    rules = [
        pppath.IsFileType("txt"),
        ppstr.Contains("Momenta"),
    ] + putils.as_list(extra_rules)

    file_finder = pppath.FileFinder(
        rules=rules,
        as_list=True,
    ) + ppdict.HashWithIncoming(key_step=Map(path_to_tp), key_sorter=lambda x: x)

    if as_path:
        return file_finder

    pipe = file_finder + ppdict.DictMap(read_array)

    if not as_array:
        return pipe

    return pipe + ppdict.DictToValuesList() + np.stack


def get_deterministic_atlas_reconstruction_names(path, subset=None):
    # TODO: move

    # in DeterministicAtlas._write_model_predictions
    # name = self.name + '__Reconstruction__' + object_name + '__subject_' + subject_id + object_extension

    file_finder = pppath.FileFinder(
        rules=[pppath.IsFileType("vtk"), ppstr.Contains("__Reconstruction__")],
        as_list=True,
    )

    # TODO: check change
    path_to_id = (
        (lambda x: x.as_posix())
        + ppstr.RegexGroupFinder(r"__subject_([A-Za-z0-9-]+)")
        + ppstr.TryToInt()
    )

    sorter = None
    if subset is not None and len(subset) > 1:
        sorter = putils.custom_order(subset)

    file_finder += ppdict.HashWithIncoming(
        key_step=Map(path_to_id), key_sorter=sorter, key_subset=subset
    )

    # dict[str or int: str]
    return file_finder(path)


def get_deterministic_atlas_flow_names(path, subset=None):
    # TODO: move

    # in DeterministicAtlas._write_model_predictions
    # name = self.name + '__flow__' + object_name + '__subject_' + subject_id + "__tp_" + str(j) + object_extension

    file_finder = pppath.FileFinder(
        rules=[pppath.IsFileType("vtk"), ppstr.Contains("__flow__")],
    )

    path_to_id = (
        (lambda x: x.as_posix())
        + ppstr.RegexGroupFinder(r"__subject_([A-Za-z0-9-]+)")
        + ppstr.TryToInt()
    )
    path_to_tp = (
        (lambda x: x.as_posix()) + pppath.PathShortener() + ppstr.DigitFinder(index=-1)
    )

    # TODO: try to call path_to_id only once if no subset
    sorter = Sorter(path_to_id) if subset is None else None

    file_finder += (
        sorter
        + FilteredGroupBy(path_to_id, subset=subset)
        + ppdict.DictMap(Sorter(path_to_tp))
    )

    # dict[str or int: list[str]]
    return file_finder(path)


def get_shooting_flow_names(path):
    # TODO: homogenize with get_deterministic_atlas_flow_names?

    file_finder = pppath.FileFinder(
        rules=[pppath.IsFileType("vtk"), ppstr.Contains("__GeodesicFlow__")],
    )

    path_to_tp = (
        pppath.PathShortener()
        + ppstr.RegexGroupFinder(pattern=r"__tp_(\d+)_")
        + (lambda x: int(x))
    )
    file_finder += Sorter(path_to_tp)

    # list[str]
    return file_finder(path)


def get_parallel_shooting_flow_names(path):
    file_finder = pppath.FileFinder(
        rules=[pppath.IsFileType("vtk"), ppstr.Contains("parallel_curve")],
    )

    path_to_tp = (
        pppath.PathShortener()
        + ppstr.RegexGroupFinder(pattern=r"_tp_(\d+)_")
        + (lambda x: int(x))
    )
    file_finder += Sorter(path_to_tp)

    # list[str]
    return file_finder(path)


def load_template(path, as_path=False, as_pv=False):
    # as_pv is ignored if as_path is True

    file_finder = pppath.FileFinder(
        rules=[
            pppath.IsFileType("vtk"),
            ppstr.Contains("__EstimatedParameters__Template"),
        ],
    )

    filename = file_finder(path)

    if as_path:
        return Path(filename)

    return _file2mesh(as_pv)(filename)


def load_cp(path, as_path=False):
    possible_filenames = (
        "DeterministicAtlas__EstimatedParameters__ControlPoints.txt",
        "final_cp.txt",
    )

    for name in possible_filenames:
        cp_name = path / name
        if cp_name.exists():
            break
    else:
        raise FileNotFoundError("Can't find any control points.")

    if as_path:
        return cp_name

    return read_array(cp_name)


def load_transported_cp(path, as_path=False):
    # if pole ladder
    cp_name = path / "final_cp.txt"
    if not cp_name.exists():
        cp_names = LoadControlPointsFlow(as_path=True)(path)
        cp_name = cp_names[list(cp_names.keys())[-1]]

    if as_path:
        return cp_name

    return read_array(cp_name)


def get_deterministic_atlas_momenta_names(path, subset=None):
    # NB: part of the way we split it now in polpo

    file_finder = pppath.FileFinder(
        rules=[
            pppath.IsFileType("txt"),
            ppstr.ContainsAll(("__Momenta__", "__subject_")),
        ],
    )

    path_to_id = ppstr.RegexGroupFinder(r"__subject_([A-Za-z0-9-]+)")

    sorter = Sorter(path_to_id) if subset is None else None

    file_finder += sorter + FilteredGroupBy(path_to_id, subset=subset)

    return file_finder(path)


def load_deterministic_atlas_momenta(path, id_, as_path=False):
    filenames = get_deterministic_atlas_flow_names(path, subset=[id_])
    mom_name = ppdict.ExtractUniqueKey()(filenames)

    if as_path:
        return mom_name

    return read_array(mom_name)


def load_momenta(path, as_path=False):
    # TODO: put together with load_deterministic_atlas_momenta?
    possible_filenames = [
        "DeterministicAtlas__EstimatedParameters__Momenta.txt",
        "transported_momenta.txt",
    ]

    for name in possible_filenames:
        mom_name = path / name
        if mom_name.exists():
            break
    else:
        raise FileNotFoundError("Can't find any momenta")

    if as_path:
        return mom_name

    return read_array(mom_name)


def load_transported_momenta(path, as_path=False):
    # TODO: really needed?
    # if pole ladder
    mom_name = path / "transported_momenta.txt"
    if not mom_name.exists():
        mom_names = LoadMomentaFlow(
            as_path=True, extra_rules=ppstr.Contains("Transported")
        )(path)
        mom_name = mom_names[list(mom_names.keys())[-1]]

    if as_path:
        return mom_name

    return read_array(mom_name)


def load_deterministic_atlas_reconstructions(
    path, subset=None, as_pv=False, as_path=False
):
    # as_pv is ignored if as_path is True
    filenames = get_deterministic_atlas_reconstruction_names(path, subset=subset)
    if as_path:
        return filenames

    return ppdict.DictMap(_file2mesh(as_pv))(filenames)


def load_deterministic_atlas_reconstruction(path, as_pv=False, as_path=False, id_=None):
    # TODO: can do better
    subset = None
    if id_ is not None:
        subset = (id_,)

    # NB: convenient
    meshes = load_deterministic_atlas_reconstructions(
        path, as_pv=as_pv, as_path=as_path, subset=subset
    )
    return ppdict.ExtractUniqueKey()(meshes)


def load_deterministic_atlas_flows(path, subset=None, as_pv=False, as_path=False):
    # as_pv is ignored if as_path is True
    filenames = get_deterministic_atlas_flow_names(path, subset=subset)
    if as_path:
        return filenames

    return ppdict.NestedDictMap(_file2mesh(as_pv), inner_is_dict=False)(filenames)


def load_deterministic_atlas_flow(path, as_pv=False, as_path=False, id_=None):
    # TODO: check use
    # TODO: improve if used
    # NB: convenient
    subset = None
    if id_ is not None:
        subset = (id_,)

    meshes = load_deterministic_atlas_flows(
        path, as_pv=as_pv, as_path=as_path, subset=subset
    )
    return ppdict.ExtractUniqueKey()(meshes)


def load_shooting_flow(path, as_pv=False, as_path=False):
    # TODO: maybe add last and reduce number methods?
    filenames = get_shooting_flow_names(path)
    if as_path:
        return filenames

    return Map(_file2mesh(as_pv))(filenames)


def load_shooted_point(path, as_pv=False, as_path=False):
    filename = get_shooting_flow_names(path)[-1]
    if as_path:
        return filename

    return _file2mesh(as_pv)(filename)


def load_parallel_shooted_point(path, as_pv=False, as_path=False):
    filename = get_parallel_shooting_flow_names(path)[-1]
    if as_path:
        return filename

    return _file2mesh(as_pv)(filename)
