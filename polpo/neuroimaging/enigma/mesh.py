from importlib.resources import files
from pathlib import Path

import numpy as np
import pyvista as pv

import polpo.preprocessing.dict as ppdict
import polpo.utils as putils
from polpo.freesurfer.mesh import FreeSurferReader
from polpo.preprocessing import Map
from polpo.preprocessing.path import (
    FileFinder,
    PathShortener,
)
from polpo.preprocessing.str import (
    DigitFinder,
    EndsWithAny,
    StartsWith,
)
from polpo.pyvista.conversion import PvFromData

from .naming import (
    aseg_id_to_name,
    get_all_subcortical_structs,
    name_to_aseg_id,
)
from .validation import validate_structs


def MeshReader():
    return FreeSurferReader() + PvFromData()


def MeshDatasetLoader(struct_subset=None, mesh_reader=False):
    # TODO: rename to MeshDatasetLoader
    # pipeline takes dirname
    if mesh_reader is None:
        mesh_reader = MeshReader()
    elif mesh_reader is False:
        mesh_reader = None

    if struct_subset is None:
        struct_subset = get_all_subcortical_structs()

    validate_structs(struct_subset)

    enigma_indices = [f"_{name_to_aseg_id(struct)}" for struct in struct_subset]
    rules = [StartsWith("resliced_mesh"), EndsWithAny(enigma_indices)]
    path_to_struct_id = PathShortener() + DigitFinder(index=-1) + aseg_id_to_name

    return FileFinder(rules=rules, as_list=True) + ppdict.HashWithIncoming(
        key_step=Map(path_to_struct_id),
        step=Map(mesh_reader),
        key_sorter=putils.custom_order(struct_subset),
    )


def read_ccbbm(filename, index_base=1):
    """Read a CCBBM/ShapeTools triangular mesh.

    This parser reads the ASCII mesh format produced by the
    ShapeTools library and related CCBBM tooling.

    The parser extracts:
    - vertex coordinates from ``Vertex`` records,
    - triangular connectivity from ``Face`` or ``Triangle``
      records.

    Parameters
    ----------
    filename : str or path-like
        Path to the mesh file.

    index_base : int, default=1
        Indexing convention used in the file connectivity.
        Use ``1`` for one-based indexing and ``0`` for
        zero-based indexing.

    Returns
    -------
    vertices : ndarray, shape=[n_vertices, 3]
        Vertex coordinates.
    faces : ndarray, shape=[n_faces, 3]
        Triangle connectivity array.
    """
    # read .m files
    vertices = []
    faces = []

    with open(filename) as file:
        for line in file:
            if line.startswith("Vertex"):
                _, idx, x, y, z = line.split()[:5]
                vertices.append((float(x), float(y), float(z)))

            elif line.startswith(("Face", "Triangle")):
                parts = line.split()
                # likely: Face <id> <v1> <v2> <v3>
                tri = [int(i) - index_base for i in parts[-3:]]
                faces.append(tri)

    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=int)

    return vertices, faces


def load_template(name, as_polydata=True):
    filename = files("polpo.enigma") / "resources" / f"atlas_{name_to_aseg_id(name)}.m"

    vertices, faces = read_ccbbm(filename)

    if as_polydata:
        return pv.PolyData.from_regular_faces(points=vertices, faces=faces)

    return vertices, faces


def read_mni_obj(filename):
    """Read an ASCII MNI/BrainVisa polygon OBJ mesh.

    This format is commonly used in neuroimaging pipelines
    originating from the MNI/MINC ecosystem (e.g. CIVET,
    BrainVisa, ENIGMA-related tooling). Despite the `.obj`
    extension, it is not the standard Wavefront OBJ format.

    The parser assumes:
    - polygon object (`P`) format,
    - triangular faces,
    - ASCII encoding,
    - one normal per vertex,
    - flat connectivity storage.

    The file layout is assumed to be:

    - header line beginning with ``P``,
    - vertex block,
    - normal block,
    - number of faces,
    - material/color block,
    - cumulative polygon-end indices,
    - flattened connectivity array.

    Parameters
    ----------
    filename : str
        Path to the mesh file.

    Returns
    -------
    vertices : ndarray, shape=[n_vertices, 3]
        Vertex coordinates.
    faces : ndarray, shape=[n_faces, 3]
        Triangle connectivity array.

    Notes
    -----
    The cumulative polygon-end block is redundant for triangular
    meshes but is part of the MNI OBJ specification.

    Face indices are zero-based.

    References
    ----------
    MNI MINC / BrainVisa polygon OBJ surface format.
    """
    blocks = Path(filename).read_text().strip().split("\n\n")

    header_and_geometry = blocks[0].splitlines()
    header = header_and_geometry[0]

    n_vertices = int(header.split()[-1])

    vertices = np.loadtxt(header_and_geometry[1 : 1 + n_vertices])

    face_block = blocks[1].splitlines()
    n_faces = int(face_block[0])

    faces = np.fromstring(
        blocks[3],
        sep=" ",
        dtype=int,
    ).reshape(n_faces, 3)

    return vertices, faces


_TEMPLATE_N_VERTICES = {
    "Accu": 930,
    "Amyg": 1368,
    "Caud": 2502,
    "Hipp": 2502,
    "Pall": 1254,
    "Puta": 2502,
    "Thal": 2502,
}


def get_template_n_vertices(struct):
    if "_" in struct:
        struct = struct.split("_")[1]

    return _TEMPLATE_N_VERTICES[struct]
