import shutil
import string

import polpo.utils as putils
from polpo.pipeline.mesh.registration import RigidAlignment
from polpo.surface_mesh.deformetrica import LddmmMetric, Point
from polpo.surface_mesh.generation.blob import create_blob

if __name__ == "__main__":
    n_meshes = 3
    bump_amp = 0.2
    outputs_dir = putils.get_results_path() / "blobs/transport_abc"

    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)

    outputs_dir.mkdir(parents=True, exist_ok=False)

    raw_meshes = [
        create_blob(resolution=10, bump_amp=bump_amp, n_bumps=5, smoothing_iter=10)
        for _ in range(n_meshes)
    ]

    prep_pipe = RigidAlignment(known_correspondences=True)

    meshes = prep_pipe(raw_meshes)

    kernel_width = 2 * bump_amp
    registration_kwargs = dict(
        kernel_width=kernel_width,
        regularisation=1.0,
        max_iter=2000,
        freeze_control_points=False,
        metric="varifold",
        tol=1e-16,
        attachment_kernel_width=bump_amp,
    )

    metric = LddmmMetric(outputs_dir, use_pole_ladder=True, **registration_kwargs)

    point_a, point_b, point_c = [
        Point(
            id_=string.ascii_uppercase[index],
            pv_surface=mesh,
            dirname=metric.dir_config.meshes_dir,
        )
        for index, mesh in enumerate(meshes)
    ]

    vec_ba = metric.log(point_a, point_b)
    vec_bc = metric.log(point_c, point_b)

    trans_vec_bc_pole = metric.parallel_transport(vec_bc, point_b, direction=vec_ba)
    trans_point_c_pole = metric.exp(trans_vec_bc_pole, point_a)

    metric.use_pole_ladder = False
    trans_vec_bc_fan = metric.parallel_transport(vec_bc, point_b, direction=vec_ba)
    trans_point_c_fan = metric.exp(trans_vec_bc_fan, point_a)
