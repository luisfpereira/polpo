import os

from api.deformetrica import Deformetrica

import polpo.utils as putils
from polpo.deformetrica.io import load_template


def estimate_deterministic_atlas(
    targets,
    output_dir,
    source=None,
    t0=0,
    max_iter=200,
    kernel_width=15.0,
    regularisation=1.0,
    number_of_time_steps=11,
    metric="landmark",
    kernel_type="torch",
    kernel_device="auto",
    initial_control_points=None,
    tol=1e-5,
    freeze_control_points=False,
    use_rk2_for_flow=False,
    use_rk2_for_shoot=False,
    preserve_volume=False,
    use_svf=False,
    dimension=3,
    print_every=20,
    attachment_kernel_width=4.0,
    initial_step_size=1e-4,
    verbosity="INFO",
):
    r"""Atlas computation.

    Estimates an average shape from a collection of shapes, and the
    deformations from this average to each sample in the collection. This is similar to
    computing a Frechet mean, i.e. to minimize the following:

    ..math::

         C(c, \mu) = \frac{1}{\alpha^2} \sum_i \left( d(q_i, \phi_1^{c_i,\mu_i}(\bar{q}))^2 + \|
         v_0^{c_i,\mu_i} \|_K^2 \right).

    where $c_i, \mu_i$ are the control points and momenta that parametrize the deformation, $v_0^{c,
    \mu}$ is the associated velocity field defined by the convolution $v_t(x) = \sum_{k=1}^{N_c}
    K(x, c^{(t)}_k) \mu^{(t)}_K$, K is the Gaussian kernel, $\phi_1^{c,\mu}$ is the flow of $v_t$
    at time 1, $\bar{q}$ is the source shape being deformed, $q$ is the target shape,
    and $\alpha$ is a regularization term that controls the tradeoff between exact matching and
    smoothness of the deformation. $d$ is a distance function on shapes (point-to-point L2,
    varifold, metric, etc).

    Resulting control points and momenta are saved in the ouput dir as txt files.

    Parameters
    ----------
    source: str or pathlib.Path
        Path to the vtk file that contains the source mesh.
    targets: list
        Path to the vtk files that contain the target meshes.
    output_dir: str or pathlib.Path
        Path a directory where results will be saved.
    kernel_width: float
        Width of the Gaussian kernel. Controls the spatial smoothness of the deformation and
        influences the number of parameters required to represent the deformation.
        Optional, default: 20.
    regularisation: float
        $\alpha$ in the above equation. Smaller values will yeild larger deformations to reduce
        the data attachment term, while larger values will allow attachment errors for a smoother
        deformation.
        Optional, default: 1.
    number_of_time_steps: int
        Number used in the discretization of the flow equation.
        Optional, default: 10.
    metric: str, {landmark, varifold, current}
        Metric to use to measure attachment between meshes. Landmark refers to L2.
    attachment_kernel_width: float,
        If using varifold or currents, width of the kernel used in the attachment metric. Defines
        the scale at which differences must be taken into account.
    dimension: int {2, 3}
        Dimension of the shape embedding space.
    kernel_type: str, {torch, keops}
        Package to use for convolutions of velocity fields and loss functions.
    kernel_device: str, {cuda, cpu}
    use_svf: bool
        Whether to use stationary velocity fields instead of time evolving velocity. The
        deformation is no longer a geodesic but there is more symmetry wrt source / target.
        Optional, default: False
    initial_control_points: str or pathlib.Path
        Path to the txt file that contains the initial control points.
        Optional
    freeze_control_points: bool
        Whether to optimize control points jointly with momenta.
        Optional, default: False
    preserve_volume: bool
        Whether to use volume preserving deformation. This modifies the metric on deformations.
        Optional, default: False
    use_rk2_for_flow: bool
        Whether to use Runge-Kutta order 2 steps in the integration of the flow equation, i.e. when
        warping the shape. If False, a Euler step is used.
        Optional, default: False
    use_rk2_for_shoot: bool
        Whether to use Runge-Kutta order 2 steps in the integration of the Hamiltonian equation that
        governs the time evolution of control points and momenta. If False, a Euler step is used.
        Optional, default: False
    print_every: int
        Sets the verbosity level of the optimization scheme.
    max_iter: int
        Maximum number of iteration in the optimization scheme.
        Optional, default: 200.
    tol: float
        Tolerance to evaluate convergence.
    initial_step_size: float
        Initial learning rate.
        Optional, default: 1e-4.
    """
    if source is None:
        source = putils.get_first(targets)

    template = {
        "shape": {
            "deformable_object_type": "SurfaceMesh",
            "kernel_type": kernel_type,
            "kernel_width": attachment_kernel_width,
            "kernel_device": kernel_device,
            "noise_std": regularisation,
            "filename": source,
            "noise_variance_prior_scale_std": None,
            "noise_variance_prior_normalized_dof": 0.01,
            "attachment_type": metric,
        }
    }

    if isinstance(targets, dict):
        targets_ = [{"shape": target} for target in targets.values()]
        subject_ids = [str(key) for key in targets.keys()]
    else:
        targets_ = [{"shape": target} for target in targets]
        subject_ids = [str(index) for index in range(len(targets_))]

    data_set = {
        "dataset_filenames": [[k] for k in targets_],
        "visit_ages": None,
        "subject_ids": subject_ids,
    }

    model = {
        "deformation_kernel_type": kernel_type,
        "deformation_kernel_width": kernel_width,
        "deformation_kernel_device": kernel_device,
        "number_of_time_points": number_of_time_steps,
        "concentration_of_time_points": number_of_time_steps - 1,
        "use_rk2_for_flow": use_rk2_for_flow,
        "use_rk2_for_shoot": use_rk2_for_shoot,
        "use_svf": use_svf,
        "preserve_volume": preserve_volume,
        "freeze_template": False,
        "freeze_control_points": freeze_control_points,
        "freeze_momenta": False,
        "freeze_noise_variance": False,
        "use_sobolev_gradient": True,
        "sobolev_kernel_width_ratio": 1,
        "initial_control_points": initial_control_points,
        "initial_cp_spacing": None,
        "initial_momenta": None,
        "dense_mode": False,
        "number_of_processes": 1,
        "dimension": dimension,
        "random_seed": None,
        "t0": t0,
        "tmin": t0,
        "tmax": 1.0,
    }

    optimization_parameters = {
        "max_iterations": max_iter,
        "freeze_template": False,
        "freeze_control_points": freeze_control_points,
        "freeze_momenta": False,
        "use_sobolev_gradient": True,
        "sobolev_kernel_width_ratio": 1,
        "max_line_search_iterations": 50,
        "initial_control_points": initial_control_points,
        "initial_cp_spacing": None,
        "initial_momenta": None,
        "dense_mode": False,
        "number_of_threads": 1,
        "print_every_n_iters": print_every,
        "downsampling_factor": 1,
        "dimension": dimension,
        "optimization_method_type": "ScipyLBFGS",
        "convergence_tolerance": tol,
        "initial_step_size": initial_step_size,
    }

    deformetrica = Deformetrica(output_dir, verbosity=verbosity)
    deformetrica.estimate_deterministic_atlas(
        template_specifications=template,
        dataset_specifications=data_set,
        model_options=model,
        estimator_options=optimization_parameters,
    )

    return load_template(output_dir, as_path=True)


def estimate_spline_regression(
    source,
    targets,
    output_dir,
    times,
    subject_id=None,
    t0=0,
    max_iter=200,
    kernel_width=15.0,
    regularisation=1.0,
    number_of_time_steps=10,
    initial_step_size=1e-4,
    kernel_type="torch",
    kernel_device="cuda",
    initial_control_points=None,
    tol=1e-5,
    freeze_control_points=False,
    use_rk2_for_flow=False,
    use_rk2_for_shoot=False,
    dimension=3,
    freeze_external_forces=False,
    target_weights=None,
    geodesic_weight=0.1,
    metric="landmark",
    attachment_kernel_width=15.0,
    print_every=20,
    verbosity="INFO",
):
    r"""Geodesic or Spline Regression.

    Estimates the best possible time-constrained deformation to fit a set of observations indexed
    by a covariable.

    The following criterion is minimized:

    ..math::

        C_S(c, \mu, u_t) &=  \frac{1}{\alpha^2d} \sum_{i=1}^d d( x_{t_i}, \phi_{t_i}(x_{t_0}))^2  +
        \int_0^1 \|u^{(t)}\|^2 dt + \|v_0^{c,\mu}\|_K^2,

    where $x_{t_i}$ are the observations observed at variable $t_i$, $c,\mu, u$ parametrize the
    deformation. $c,\mu$ define a velocity field by the convolution $v_t(x) = \sum_{k=1}^{N_c}
    K(x, c^{(t)}_k) \mu^{(t)}_K$ where K is the Gaussian kernel. $u^t$ is a second-order term
    that can be interpreted as random external forces smoothly perturbing the trajectory around a
    mean geodesic. If `freeze_external_forces` is set to True, they are fixed to 0 and in this
    case the regression model estimates a geodesic.

    Parameters
    ----------
    source: str or pathlib.Path
        Path to the vtk file that contains the source mesh.
    targets: list of dict
        Path to the vtk files that contain the target meshes. Must be formatted as a list of
        dictionaries, where each dict represents a time points and has a key 'shape' with the
        path to the shape as value.
    times: list of floats in [0, 1]
        Covariable used in the regression.
    subject_id: list of str
        Not used.
    t0: float
        Time of the first shape.
    initial_step_size: float
        Initial learning rate.
        Optional, default: 1e-4.
    freeze_external_forces: bool
        Whether to use external forces in the regression model. When used, splines are used
        instead of geodesics.
    output_dir: str or pathlib.Path
        Path to a directory where results will be saved. It will be created if it does not
        already exist.
    kernel_width: float
        Width of the Gaussian kernel. Controls the spatial smoothness of the deformation and
        influences the number of parameters required to represent the deformation.
        Optional, default: 20.
    regularisation: float
        $\alpha$ in the above equation. Smaller values will yeild larger deformations to reduce
        the data attachment term, while larger values will allow attachment errors for a smoother
        deformation.
        Optional, default: 1.
    number_of_time_steps: int
        Number used in the discretization of the flow equation.
        Optional, default: 10.
    metric: str, {landmark, varifold, current}
        Metric to use to measure attachment between meshes. Landmark refers to L2.
    attachment_kernel_width: float,
        If using varifold or currents, width of the kernel used in the attachment metric. Defines
        the scale at which differences must be taken into account.
    dimension: int {2, 3}
        Dimension of the shape embedding space.
    kernel_type: str, {torch, keops}
        Package to use for convolutions of velocity fields and loss functions.
    kernel_device: str, {cuda, cpu}
    initial_control_points: str or pathlib.Path
        Path to the txt file that contains the initial control points.
        Optional
    freeze_control_points: bool
        Whether to optimize control points jointly with momenta.
        Optional, default: False
    use_rk2_for_flow: bool
        Whether to use Runge-Kutta order 2 steps in the integration of the flow equation, i.e. when
        warping the shape. If False, a Euler step is used.
        Optional, default: False
    use_rk2_for_shoot: bool
        Whether to use Runge-Kutta order 2 steps in the integration of the Hamiltonian equation that
        governs the time evolution of control points and momenta. If False, a Euler step is used.
        Optional, default: False
    print_every: int
        Sets the verbosity level of the optimization scheme.
    target_weights: list or array
        Coefficient to weight observations' contributions to the loss function.
    geodesic_weight: float
        Coefficient to weight the geodesic part compared to the external forces.
        Optional, default: 0.1.
    filter_cp: bool
        Whether to filter control points saved in the vtk file to exclude those whose momenum
        vector is not significative and does not contribute to the deformation.
        Optional, default: False
    threshold: float
        Threshold to use on momenta norm when filtering. Ignored if `filter_cp` is set to `False`.
    max_iter: int
        Maximum number of iteration in the optimization scheme.
        Optional, default: 200.
    tol: float
        Tolerance to evaluate convergence.
    """
    if subject_id is None:
        subject_id = ["patient"]
    template = {
        "shape": {
            "deformable_object_type": "SurfaceMesh",
            "kernel_type": kernel_type,
            "kernel_width": attachment_kernel_width,
            "kernel_device": kernel_device,
            "noise_std": regularisation,
            "filename": source,
            "noise_variance_prior_scale_std": None,
            "noise_variance_prior_normalized_dof": 0.01,
            "attachment_type": metric,
        }
    }

    data_set = {
        "visit_ages": [times],
        "dataset_filenames": [{"shape": target} for target in targets],
        "subject_ids": subject_id,
    }

    model = {
        "deformation_kernel_type": kernel_type,
        "deformation_kernel_width": kernel_width,
        "deformation_kernel_device": kernel_device,
        "number_of_time_points": number_of_time_steps + 1,
        "concentration_of_time_points": number_of_time_steps,
        "use_rk2_for_flow": use_rk2_for_flow,
        "use_rk2_for_shoot": use_rk2_for_shoot,
        "freeze_template": True,
        "freeze_control_points": freeze_control_points,
        "freeze_external_forces": freeze_external_forces,
        "freeze_momenta": False,
        "freeze_noise_variance": False,
        "use_sobolev_gradient": True,
        "sobolev_kernel_width_ratio": 1,
        "initial_control_points": initial_control_points,
        "initial_cp_spacing": None,
        "initial_momenta": None,
        "dense_mode": False,
        "number_of_processes": 1,
        "dimension": dimension,
        "random_seed": None,
        "t0": t0,
        "tmin": min(times),
        "tmax": max(times),
        "target_weights": target_weights,
        "geodesic_weight": geodesic_weight,
    }

    optimization_parameters = {
        "initial_step_size": initial_step_size,
        "scale_initial_step_size": True,
        "line_search_shrink": 0.5,
        "line_search_expand": 1.5,
        "max_line_search_iterations": 30,
        "optimized_log_likelihood": "complete",
        "optimization_method_type": "ScipyLBFGS",
        "max_iterations": max_iter,
        "convergence_tolerance": tol,
        "print_every_n_iters": print_every,
        "save_every_n_iters": 100,
        "state_file": None,
        "load_state_file": False,
    }

    if subject_id != "patient":
        patient_output_dir = os.path.join(output_dir, subject_id[0])
    else:
        patient_output_dir = output_dir

    deformetrica = Deformetrica(patient_output_dir, verbosity=verbosity)
    deformetrica.estimate_spline_regression(
        template_specifications=template,
        dataset_specifications=data_set,
        model_options=model,
        estimator_options=optimization_parameters,
    )

    # # TODO: uncomment
    # # agregate results in vtk file for paraview
    # path_cp = join(output_dir, lddmm_strings.cp_str_spline)
    # cp = read_2D_array(path_cp)
    # path_momenta = join(output_dir, lddmm_strings.mom_str_spline)
    # momenta = read_3D_array(path_momenta)
    # poly_cp = momenta_to_vtk(cp, momenta, kernel_width, filter_cp, threshold)
    # poly_cp.save(join(output_dir, "initial_control_points.vtk"))

    # if not freeze_external_forces:
    #     forces = read_3D_array(join(output_dir, lddmm_strings.ext_forces_str))
    #     external_forces_to_vtk(cp, forces, output_dir, filter_cp, threshold)
