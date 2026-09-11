from api.deformetrica import Deformetrica


def estimate_registration(
    source,
    target,
    output_dir,
    target_id="target",
    kernel_width=20.0,
    regularisation=1.0,
    number_of_time_steps=10,
    metric="landmark",
    kernel_type="torch",
    kernel_device="cuda",
    tol=1e-5,
    use_svf=False,
    initial_control_points=None,
    max_iter=200,
    freeze_control_points=False,
    use_rk2_for_shoot=False,
    use_rk2_for_flow=False,
    dimension=3,
    use_rk4_for_shoot=False,
    preserve_volume=False,
    print_every=20,
    attachment_kernel_width=4.0,
    verbosity="INFO",
):
    r"""Registration.

    Estimates the best possible deformation between two shapes, i.e. minimizes the following
    criterion:

    ..math::

         C(c, \mu) = \frac{1}{\alpha^2} d(q, \phi_1^{c,\mu}(\bar{q}))^2 + \| v_0^{c,
         \mu} \|_K^2.

    where $c, \mu$ are the control points and momenta that parametrize the deformation, $v_0^{c,
    \mu}$ is the associated velocity field defined by the convolution $v_t(x) = \sum_{k=1}^{N_c}
    K(x, c^{(t)}_k) \mu^{(t)}_K$, K is the Gaussian kernel, $\phi_1^{c,\mu}$ is the flow of $v_t$
    at time 1, $\bar{q}$ is the source shape being deformed, $q$ is the target shape,
    and $\alpha$ is a regularization term that controls the tradeoff between exact matching and
    smoothness of the deformation. $d$ is a distance function on shapes (point-to-point L2,
    varifold, metric, etc).

    Control points can be passed as parameters or are initialized on a grid that contains the
    source shapes. They are optimized if `freeze_control_points` is set to false.

    Resulting control points and momenta are saved in the ouput dir as txt files. Control points
    are also saved with attached momenta as a vtk file to allow visualization with paraview.

    Parameters
    ----------
    source: str or pathlib.Path
        Path to the vtk file that contains the source mesh.
    target: str or pathlib.Path
        Path to the vtk file that contains the target mesh.
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
    use_rk4_for_shoot: bool
        Whether to use Runge-Kutta order 4 steps in the integration of the Hamiltonian equation that
        governs the time evolution of control points and momenta. Overrides use_rk2_for_shoot.
        RK4 steps are required when estimating a geodesic that will be used for parallel transport.
        Optional, default: False
    print_every: int
        Sets the verbosity level of the optimization scheme.
    threshold: float
        Threshold to use on momenta norm when filtering. Ignored if `filter_cp` is set to `False`.
    max_iter: int
        Maximum number of iteration in the optimization scheme.
        Optional, default: 200.
    tol: float
        Tolerance to evaluate convergence.
    """
    optimization_parameters = {
        "max_iterations": max_iter,
        "freeze_template": False,
        "freeze_control_points": freeze_control_points,
        "freeze_momenta": False,
        "use_sobolev_gradient": True,
        "sobolev_kernel_width_ratio": 1,
        "initial_control_points": initial_control_points,
        "initial_cp_spacing": None,
        "initial_momenta": None,
        "dense_mode": False,
        "number_of_threads": 1,
        "print_every_n_iters": print_every,
        "downsampling_factor": 1,
        "dimension": 3,
        "optimization_method_type": "ScipyLBFGS",
    }

    deformetrica = Deformetrica(output_dir, verbosity=verbosity)

    model_options = {
        "deformation_kernel_type": kernel_type,
        "deformation_kernel_width": kernel_width,
        "deformation_kernel_device": kernel_device,
        "use_svf": use_svf,
        "preserve_volume": preserve_volume,
        "number_of_time_points": number_of_time_steps,
        "use_rk2_for_shoot": use_rk2_for_shoot,
        "use_rk4_for_shoot": use_rk4_for_shoot,
        "use_rk2_for_flow": use_rk2_for_flow,
        "freeze_template": False,
        "freeze_control_points": freeze_control_points,
        "initial_control_points": initial_control_points,
        "dimension": dimension,
        "output_dir": output_dir,
    }

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
        "visit_ages": [[]],
        "dataset_filenames": [[{"shape": target}]],
        "subject_ids": [target_id],
    }

    deformetrica.estimate_registration(
        template_specifications=template,
        dataset_specifications=data_set,
        model_options=model_options,
        estimator_options=optimization_parameters,
    )
