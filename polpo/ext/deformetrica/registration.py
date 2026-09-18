"""Deformetrica registration utilities."""

from api.deformetrica import Deformetrica


def _make_estimator_options(
    max_iter,
    freeze_control_points,
    initial_control_points,
    print_every,
    dimension,
):
    return {
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
        "dimension": dimension,
        "optimization_method_type": "ScipyLBFGS",
    }


def _make_model_options(
    output_dir,
    kernel_width,
    kernel_type,
    kernel_device,
    use_svf,
    preserve_volume,
    number_of_time_steps,
    use_rk2_for_shoot,
    use_rk4_for_shoot,
    use_rk2_for_flow,
    freeze_control_points,
    initial_control_points,
    dimension,
):
    return {
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


def _make_template_specifications(
    source,
    metric,
    kernel_width,
    kernel_type,
    kernel_device,
    regularisation,
):
    return {
        "shape": {
            "deformable_object_type": "SurfaceMesh",
            "kernel_type": kernel_type,
            "kernel_width": kernel_width,
            "kernel_device": kernel_device,
            "noise_std": regularisation,
            "filename": source,
            "noise_variance_prior_scale_std": None,
            "noise_variance_prior_normalized_dof": 0.01,
            "attachment_type": metric,
        }
    }


def _make_dataset_specifications(target, target_id):
    return {
        "visit_ages": [[]],
        "dataset_filenames": [[{"shape": target}]],
        "subject_ids": [target_id],
    }


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
    """Estimate a registration between two surfaces."""
    estimator_options = _make_estimator_options(
        max_iter=max_iter,
        freeze_control_points=freeze_control_points,
        initial_control_points=initial_control_points,
        print_every=print_every,
        dimension=dimension,
    )

    model_options = _make_model_options(
        output_dir=output_dir,
        kernel_width=kernel_width,
        kernel_type=kernel_type,
        kernel_device=kernel_device,
        use_svf=use_svf,
        preserve_volume=preserve_volume,
        number_of_time_steps=number_of_time_steps,
        use_rk2_for_shoot=use_rk2_for_shoot,
        use_rk4_for_shoot=use_rk4_for_shoot,
        use_rk2_for_flow=use_rk2_for_flow,
        freeze_control_points=freeze_control_points,
        initial_control_points=initial_control_points,
        dimension=dimension,
    )

    template_specifications = _make_template_specifications(
        source=source,
        metric=metric,
        kernel_width=attachment_kernel_width,
        kernel_type=kernel_type,
        kernel_device=kernel_device,
        regularisation=regularisation,
    )

    dataset_specifications = _make_dataset_specifications(
        target=target,
        target_id=target_id,
    )

    deformetrica = Deformetrica(
        output_dir,
        verbosity=verbosity,
    )

    deformetrica.estimate_registration(
        template_specifications=template_specifications,
        dataset_specifications=dataset_specifications,
        model_options=model_options,
        estimator_options=estimator_options,
    )
