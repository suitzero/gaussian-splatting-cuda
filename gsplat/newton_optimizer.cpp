// src/newton_optimizer.cpp
#include "core/newton_optimizer.hpp" 
#include "newton_kernels.cuh" // Path relative to include paths, or needs adjustment
#include "core/torch_utils.hpp" // Assuming torch_utils is still in core/
#include <iostream> // For std::cout debug prints

// Constructor
NewtonOptimizer::NewtonOptimizer(SplatData& splat_data,
                                 const gs::param::OptimizationParameters& opt_params,
                                 Options options)
    : model_(splat_data), opt_params_ref_(opt_params), options_(options) {
    // TODO: Initialization if needed, e.g. pre-allocate tensors if sizes are fixed
}

// --- Loss Derivatives ---
NewtonOptimizer::LossDerivatives NewtonOptimizer::compute_loss_derivatives_cuda(
    const torch::Tensor& rendered_image,
    const torch::Tensor& gt_image,
    float lambda_dssim,
    bool use_l2_loss_term) {

    TORCH_CHECK(rendered_image.device().is_cuda(), "rendered_image must be a CUDA tensor");
    TORCH_CHECK(gt_image.device().is_cuda(), "gt_image must be a CUDA tensor");
    TORCH_CHECK(rendered_image.sizes() == gt_image.sizes(), "rendered_image and gt_image must have the same size");
    TORCH_CHECK(rendered_image.dim() == 3 && rendered_image.size(2) == 3, "Images must be HxWx3, got ", rendered_image.sizes());

    auto tensor_options = torch::TensorOptions().device(rendered_image.device()).dtype(rendered_image.dtype());
    torch::Tensor dL_dc = torch::zeros_like(rendered_image, tensor_options);
    torch::Tensor d2L_dc2_diag = torch::zeros_like(rendered_image, tensor_options);

    NewtonKernels::compute_loss_derivatives_kernel_launcher(
        rendered_image, gt_image, lambda_dssim, use_l2_loss_term,
        dL_dc, d2L_dc2_diag
    );

    return {dL_dc, d2L_dc2_diag};
}

// --- Position (Means) ---
// Computes only the Hessian components for position. Gradient comes from autograd.
NewtonOptimizer::PositionHessianOutput NewtonOptimizer::compute_position_hessian_components_cuda(
    const SplatData& model_snapshot,
    const torch::Tensor& visibility_mask_for_model,
    const Camera& camera,
    const gs::RenderOutput& render_output,
    const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian, // Directly passed
    int num_visible_gaussians_in_total_model
) {
    // Use const getter for SplatData when model_snapshot is const
    torch::Tensor means_tensor = model_snapshot.get_means();
    auto dev = means_tensor.device();
    auto dtype = means_tensor.dtype();
    auto tensor_opts = torch::TensorOptions().device(dev).dtype(dtype);

    // Output tensors for the *num_visible_gaussians_in_total_model*
    torch::Tensor H_p_output_packed = torch::zeros({num_visible_gaussians_in_total_model, 6}, tensor_opts);
    // grad_p_output is removed from here, it will come from autograd_grad_means

    // Prepare camera parameters
    torch::Tensor view_mat_tensor_orig = camera.world_view_transform().to(dev).to(dtype);
    torch::Tensor view_mat_tensor = view_mat_tensor_orig.contiguous();
    torch::Tensor K_matrix = camera.K().to(dev).to(dtype).contiguous();

    // Compute camera center C_w = -R_wc^T * t_wc from world_view_transform V = [R_wc | t_wc]
    // Assuming V is [4,4] world-to-camera.
    torch::Tensor view_mat_2d = view_mat_tensor.select(0, 0); // Get [4,4] matrix assuming batch size is 1
    torch::Tensor R_wc_2d = view_mat_2d.slice(0, 0, 3).slice(1, 0, 3); // [3,3]
    torch::Tensor t_wc_2d = view_mat_2d.slice(0, 0, 3).slice(1, 3, 4); // [3,1]
    torch::Tensor R_wc = R_wc_2d.unsqueeze(0); // [1,3,3]
    torch::Tensor t_wc = t_wc_2d.unsqueeze(0); // [1,3,1]

    // Debug prints for shapes and strides
    if (options_.debug_print_shapes) {
        std::cout << "[DEBUG] compute_pos_hess: R_wc_T shape: " << R_wc.transpose(-2,-1).sizes()
                  << " strides: " << R_wc.transpose(-2,-1).strides()
                  << " contiguous: " << R_wc.transpose(-2,-1).is_contiguous() << std::endl;
        std::cout << "[DEBUG] compute_pos_hess: t_wc shape: " << t_wc.sizes()
                  << " strides: " << t_wc.strides()
                  << " contiguous: " << t_wc.is_contiguous() << std::endl;
        std::cout << "[DEBUG] compute_pos_hess: t_wc.contiguous() shape: " << t_wc.contiguous().sizes()
                  << " strides: " << t_wc.contiguous().strides()
                  << " contiguous: " << t_wc.contiguous().is_contiguous() << std::endl;

        // Dtype and Contiguity checks
        auto print_tensor_info = [&](const std::string& name, const torch::Tensor& tensor) {
            if (!tensor.defined()) {
                std::cout << "[DEBUG] INFO_CHECK " << name << ": UNDEFINED" << std::endl;
                return;
            }
            std::cout << "[DEBUG] INFO_CHECK " << name << ": dtype=" << tensor.scalar_type()
                      << ", contiguous=" << tensor.is_contiguous()
                      << ", shape=" << tensor.sizes() << std::endl;
        };

        print_tensor_info("model_snapshot.get_means()", model_snapshot.get_means());
        print_tensor_info("model_snapshot.get_scaling()", model_snapshot.get_scaling());
        print_tensor_info("model_snapshot.get_rotation()", model_snapshot.get_rotation());
        print_tensor_info("model_snapshot.get_opacity()", model_snapshot.get_opacity());
        print_tensor_info("model_snapshot.get_shs()", model_snapshot.get_shs());
        print_tensor_info("view_mat_tensor", view_mat_tensor);
        print_tensor_info("K_matrix", K_matrix);
        // cam_pos_tensor check will be after its definition
        print_tensor_info("render_output.means2d", render_output.means2d);
        print_tensor_info("render_output.depths", render_output.depths);
        print_tensor_info("render_output.radii (original)", render_output.radii); // Check original radii
        // print_tensor_info("loss_derivs.dL_dc", loss_derivs.dL_dc); // dL_dc comes from autograd
        print_tensor_info("d2L_dc2_diag_pixelwise_for_hessian", d2L_dc2_diag_pixelwise_for_hessian);
        print_tensor_info("H_p_output_packed", H_p_output_packed);
        // print_tensor_info("grad_p_output", grad_p_output); // grad_p comes from autograd
        // visibility_mask_for_model is bool, get_const_data_ptr<bool> will check its contiguity.
        print_tensor_info("visibility_mask_for_model", visibility_mask_for_model);
    }

    // Transpose the inner two dimensions for matrix transpose, robust to batches.
    torch::Tensor cam_pos_tensor = -torch::matmul(R_wc.transpose(-2, -1), t_wc.contiguous()).squeeze();
    if (cam_pos_tensor.dim() > 1) cam_pos_tensor = cam_pos_tensor.squeeze(); // Ensure it's [3] or [B,3]


    // The kernel needs to map RenderOutput's culled set of Gaussians (means2d, depths, radii)
    // back to the original model's Gaussians, or use the visibility_mask_for_model.
    // This is a complex part of the kernel design.
    // For now, we pass what we have. The kernel must be robust.
    // `render_output.visibility_indices` could be a map from render_output's internal indexing to original model indices.
    // `render_output.visibility_filter` could be a boolean mask on the *culled* set from rasterizer.

    if (options_.debug_print_shapes) {
        // Check cam_pos_tensor after its definition and potential squeeze
        std::cout << "[DEBUG] INFO_CHECK cam_pos_tensor: dtype=" << cam_pos_tensor.scalar_type()
                  << ", contiguous=" << cam_pos_tensor.is_contiguous()
                  << ", shape=" << cam_pos_tensor.sizes() << std::endl;
    }

    // Handle render_output.radii dtype
    torch::Tensor radii_for_kernel_tensor;
    if (render_output.radii.defined()) {
        if (render_output.radii.scalar_type() != torch::kFloat) {
            if(options_.debug_print_shapes) { // Also print if we are recasting
                 std::cout << "[DEBUG] Recasting render_output.radii from " << render_output.radii.scalar_type() << " to Float." << std::endl;
            }
            radii_for_kernel_tensor = render_output.radii.to(torch::kFloat);
        } else {
            radii_for_kernel_tensor = render_output.radii;
        }
    }
    // If render_output.radii was undefined, radii_for_kernel_tensor remains undefined.
    // get_const_data_ptr will handle undefined tensor by returning nullptr.

    // Moved p_total_for_kernel definition and try-catch block earlier
    int p_total_for_kernel = 0;
    try {
        // This directly calls model_snapshot.size() which is model_snapshot._means.size(0)
        // Ensure means is defined before trying to access its size for p_total_for_kernel
        if (model_snapshot.get_means().defined()) {
            p_total_for_kernel = static_cast<int>(model_snapshot.size());
            if (options_.debug_print_shapes) {
                std::cout << "[NewtonOpt POS_HESS] model_snapshot.size() call successful. P_total_for_kernel = " << p_total_for_kernel << std::endl;
            }
        } else {
            if (options_.debug_print_shapes) {
                std::cout << "[NewtonOpt POS_HESS] model_snapshot.get_means() is UNDEFINED. Cannot call .size(). Setting P_total_for_kernel to 0." << std::endl;
            }
            // p_total_for_kernel remains 0, or handle as an error
        }
    } catch (const c10::Error& e) {
        if (options_.debug_print_shapes) {
            std::cout << "[NewtonOpt POS_HESS] model_snapshot.size() call FAILED (c10::Error): " << e.what_without_backtrace() << std::endl;
        }
        throw;
    } catch (const std::exception& e) {
        if (options_.debug_print_shapes) {
            std::cout << "[NewtonOpt POS_HESS] model_snapshot.size() call FAILED (std::exception): " << e.what() << std::endl;
        }
        throw;
    } catch (...) {
        if (options_.debug_print_shapes) {
            std::cout << "[NewtonOpt POS_HESS] model_snapshot.size() call FAILED (unknown exception)." << std::endl;
        }
        throw;
    }

    // Define the verbose checker lambda (only if debug_print_shapes is on)
    std::function<void(const std::string&, const torch::Tensor&, const std::string&)> verbose_tensor_check_lambda;
    if (options_.debug_print_shapes) {
        verbose_tensor_check_lambda =
            [](const std::string& name, const torch::Tensor& tensor, const std::string& expected_type_str) {
            std::cout << "[VERBOSE_CHECK] Tensor: " << name;
            if (!tensor.defined()) {
                std::cout << " - UNDEFINED" << std::endl;
                return;
            }
            std::cout << " - Device: " << tensor.device()
                      << ", Dtype: " << tensor.scalar_type() << " (Exp: " << expected_type_str << ")"
                      << ", Contig: " << tensor.is_contiguous()
                      << ", Sizes: " << tensor.sizes()
                      << ", Numel: " << tensor.numel();
            try {
                if (tensor.numel() > 0) {
                    // Attempt to access data_ptr
                    if (expected_type_str == "float") tensor.data_ptr<float>();
                    else if (expected_type_str == "bool") tensor.data_ptr<bool>();
                    else if (expected_type_str == "int") tensor.data_ptr<int>();
                    // Add other types if needed
                    std::cout << ", data_ptr<" << expected_type_str << ">: OK";
                } else {
                    std::cout << ", data_ptr<" << expected_type_str << ">: Skipped (empty)";
                }
            } catch (const c10::Error& e) {
                std::cout << ", data_ptr<" << expected_type_str << ">: FAILED (c10::Error): " << e.what_without_backtrace();
            } catch (const std::exception& e) {
                std::cout << ", data_ptr<" << expected_type_str << ">: FAILED (std::exception): " << e.what();
            } catch (...) {
                std::cout << ", data_ptr<" << expected_type_str << ">: FAILED (unknown exception)";
            }
        };

        // Call for existing INFO_CHECK tensors (now done by the lambda)
        verbose_tensor_check_lambda("model_snapshot.get_means()", model_snapshot.get_means(), "float");
        verbose_tensor_check_lambda("model_snapshot.get_scaling()", model_snapshot.get_scaling(), "float");
        verbose_tensor_check_lambda("model_snapshot.get_rotation()", model_snapshot.get_rotation(), "float");
        verbose_tensor_check_lambda("model_snapshot.get_opacity()", model_snapshot.get_opacity(), "float");
        verbose_tensor_check_lambda("model_snapshot.get_shs()", model_snapshot.get_shs(), "float");
        verbose_tensor_check_lambda("view_mat_tensor", view_mat_tensor, "float");
        verbose_tensor_check_lambda("K_matrix", K_matrix, "float");
        verbose_tensor_check_lambda("cam_pos_tensor", cam_pos_tensor, "float");
        verbose_tensor_check_lambda("render_output.means2d", render_output.means2d, "float");
        verbose_tensor_check_lambda("render_output.depths", render_output.depths, "float");
        verbose_tensor_check_lambda("radii_for_kernel_tensor", radii_for_kernel_tensor, "float"); // After potential cast
        verbose_tensor_check_lambda("visibility_mask_for_model", visibility_mask_for_model, "bool");
        // verbose_tensor_check_lambda("loss_derivs.dL_dc", loss_derivs.dL_dc, "float"); // dL_dc comes from autograd
        verbose_tensor_check_lambda("d2L_dc2_diag_pixelwise_for_hessian", d2L_dc2_diag_pixelwise_for_hessian, "float");
        verbose_tensor_check_lambda("H_p_output_packed", H_p_output_packed, "float");
        // verbose_tensor_check_lambda("grad_p_output", grad_p_output, "float"); // grad_p comes from autograd
    }

    // Prepare arguments for kernel launcher by getting data pointers
    const torch::Tensor& arg_means3D = model_snapshot.get_means();
    const torch::Tensor& arg_scales = model_snapshot.get_scaling();
    const torch::Tensor& arg_rotations = model_snapshot.get_rotation();
    const torch::Tensor& arg_opacities = model_snapshot.get_opacity();
    const torch::Tensor& arg_shs = model_snapshot.get_shs();
    // view_mat_tensor, K_matrix, cam_pos_tensor, radii_for_kernel_tensor, visibility_mask_for_model,
    // loss_derivs.dL_dc, loss_derivs.d2L_dc2_diag, H_p_output_packed, grad_p_output are already local variables.

    if (options_.debug_print_shapes) {
        // verbose_tensor_check_lambda was defined earlier if options_.debug_print_shapes is true
        verbose_tensor_check_lambda("arg_means3D", arg_means3D, "float");
        verbose_tensor_check_lambda("arg_scales", arg_scales, "float");
        verbose_tensor_check_lambda("arg_rotations", arg_rotations, "float");
        verbose_tensor_check_lambda("arg_opacities", arg_opacities, "float");
        verbose_tensor_check_lambda("arg_shs", arg_shs, "float");
        verbose_tensor_check_lambda("view_mat_tensor", view_mat_tensor, "float"); // Already local
        verbose_tensor_check_lambda("K_matrix", K_matrix, "float"); // Already local
        verbose_tensor_check_lambda("cam_pos_tensor", cam_pos_tensor, "float"); // Already local
        verbose_tensor_check_lambda("render_output.means2d", render_output.means2d, "float");
        verbose_tensor_check_lambda("render_output.depths", render_output.depths, "float");
        verbose_tensor_check_lambda("radii_for_kernel_tensor", radii_for_kernel_tensor, "float"); // Already local
        verbose_tensor_check_lambda("visibility_mask_for_model", visibility_mask_for_model, "bool"); // Already local
        // verbose_tensor_check_lambda("loss_derivs.dL_dc", loss_derivs.dL_dc, "float");
        verbose_tensor_check_lambda("d2L_dc2_diag_pixelwise_for_hessian", d2L_dc2_diag_pixelwise_for_hessian, "float");
        verbose_tensor_check_lambda("H_p_output_packed", H_p_output_packed, "float"); // Already local
        // verbose_tensor_check_lambda("grad_p_output", grad_p_output, "float");
    }

    const float* means_3d_all_ptr = gs::torch_utils::get_const_data_ptr<float>(arg_means3D, "arg_means3D");
    const float* scales_all_ptr = gs::torch_utils::get_const_data_ptr<float>(arg_scales, "arg_scales");
    const float* rotations_all_ptr = gs::torch_utils::get_const_data_ptr<float>(arg_rotations, "arg_rotations");
    const float* opacities_all_ptr = gs::torch_utils::get_const_data_ptr<float>(arg_opacities, "arg_opacities");
    const float* shs_all_ptr = gs::torch_utils::get_const_data_ptr<float>(arg_shs, "arg_shs");
    const float* view_matrix_ptr = gs::torch_utils::get_const_data_ptr<float>(view_mat_tensor, "view_mat_tensor");
    const float* K_matrix_ptr = gs::torch_utils::get_const_data_ptr<float>(K_matrix, "K_matrix");
    const float* cam_pos_world_ptr = gs::torch_utils::get_const_data_ptr<float>(cam_pos_tensor, "cam_pos_tensor");
    const float* means_2d_render_ptr = gs::torch_utils::get_const_data_ptr<float>(render_output.means2d, "render_output.means2d");
    const float* depths_render_ptr = gs::torch_utils::get_const_data_ptr<float>(render_output.depths, "render_output.depths");
    const float* radii_render_ptr = gs::torch_utils::get_const_data_ptr<float>(radii_for_kernel_tensor, "radii_for_kernel_tensor");
    // No longer need visibility_mask_for_model_ptr here, pass the tensor directly
    // const float* dL_dc_pixelwise_ptr = gs::torch_utils::get_const_data_ptr<float>(loss_derivs.dL_dc, "loss_derivs.dL_dc"); // Removed
    const float* d2L_dc2_diag_pixelwise_for_hessian_ptr = gs::torch_utils::get_const_data_ptr<float>(d2L_dc2_diag_pixelwise_for_hessian, "d2L_dc2_diag_pixelwise_for_hessian");
    float* H_p_output_packed_ptr = gs::torch_utils::get_data_ptr<float>(H_p_output_packed, "H_p_output_packed");
    // float* grad_p_output_ptr = gs::torch_utils::get_data_ptr<float>(grad_p_output, "grad_p_output"); // Removed

    NewtonKernels::compute_position_hessian_components_kernel_launcher(
        render_output.height, render_output.width, static_cast<int>(d2L_dc2_diag_pixelwise_for_hessian.size(-1)), // Image: H, W, C
        p_total_for_kernel, // Total P Gaussians in model
        means_3d_all_ptr,
        scales_all_ptr,
        rotations_all_ptr,
        opacities_all_ptr,
        shs_all_ptr,
        model_snapshot.get_active_sh_degree(),
        static_cast<int>(model_snapshot.get_shs().size(1)), // sh_coeffs_dim (sh_degree+1)^2
        view_matrix_ptr, // world-to-camera view matrix
        K_matrix_ptr,    // camera projection matrix (intrinsic K, possibly combined with extrinsics for full P) - named projection_matrix_for_jacobian in kernel
        cam_pos_world_ptr,
        visibility_mask_for_model, // Pass the tensor object directly
        // dL_dc_pixelwise_ptr, // Removed
        d2L_dc2_diag_pixelwise_for_hessian_ptr, // Use new pointer
        num_visible_gaussians_in_total_model, // Number of Gaussians to produce output for (visible and in model)
        H_p_output_packed_ptr,
        // grad_p_output_ptr, // Removed
        options_.debug_print_shapes // Pass the flag
    );

    return {H_p_output_packed}; // grad_p_output removed from struct
}

torch::Tensor NewtonOptimizer::compute_projected_position_hessian_and_gradient(
    const torch::Tensor& H_p_packed,
    const torch::Tensor& grad_p,
    const torch::Tensor& means_3d_visible,
    const Camera& camera,
    torch::Tensor& out_grad_v
) {
    TORCH_CHECK(H_p_packed.device().is_cuda() && grad_p.device().is_cuda() &&
                means_3d_visible.device().is_cuda() && out_grad_v.device().is_cuda(),
                "All tensors for projection must be CUDA tensors");

    int num_visible_gaussians = H_p_packed.size(0);
    TORCH_CHECK(grad_p.size(0) == num_visible_gaussians &&
                means_3d_visible.size(0) == num_visible_gaussians &&
                out_grad_v.size(0) == num_visible_gaussians, "Size mismatch in projection inputs/outputs");

    auto tensor_opts = H_p_packed.options();
    torch::Tensor H_v_packed = torch::zeros({num_visible_gaussians, 3}, tensor_opts); // 3 for symmetric 2x2

    torch::Tensor view_mat_tensor_orig = camera.world_view_transform().to(tensor_opts.device());
    torch::Tensor view_mat_tensor = view_mat_tensor_orig.contiguous();
    // Compute camera center C_w = -R_wc^T * t_wc
    torch::Tensor view_mat_2d_proj = view_mat_tensor.select(0, 0); // Assuming batch size is 1
    torch::Tensor R_wc_2d_proj = view_mat_2d_proj.slice(0, 0, 3).slice(1, 0, 3);
    torch::Tensor t_wc_2d_proj = view_mat_2d_proj.slice(0, 0, 3).slice(1, 3, 4);
    torch::Tensor R_wc_proj = R_wc_2d_proj.unsqueeze(0);
    torch::Tensor t_wc_proj = t_wc_2d_proj.unsqueeze(0);

    if (options_.debug_print_shapes) {
        std::cout << "[DEBUG] compute_proj_hess_grad: R_wc_proj_T shape: " << R_wc_proj.transpose(-2,-1).sizes()
                  << " strides: " << R_wc_proj.transpose(-2,-1).strides()
                  << " contiguous: " << R_wc_proj.transpose(-2,-1).is_contiguous() << std::endl;
        std::cout << "[DEBUG] compute_proj_hess_grad: t_wc_proj shape: " << t_wc_proj.sizes()
                  << " strides: " << t_wc_proj.strides()
                  << " contiguous: " << t_wc_proj.is_contiguous() << std::endl;
        std::cout << "[DEBUG] compute_proj_hess_grad: t_wc_proj.contiguous() shape: " << t_wc_proj.contiguous().sizes()
                  << " strides: " << t_wc_proj.contiguous().strides()
                  << " contiguous: " << t_wc_proj.contiguous().is_contiguous() << std::endl;
    }

    // Transpose the inner two dimensions for matrix transpose, robust to batches.
    torch::Tensor cam_pos_tensor = -torch::matmul(R_wc_proj.transpose(-2, -1), t_wc_proj.contiguous()).squeeze();
    if (cam_pos_tensor.dim() > 1) cam_pos_tensor = cam_pos_tensor.squeeze();
    cam_pos_tensor = cam_pos_tensor.to(tensor_opts.device());


    NewtonKernels::project_position_hessian_gradient_kernel_launcher(
        num_visible_gaussians,
        gs::torch_utils::get_const_data_ptr<float>(H_p_packed),
        gs::torch_utils::get_const_data_ptr<float>(grad_p),
        gs::torch_utils::get_const_data_ptr<float>(means_3d_visible),
        gs::torch_utils::get_const_data_ptr<float>(view_mat_tensor),
        gs::torch_utils::get_const_data_ptr<float>(cam_pos_tensor),
        gs::torch_utils::get_data_ptr<float>(H_v_packed),
        gs::torch_utils::get_data_ptr<float>(out_grad_v)
    );
    return H_v_packed;
}

torch::Tensor NewtonOptimizer::solve_and_project_position_updates(
    const torch::Tensor& H_v_projected_packed, // [N_vis, 3]
    const torch::Tensor& grad_v_projected,     // [N_vis, 2]
    const torch::Tensor& means_3d_visible,     // [N_vis, 3]
    const Camera& camera,
    double damping,
    double step_scale
) {
    int num_visible_gaussians = H_v_projected_packed.size(0);
    auto tensor_opts = H_v_projected_packed.options();

    torch::Tensor delta_v = torch::zeros({num_visible_gaussians, 2}, tensor_opts);
    NewtonKernels::batch_solve_2x2_system_kernel_launcher(
        num_visible_gaussians,
        gs::torch_utils::get_const_data_ptr<float>(H_v_projected_packed),
        gs::torch_utils::get_const_data_ptr<float>(grad_v_projected),
        static_cast<float>(damping),
        static_cast<float>(step_scale), // step_scale applied inside kernel
        gs::torch_utils::get_data_ptr<float>(delta_v)
    );

    torch::Tensor delta_p = torch::zeros({num_visible_gaussians, 3}, tensor_opts);
    torch::Tensor view_mat_tensor_orig = camera.world_view_transform().to(tensor_opts.device());
    torch::Tensor view_mat_tensor = view_mat_tensor_orig.contiguous();
    // Compute camera center C_w = -R_wc^T * t_wc
    torch::Tensor view_mat_2d_solve = view_mat_tensor.select(0, 0); // Assuming batch size is 1
    torch::Tensor R_wc_2d_solve = view_mat_2d_solve.slice(0, 0, 3).slice(1, 0, 3);
    torch::Tensor t_wc_2d_solve = view_mat_2d_solve.slice(0, 0, 3).slice(1, 3, 4);
    torch::Tensor R_wc_solve = R_wc_2d_solve.unsqueeze(0);
    torch::Tensor t_wc_solve = t_wc_2d_solve.unsqueeze(0);

    if (options_.debug_print_shapes) {
        std::cout << "[DEBUG] solve_and_proj: R_wc_solve_T shape: " << R_wc_solve.transpose(-2,-1).sizes()
                  << " strides: " << R_wc_solve.transpose(-2,-1).strides()
                  << " contiguous: " << R_wc_solve.transpose(-2,-1).is_contiguous() << std::endl;
        std::cout << "[DEBUG] solve_and_proj: t_wc_solve shape: " << t_wc_solve.sizes()
                  << " strides: " << t_wc_solve.strides()
                  << " contiguous: " << t_wc_solve.is_contiguous() << std::endl;
        std::cout << "[DEBUG] solve_and_proj: t_wc_solve.contiguous() shape: " << t_wc_solve.contiguous().sizes()
                  << " strides: " << t_wc_solve.contiguous().strides()
                  << " contiguous: " << t_wc_solve.contiguous().is_contiguous() << std::endl;
    }

    // Transpose the inner two dimensions for matrix transpose, robust to batches.
    torch::Tensor cam_pos_tensor = -torch::matmul(R_wc_solve.transpose(-2, -1), t_wc_solve.contiguous()).squeeze();
    if (cam_pos_tensor.dim() > 1) cam_pos_tensor = cam_pos_tensor.squeeze();
    cam_pos_tensor = cam_pos_tensor.to(tensor_opts.device());

    NewtonKernels::project_update_to_3d_kernel_launcher(
        num_visible_gaussians,
        gs::torch_utils::get_const_data_ptr<float>(delta_v),
        gs::torch_utils::get_const_data_ptr<float>(means_3d_visible),
        gs::torch_utils::get_const_data_ptr<float>(view_mat_tensor),
        gs::torch_utils::get_const_data_ptr<float>(cam_pos_tensor),
        gs::torch_utils::get_data_ptr<float>(delta_p)
    );
    return delta_p;
}


// Main step function
void NewtonOptimizer::step(int iteration,
                           const torch::Tensor& visibility_mask_for_model, // Boolean mask for model_.means() [N_total]
                           const torch::Tensor& autograd_grad_means_total,       // [N_total, 3]
                           const torch::Tensor& autograd_grad_scales_raw_total,  // [N_total, 3]
                           const torch::Tensor& autograd_grad_rotation_raw_total, // [N_total, 4]
                           const torch::Tensor& autograd_grad_opacity_raw_total, // [N_total, 1]
                           const torch::Tensor& autograd_grad_sh0_total,         // [N_total, 1, 3]
                           const torch::Tensor& autograd_grad_shN_total,         // [N_total, K-1, 3]
                           const gs::RenderOutput& current_render_output,
                           const Camera& primary_camera,
                           const torch::Tensor& primary_gt_image,
                           const std::vector<std::pair<const Camera*, torch::Tensor>>& knn_secondary_targets_data) {

    torch::NoGradGuard no_grad; // Ensure no graph operations are tracked for optimizer steps

    torch::Tensor visible_indices = torch::where(visibility_mask_for_model)[0];
    int num_visible_gaussians_in_model = visible_indices.size(0);

    if (options_.debug_print_shapes) {
        torch::Tensor visibility_sum_tensor = visibility_mask_for_model.sum();
        long visibility_sum = visibility_sum_tensor.defined() ? visibility_sum_tensor.item<int64_t>() : -1L;
        std::cout << "[NewtonOpt] Step - Iteration: " << iteration
                  << ", num_visible_gaussians_in_model (from mask): " << num_visible_gaussians_in_model
                  << ", visibility_mask_for_model sum: " << visibility_sum
                  << std::endl;
    }

    if (num_visible_gaussians_in_model == 0) {
        if (options_.debug_print_shapes) {
             std::cout << "[NewtonOpt] Step: No visible Gaussians based on mask at iteration " << iteration << ". Skipping Newton update." << std::endl;
        }
        return; // Early exit if no Gaussians are visible
    }

    // I. Compute d2L/dc2 for primary target. This is needed for all attribute Hessians.
    // dL/dc for the primary target (used for SH gradient) will also come from this.
    // Autograd gradients are used for means, scales, rotations, opacities.
    torch::Tensor rendered_image_squeezed = current_render_output.image.squeeze(0);
    torch::Tensor gt_image_prepared = primary_gt_image;

    // Ensure rendered_image is HWC
    if (rendered_image_squeezed.dim() == 3 && rendered_image_squeezed.size(0) == 3) {
        // Input is CHW [3, H, W], permute to HWC [H, W, 3]
        rendered_image_squeezed = rendered_image_squeezed.permute({1, 2, 0}).contiguous();
    }

    // Ensure gt_image is HWC to match rendered_image for the checks inside compute_loss_derivatives_cuda
    // and for consistency if the kernel itself expects HWC for both.
    if (gt_image_prepared.dim() == 3 && gt_image_prepared.size(0) == 3) {
        // Input is CHW [3, H, W], permute to HWC [H, W, 3]
        gt_image_prepared = gt_image_prepared.permute({1, 2, 0}).contiguous();
    }

    LossDerivatives primary_loss_derivs = compute_loss_derivatives_cuda(
        rendered_image_squeezed,
        gt_image_prepared,
        options_.lambda_dssim_for_hessian,
        options_.use_l2_for_hessian_L_term
    );

    // Only proceed with means optimization if enabled
    if (options_.optimize_means) {
        torch::Tensor means_visible_from_model = model_.means().detach().index_select(0, visible_indices);
        torch::Tensor grad_means_visible_autograd = autograd_grad_means_total.index_select(0, visible_indices);

    // II. Compute Hessian components (H_p) for primary target. Gradient g_p comes from autograd.
        if (options_.debug_print_shapes) {
            std::cout << "[NewtonOpt STEP] Checking this->model_ BEFORE call to compute_position_hessian_components_cuda:" << std::endl;
            // ... (debug prints for model state can remain)
        }
        PositionHessianOutput primary_hess_output = compute_position_hessian_components_cuda(
            model_,
            visibility_mask_for_model,
            primary_camera,
            current_render_output,
            primary_loss_derivs.d2L_dc2_diag, // Pass only the d2L/dc2 part
            num_visible_gaussians_in_model
        );

        torch::Tensor H_p_total_packed = primary_hess_output.H_p_packed.clone(); // [N_vis_model, 6]
        // Gradient g_p_total_visible now comes from autograd_grad_means
        torch::Tensor g_p_total_visible = grad_means_visible_autograd.clone(); // [N_vis_model, 3]


    // III. Handle Secondary Targets for Overshoot Prevention (primarily for Hessian accumulation)
        if (options_.knn_k > 0 && !knn_secondary_targets_data.empty()) {
        for (const auto& knn_data : knn_secondary_targets_data) {
            const Camera* secondary_camera = knn_data.first;
            const torch::Tensor& secondary_gt_image = knn_data.second; // Assumed [H,W,C] on device

            // Render this secondary view (simplified: actual render call needed)
            // gs::RenderOutput secondary_render_output = gs::rasterize(...);
            // For now, let's assume we have a placeholder or skip actual rendering for secondary targets
            // to avoid making this step too complex with re-rendering.
            // The paper says "Hessians and gradients of secondary targets are sparsely evaluated".
            // This implies a simplified rendering/evaluation for them.
            // Let's assume, for now, we reuse primary_render_output's structure but with secondary camera and GT.
            // This is a simplification! A proper implementation needs to render secondary views.
            if (options_.debug_print_shapes) {
                std::cout << "[NewtonOpt KNN] Processing secondary target for camera UID (if available): "
                          << (secondary_camera ? std::to_string(secondary_camera->uid()) : "N/A") << std::endl;
            }

            // 1. Define background color for secondary render (e.g., black or gray)
            //    Using a default black background for secondary targets for now.
            torch::Tensor secondary_bg_color = torch::tensor({0.0f, 0.0f, 0.0f}, model_.get_means().options());

            // 2. Render secondary view
            //    Ensure camera parameters are on the correct device for rasterize if not already.
            //    The `rasterize` function takes Camera&, implying it might modify it or expect it to be mutable for some reason (e.g. update matrices).
            //    However, our secondary_camera is const. This might require a const_cast or adjustment in rasterize,
            //    or rasterize only needs const access to camera properties it uses. For now, assume rasterize can handle const Camera& effectively or uses a copy.
            //    Let's make a copy of the camera object to be safe if rasterize needs non-const, though it's not ideal.
            //    A better solution would be for rasterize to take const Camera& if it doesn't modify it.
            //    For now, we proceed assuming rasterize is safe with a const Camera passed by value or that its non-const methods are not called.
            //    The Camera object itself does not store CUDA tensors that are modified by rasterize.

            gs::RenderOutput secondary_render_output = gs::rasterize(
                const_cast<Camera&>(*secondary_camera), // TODO: Check if rasterize truly needs non-const Camera&
                model_,
                secondary_bg_color,
                1.0f, // scaling_modifier
                false, // packed
                false, // antialiased
                gs::RenderMode::RGB // Assuming RGB is sufficient for loss derivatives
            );

            if (!secondary_render_output.image.defined() || secondary_render_output.image.numel() == 0) {
                if (options_.debug_print_shapes) {
                    std::cout << "[NewtonOpt KNN] Secondary render output image is empty. Skipping this KNN target." << std::endl;
                }
                continue;
            }

            // 3. Prepare images for loss derivative computation (ensure HWC, on device)
            torch::Tensor sec_rendered_img_squeezed = secondary_render_output.image.squeeze(0);
            if (sec_rendered_img_squeezed.dim() == 3 && sec_rendered_img_squeezed.size(0) == 3) { // CHW to HWC
                sec_rendered_img_squeezed = sec_rendered_img_squeezed.permute({1, 2, 0}).contiguous();
            }
            torch::Tensor sec_gt_img_prepared = secondary_gt_image; // Already downsampled and on device from strategy
            if (sec_gt_img_prepared.dim() == 3 && sec_gt_img_prepared.size(0) == 3) { // CHW to HWC
                 sec_gt_img_prepared = sec_gt_img_prepared.permute({1, 2, 0}).contiguous();
            }
             TORCH_CHECK(sec_rendered_img_squeezed.sizes() == sec_gt_img_prepared.sizes(),
                        "Secondary rendered and GT image sizes mismatch: ", sec_rendered_img_squeezed.sizes(), " vs ", sec_gt_img_prepared.sizes());


            // 4. Compute loss derivatives for secondary view
            LossDerivatives secondary_loss_derivs = compute_loss_derivatives_cuda(
                sec_rendered_img_squeezed,
                sec_gt_img_prepared,
                options_.lambda_dssim_for_hessian, // Use same lambda as primary
                options_.use_l2_for_hessian_L_term // Use same L-term choice
            );

            // 5. Compute Hessian components for secondary view.
            //    Gradient g for secondary views will not be used from here, as primary g is from autograd.
            //    Secondary views primarily contribute to stabilizing the Hessian.
            PositionHessianOutput secondary_hess_output = compute_position_hessian_components_cuda(
                model_,
                visibility_mask_for_model, // Re-use primary visibility mask
                *secondary_camera,
                secondary_render_output,
                secondary_loss_derivs.d2L_dc2_diag, // Pass only d2L/dc2 for Hessian
                num_visible_gaussians_in_model      // Re-use count from primary visibility
            );

            // 6. Accumulate Hessian
            if (secondary_hess_output.H_p_packed.defined() && secondary_hess_output.H_p_packed.numel() > 0) {
                 H_p_total_packed.add_(secondary_hess_output.H_p_packed);
            }
            // Do NOT accumulate gradient from secondary_hess_output.grad_p as it's removed
            // and primary gradient comes from autograd.
            // if (secondary_hess_output.grad_p.defined() && secondary_hess_output.grad_p.numel() > 0) {
            //    g_p_total_visible.add_(secondary_hess_output.grad_p); // This line is removed.
            // }
        }
        }

    // IV. Project Hessian and Gradient to 2D camera plane (U_k^T H U_k, U_k^T g)
    torch::Tensor grad_v_projected = torch::zeros({num_visible_gaussians_in_model, 2}, g_p_total_visible.options());
    torch::Tensor H_v_projected_packed = compute_projected_position_hessian_and_gradient(
        H_p_total_packed, g_p_total_visible, means_visible_from_model, primary_camera, grad_v_projected
    );

    // V & VI. Solve for Δv, re-project to Δp
    torch::Tensor delta_p = solve_and_project_position_updates(
        H_v_projected_packed, grad_v_projected, means_visible_from_model, primary_camera,
        options_.damping, options_.step_scale
    );

    // VII. Update model means
        if (delta_p.defined() && delta_p.numel() > 0) { // Check if delta_p is valid
            model_.means().index_add_(0, visible_indices, delta_p);
        }
    } // End of if (options_.optimize_means)

    // === 2. SCALING OPTIMIZATION ===
    if (options_.optimize_scales) {
        if (options_.debug_print_shapes) std::cout << "[NewtonOpt] Calling compute_scale_updates_newton..." << std::endl;
        torch::Tensor grad_scales_raw_visible_autograd = autograd_grad_scales_raw_total.index_select(0, visible_indices);
        AttributeUpdateOutput scale_update = compute_scale_updates_newton(
            visible_indices,
            grad_scales_raw_visible_autograd,
            primary_loss_derivs.d2L_dc2_diag, // Pass d2L/dc2 for Hessian
            primary_camera,
            current_render_output
        );
        if (scale_update.success && scale_update.delta.defined() && scale_update.delta.numel() > 0) {
            // scale_update.delta is delta_log_s (delta for raw log_scales)
            model_.scaling_raw().index_add_(0, visible_indices, scale_update.delta);
        }
    }

    // === 3. ROTATION OPTIMIZATION ===
    if (options_.optimize_rotations) {
        if (options_.debug_print_shapes) std::cout << "[NewtonOpt] Calling compute_rotation_updates_newton..." << std::endl;
        torch::Tensor grad_rotation_raw_visible_autograd = autograd_grad_rotation_raw_total.index_select(0, visible_indices);
        // Note: The autograd_grad_rotation_raw is w.r.t. the 4 quaternion components.
        // The paper's Hessian is for a scalar angle theta_k.
        // This requires careful handling: either the Hessian kernel is for quaternions,
        // or the autograd gradient needs to be projected to an angle gradient if H_theta is scalar.
        // For now, assuming compute_rotation_updates_newton handles this complexity.
        AttributeUpdateOutput rot_update = compute_rotation_updates_newton(
            visible_indices,
            grad_rotation_raw_visible_autograd, // This is grad w.r.t. raw quaternions
            primary_loss_derivs.d2L_dc2_diag,
            primary_camera,
            current_render_output
        );
        if (rot_update.success && rot_update.delta.defined() && rot_update.delta.numel() > 0) {
            // rot_update.delta is expected to be a delta quaternion (e.g., from axis-angle update)
            torch::Tensor q_delta = rot_update.delta; // [N_vis, 4] (w,x,y,z)
            torch::Tensor q_old_visible = model_.rotation_raw().index_select(0, visible_indices).detach(); // [N_vis, 4]

            // Quaternion multiplication: q_new = q_delta * q_old_visible
            // q_delta = (w_d, x_d, y_d, z_d)
            // q_old_visible = (w_o, x_o, y_o, z_o)
            torch::Tensor w_d = q_delta.select(1,0); torch::Tensor x_d = q_delta.select(1,1);
            torch::Tensor y_d = q_delta.select(1,2); torch::Tensor z_d = q_delta.select(1,3);

            torch::Tensor w_o = q_old_visible.select(1,0); torch::Tensor x_o = q_old_visible.select(1,1);
            torch::Tensor y_o = q_old_visible.select(1,2); torch::Tensor z_o = q_old_visible.select(1,3);

            torch::Tensor q_new_w = w_d * w_o - x_d * x_o - y_d * y_o - z_d * z_o;
            torch::Tensor q_new_x = w_d * x_o + x_d * w_o + y_d * z_o - z_d * y_o;
            torch::Tensor q_new_y = w_d * y_o - x_d * z_o + y_d * w_o + z_d * x_o;
            torch::Tensor q_new_z = w_d * z_o + x_d * y_o - y_d * x_o + z_d * w_o;

            torch::Tensor q_new_stacked = torch::stack({q_new_w, q_new_x, q_new_y, q_new_z}, /*dim=*/-1);
            // Normalize the new quaternions
            torch::Tensor q_new_normalized = torch::nn::functional::normalize(q_new_stacked, torch::nn::functional::NormalizeFuncOptions().dim(1).eps(1e-9));

            model_.rotation_raw().index_copy_(0, visible_indices, q_new_normalized);
        }
    }

    // === 4. OPACITY OPTIMIZATION ===
    if (options_.optimize_opacities) {
        if (options_.debug_print_shapes) std::cout << "[NewtonOpt] Calling compute_opacity_updates_newton..." << std::endl;
        torch::Tensor grad_opacity_raw_visible_autograd = autograd_grad_opacity_raw_total.index_select(0, visible_indices);
        AttributeUpdateOutput opacity_update = compute_opacity_updates_newton(
            visible_indices,
            grad_opacity_raw_visible_autograd,
            primary_loss_derivs.d2L_dc2_diag,
            primary_camera,
            current_render_output
        );
        if (opacity_update.success && opacity_update.delta.defined() && opacity_update.delta.numel() > 0) {
            // opacity_update.delta is delta_for_logits (delta for raw opacity logits)
            model_.opacity_raw().index_add_(0, visible_indices, opacity_update.delta);
        }
    }

    // === 5. SH COEFFICIENTS (COLOR) OPTIMIZATION ===
    if (options_.optimize_shs) {
        if (options_.debug_print_shapes) std::cout << "[NewtonOpt] Calling compute_sh_updates_newton (Placeholder)..." << std::endl;
        AttributeUpdateOutput sh_update = compute_sh_updates_newton(
            visible_indices, primary_loss_derivs, primary_camera,
            current_render_output
        );
        if (sh_update.success && sh_update.delta.defined() && sh_update.delta.numel() > 0) {
            model_.get_shs().index_add_(0, visible_indices, sh_update.delta);
        }
    }
}

// --- Definitions for Attribute Optimization Stubs ---

NewtonOptimizer::AttributeUpdateOutput NewtonOptimizer::compute_scale_updates_newton(
    const torch::Tensor& visible_indices,
    const torch::Tensor& autograd_grad_scales_raw_visible, // Gradient for raw log_scales [N_vis, 3]
    const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian, // [H, W, 3]
    const Camera& camera,
    const gs::RenderOutput& render_output) {

    if (options_.debug_print_shapes) std::cout << "[NewtonOpt] compute_scale_updates_newton for " << visible_indices.numel() << " Gaussians." << std::endl;

    if (visible_indices.numel() == 0) {
        return AttributeUpdateOutput(torch::empty({0}, model_.scaling_raw().options()), true); // Success, but no work
    }

    int num_vis_gaussians = static_cast<int>(visible_indices.numel());
    auto tensor_opts_float = model_.scaling_raw().options(); // Ensure options from a valid tensor

    // --- Compute Hessian H_s for log_scales ---
    torch::Tensor H_s_packed = torch::zeros({num_vis_gaussians, 6}, tensor_opts_float);

    // These tensors are needed by the kernel, get them on the correct device
    auto dev = model_.get_means().device();
    torch::Tensor view_mat_tensor = camera.world_view_transform().to(dev).contiguous();
    torch::Tensor K_matrix_tensor = camera.K().to(dev).contiguous();
    torch::Tensor cam_pos_world_tensor = camera.camera_center().to(dev).contiguous();


    NewtonKernels::compute_scale_hessian_components_kernel_launcher(
        render_output.height, render_output.width, static_cast<int>(d2L_dc2_diag_pixelwise_for_hessian.size(-1)), // C_img
        static_cast<int>(model_.size()), // P_total
        model_.get_means(),         // Pass full model means
        model_.scaling_raw(),       // Pass full model raw log_scales
        model_.get_rotation(),      // Pass full model rotations (quats)
        model_.opacity_raw(),       // Pass full model raw opacities (logits)
        model_.get_shs(),           // Pass full model SHs
        model_.get_active_sh_degree(),
        view_mat_tensor,
        K_matrix_tensor,
        cam_pos_world_tensor,
        render_output,
        visible_indices,            // Pass only indices of visible Gaussians
        d2L_dc2_diag_pixelwise_for_hessian,
        H_s_packed                  // Output Hessian for visible Gaussians
    );
    // H_s_packed is currently filled with zeros by the placeholder kernel.

    // --- Solve the linear system H_s * delta_log_s = -g_s (autograd_grad_scales_raw_visible) ---
    torch::Tensor delta_log_s = torch::zeros_like(autograd_grad_scales_raw_visible);

    if (num_vis_gaussians > 0) {
         NewtonKernels::batch_solve_3x3_system_kernel_launcher(
            num_vis_gaussians,
            H_s_packed, // Hessian computed by our new (placeholder) kernel
            autograd_grad_scales_raw_visible, // Gradient from autograd
            static_cast<float>(options_.damping),
            delta_log_s     // Output: delta_log_s
        );
        // The solver already does H_inv * (-g), so we multiply by step_scale here.
        // Or, more accurately, the solver computes x = (H+dI)^-1 (-g)
        // We want delta = step_scale * x
        delta_log_s.mul_(-options_.step_scale); // Correct: solver gives (H+dI)^-1(g), we want -step * (H+dI)^-1(g) if g is dL/dx.
                                               // If g is -dL/dx, then delta = step_scale * (H+dI)^-1(g).
                                               // Autograd provides dL/dx. So g = dL/dx. We want dx = -step * H_inv * g.
                                               // batch_solve_3x3 outputs (H+dI)^-1(-g). So if g_s is dL/dx, output is (H+dI)^-1(-dL/dx).
                                               // We then multiply by step_scale.
                                               // So, delta_log_s = step_scale * (H+dI)^-1 (-autograd_grad_scales_raw_visible)
                                               // The solver calculates x for Ax=b. Here A=H, b=-g. So x = H^-1(-g).
                                               // Then delta = step_scale * x.
                                               // The batch_solve_3x3_system_kernel appears to solve Hx = -g.
                                               // So, the output `delta_log_s` is already -(H+dI)^-1 g.
                                               // We just need to scale it.
        // Re-evaluating the batch_solve_3x3_symmetric_system_kernel:
        // xp[0]=invDetA*((a11*a22-a12*a21)*(-gp[0]) + ... )
        // This means it solves H * xp = -gp. So xp = -H_inv * gp.
        // If gp is autograd_grad (dL/dx_raw), then xp = -H_inv * (dL/dx_raw).
        // We want to apply update: x_new = x_old - step_scale * H_inv * (dL/dx_raw)
        // So, the update is step_scale * xp.
        delta_log_s.mul_(options_.step_scale);


        delta_log_s.nan_to_num_(0.0, 0.0, 0.0);
    }

    return AttributeUpdateOutput(delta_log_s, true);
}

NewtonOptimizer::AttributeUpdateOutput NewtonOptimizer::compute_rotation_updates_newton(
    const torch::Tensor& visible_indices,
    const torch::Tensor& autograd_grad_rotation_raw_visible, // Grad w.r.t raw quaternions [N_vis, 4]
    const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian,
    const Camera& camera,
    const gs::RenderOutput& render_output) {

    if (options_.debug_print_shapes) std::cout << "[NewtonOpt] compute_rotation_updates_newton for " << visible_indices.numel() << " Gaussians." << std::endl;

    if (visible_indices.numel() == 0) {
        return AttributeUpdateOutput(torch::empty({0}, model_.rotation_raw().options()), true);
    }

    int num_vis_gaussians = static_cast<int>(visible_indices.numel());
    auto tensor_opts_float = model_.rotation_raw().options();
    auto dev = model_.get_means().device();

    // --- Compute r_k_vecs (view vectors to Gaussians, used as rotation axes) ---
    const torch::Tensor current_means_visible = model_.get_means().index_select(0, visible_indices).detach();
    torch::Tensor cam_pos_world_tensor = camera.camera_center().to(dev).contiguous();
    torch::Tensor r_k_vecs = current_means_visible - cam_pos_world_tensor.unsqueeze(0); // [N_vis, 3]

    // --- Compute Hessian H_theta for rotation angle theta_k ---
    // H_theta is scalar per Gaussian [N_vis, 1]
    torch::Tensor H_theta = torch::zeros({num_vis_gaussians, 1}, tensor_opts_float);

    torch::Tensor view_mat_tensor = camera.world_view_transform().to(dev).contiguous();
    torch::Tensor K_matrix_tensor = camera.K().to(dev).contiguous();

    NewtonKernels::compute_rotation_hessian_components_kernel_launcher(
        render_output.height, render_output.width, static_cast<int>(d2L_dc2_diag_pixelwise_for_hessian.size(-1)),
        static_cast<int>(model_.size()), // P_total
        model_.get_means(),
        model_.scaling_raw(),
        model_.rotation_raw(),
        model_.opacity_raw(),
        model_.get_shs(),
        model_.get_active_sh_degree(),
        view_mat_tensor,
        K_matrix_tensor,
        cam_pos_world_tensor,
        r_k_vecs, // Pass r_k_vecs for visible Gaussians
        render_output,
        visible_indices,
        d2L_dc2_diag_pixelwise_for_hessian,
        H_theta // Output Hessian
    );
    // H_theta is currently filled with zeros by the placeholder kernel.

    // --- Gradient for angle theta_k ---
    // This is the tricky part: autograd_grad_rotation_raw_visible is dL/dq (quaternion).
    // If H_theta is for an angle, we need dL/d_theta.
    // dL/d_theta = (dL/dq) * (dq/d_theta).
    // dq/d_theta for q = [cos(theta/2), sin(theta/2)*axis] is non-trivial.
    // For now, as a placeholder, if H_theta is scalar, g_theta must also be scalar.
    // Let's use the norm of the xyz part of dL/dq as a proxy for g_theta (VERY ROUGH APPROXIMATION).
    // This part needs proper derivation based on the chosen rotation parameterization for H.
    torch::Tensor g_theta_proxy = torch::norm(autograd_grad_rotation_raw_visible.slice(1, 1, 4), /*p=*/2, /*dim=*/1, /*keepdim=*/true);
    // Ensure g_theta_proxy has same device and dtype as H_theta for the solver
    g_theta_proxy = g_theta_proxy.to(H_theta.options());


    // --- Solve the linear system H_theta * delta_theta = -g_theta_proxy ---
    torch::Tensor delta_theta = torch::zeros_like(g_theta_proxy); // Should be [N_vis, 1]

    if (num_vis_gaussians > 0) {
        NewtonKernels::batch_solve_1x1_system_kernel_launcher(
            num_vis_gaussians,
            H_theta,
            g_theta_proxy, // Using proxy gradient
            static_cast<float>(options_.damping),
            delta_theta // Output
        );
        // batch_solve_1x1 already computes -g/H. We need to scale by step_scale.
        // The solver output is x for Hx = -g. So x = -H_inv * g.
        // We want update = step_scale * x.
        delta_theta.mul_(options_.step_scale);
        delta_theta.nan_to_num_(0.0, 0.0, 0.0);
    }

    // --- Convert delta_theta (angles) and r_k_vecs (axes) to delta quaternions ---
    torch::Tensor delta_quats = torch::zeros({num_vis_gaussians, 4}, tensor_opts_float);

    if (num_vis_gaussians > 0) {
        torch::Tensor r_k_normalized = torch::nn::functional::normalize(r_k_vecs, torch::nn::functional::NormalizeFuncOptions().dim(1).eps(1e-9));
        // Ensure delta_theta is [N_vis] before operations if it's [N_vis, 1]
        torch::Tensor half_angles = delta_theta.squeeze(-1) * 0.5f;
        torch::Tensor cos_half_angles = torch::cos(half_angles);
        torch::Tensor sin_half_angles = torch::sin(half_angles);

        delta_quats.select(1, 0) = cos_half_angles; // w component
        delta_quats.slice(1, 1, 4) = r_k_normalized * sin_half_angles.unsqueeze(-1); // xyz components
        delta_quats = torch::nn::functional::normalize(delta_quats, torch::nn::functional::NormalizeFuncOptions().dim(1).eps(1e-9));
    }

    return AttributeUpdateOutput(delta_quats, true);
}

NewtonOptimizer::AttributeUpdateOutput NewtonOptimizer::compute_opacity_updates_newton(
    const torch::Tensor& visible_indices,
    const torch::Tensor& autograd_grad_opacity_raw_visible, // Grad w.r.t. raw logits [N_vis, 1]
    const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian,
    const Camera& camera,
    const gs::RenderOutput& render_output) {

    if (options_.debug_print_shapes) std::cout << "[NewtonOpt] compute_opacity_updates_newton for " << visible_indices.numel() << " Gaussians." << std::endl;

    if (visible_indices.numel() == 0) {
        return AttributeUpdateOutput(torch::empty({0}, model_.opacity_raw().options()), true);
    }

    int num_vis_gaussians = static_cast<int>(visible_indices.numel());
    auto tensor_opts_float = model_.opacity_raw().options();
    auto dev = model_.get_means().device();

    // --- Compute base Hessian H_opacity_base for opacity logits ---
    // H_opacity_base is scalar per Gaussian [N_vis] or [N_vis, 1]
    torch::Tensor H_opacity_base = torch::zeros({num_vis_gaussians}, tensor_opts_float);
                                            // Make it [N_vis] for 1x1 solver compatibility

    torch::Tensor view_mat_tensor = camera.world_view_transform().to(dev).contiguous();
    torch::Tensor K_matrix_tensor = camera.K().to(dev).contiguous();
    torch::Tensor cam_pos_world_tensor = camera.camera_center().to(dev).contiguous();

    NewtonKernels::compute_opacity_hessian_components_kernel_launcher(
        render_output.height, render_output.width, static_cast<int>(d2L_dc2_diag_pixelwise_for_hessian.size(-1)),
        static_cast<int>(model_.size()), // P_total
        model_.get_means(),
        model_.scaling_raw(),
        model_.rotation_raw(),
        model_.opacity_raw(), // Pass raw logits
        model_.get_shs(),
        model_.get_active_sh_degree(),
        view_mat_tensor,
        K_matrix_tensor,
        cam_pos_world_tensor,
        render_output,
        visible_indices,
        d2L_dc2_diag_pixelwise_for_hessian,
        H_opacity_base // Output Hessian base term
    );
    // H_opacity_base is currently filled with zeros by the placeholder kernel.

    // --- Add Barrier Term to Hessian ---
    // Barrier terms are typically for the value after activation (sigma in (0,1)), not logits.
    // The paper states: "Hessian and gradient w.r.t. L^t should also incorporate barrier loss i.e., -1/σ_k - 1/(1-σ_k) and 1/(1-σ_k)^2 - 1/σ_k^2."
    // These additions are to g_L_sigma and H_L_sigma.
    // If we optimize logits, the barrier must be transformed or applied differently.
    // For now, let's assume the barrier is handled by the gradient `autograd_grad_opacity_raw_visible` if it includes a barrier loss term,
    // or we add the barrier to H_opacity_base *after* transforming H_opacity_base to be H_sigma_base.
    // This part is complex. The paper's Hessian is likely d2(L+L_barrier)/d(logit)^2.
    // A simpler approach for now: use H_opacity_base as is, and assume barrier loss is part of autograd_grad.
    torch::Tensor H_opacity_total = H_opacity_base; // Placeholder: No explicit barrier added to Hessian here yet.
                                                 // This needs careful derivation if H is for logits and barrier for sigma.

    // --- Solve the linear system H_opacity_logit * delta_logit = -g_logit ---
    // autograd_grad_opacity_raw_visible is dL/d_logit.
    torch::Tensor delta_logits = torch::zeros_like(autograd_grad_opacity_raw_visible);

    if (num_vis_gaussians > 0) {
        // Ensure g_opacity is [N_vis] if H_opacity_total is [N_vis] for 1x1 solver
        torch::Tensor g_opacity_for_solver = autograd_grad_opacity_raw_visible.squeeze(-1);
        if (g_opacity_for_solver.dim() == 0 && num_vis_gaussians == 1) { // Handle case of single element
             g_opacity_for_solver = g_opacity_for_solver.unsqueeze(0);
        }


        NewtonKernels::batch_solve_1x1_system_kernel_launcher(
            num_vis_gaussians,
            H_opacity_total, // Hessian for logits
            g_opacity_for_solver, // Gradient for logits
            static_cast<float>(options_.damping),
            delta_logits.squeeze(-1) // Output delta for logits (ensure it's [N_vis])
        );
        // batch_solve_1x1 output is -H_inv * g. We want step_scale * (-H_inv * g).
        delta_logits.mul_(options_.step_scale);
        delta_logits.nan_to_num_(0.0, 0.0, 0.0);
        if (delta_logits.dim() == 1) { // Ensure it's [N_vis, 1] for update
            delta_logits = delta_logits.unsqueeze(-1);
        }
    }

    return AttributeUpdateOutput(delta_logits, true);
}

NewtonOptimizer::AttributeUpdateOutput NewtonOptimizer::compute_sh_updates_newton(
    const torch::Tensor& visible_indices,
    const LossDerivatives& loss_derivs,
    const Camera& camera, // Needed for view direction r_k for SH basis B_k
    const gs::RenderOutput& render_output) {

    if (options_.debug_print_shapes) std::cout << "[NewtonOpt] STUB: compute_sh_updates_newton called for " << visible_indices.numel() << " Gaussians." << std::endl;

    if (visible_indices.numel() == 0) {
        return AttributeUpdateOutput(torch::empty({0}, model_.get_shs().options()), true);
    }

    // --- Get necessary data ---
    const torch::Tensor current_shs_for_opt = model_.get_shs().index_select(0, visible_indices).detach(); // [N_vis, (deg+1)^2, 3]
    const torch::Tensor current_means = model_.get_means().index_select(0, visible_indices).detach(); // For r_k

    int num_vis_gaussians = static_cast<int>(visible_indices.numel());
    int sh_dim_flat = static_cast<int>(current_shs_for_opt.size(1) * current_shs_for_opt.size(2)); // (deg+1)^2 * 3
    auto tensor_opts_float = current_shs_for_opt.options();
    auto device = current_shs_for_opt.device();

    // Compute view directions r_k = p_k - C_w
    torch::Tensor view_mat_tensor = camera.world_view_transform().to(tensor_opts_float.device()).contiguous();
    torch::Tensor view_mat_2d = view_mat_tensor.select(0,0);
    torch::Tensor R_wc_2d = view_mat_2d.slice(0,0,3).slice(1,0,3);
    torch::Tensor t_wc_2d = view_mat_2d.slice(0,0,3).slice(1,3,4);
    torch::Tensor cam_pos_world = -torch::matmul(R_wc_2d.t(), t_wc_2d).squeeze(); // [3]
    torch::Tensor r_k_vecs = current_means - cam_pos_world.unsqueeze(0); // [N_vis, 3]
    torch::Tensor r_k_vecs_normalized = torch::nn::functional::normalize(r_k_vecs, torch::nn::functional::NormalizeFuncOptions().dim(1).eps(1e-9));


    // --- Placeholder outputs from conceptual CUDA kernels ---
    // H_ck: [N_vis, sh_dim_flat, sh_dim_flat] (block diagonal, or flattened for batched solve)
    // g_ck: [N_vis, sh_dim_flat]
    // The paper optimizes per color component, and ∂²c_R/∂c_{k,R}² = 0 for direct color.
    // This implies H_ck might be simpler, possibly diagonal or block-diagonal per channel.
    // For now, let's assume a general solve for a flattened sh_dim_flat vector per Gaussian.
    // A more precise implementation would handle the per-channel decoupling.
    // The size of the system per Gaussian is ((deg+1)^2) x ((deg+1)^2) for each of R,G,B channels.

    int sh_coeffs_per_channel = static_cast<int>(current_shs_for_opt.size(1)); // (deg+1)^2 // Corrected variable name
    // For H_ck, if decoupled per channel, it's 3 blocks of [N_vis, sh_coeffs_per_channel, sh_coeffs_per_channel]
    // Or, if solving all SH coeffs together: [N_vis, sh_dim_flat, sh_dim_flat]
    // For simplicity in stub, let's assume we get a flattened gradient and a diagonal Hessian.
    torch::Tensor H_ck_diag = torch::zeros({num_vis_gaussians, sh_dim_flat}, tensor_opts_float); // Initialize to zeros
    torch::Tensor g_ck = torch::zeros({num_vis_gaussians, sh_dim_flat}, tensor_opts_float);


    // Conceptual CUDA kernel calls:
    // 1. Kernel to compute SH basis functions B_k(r_k) for each visible Gaussian.
    //    Output: sh_bases [N_vis, (deg+1)^2]
    torch::Tensor sh_bases_values = torch::empty({0}); // Default undefined
    if (num_vis_gaussians > 0) {
        sh_bases_values = NewtonKernels::compute_sh_bases_kernel_launcher(
            model_.get_active_sh_degree(), r_k_vecs_normalized);
    }


    // 2. Kernel to compute Jacobian J_sh = ∂c_pixel/∂c_k and then accumulate H_ck_base and g_ck_base.
    //    J_sh_pixel_channel = G_k * σ_k * (Π_alpha_front) * B_k_channel_coeff
    //    Paper: ∂c_R/∂c_{k,R} = sum_{gaussians} G_k σ_k (Π(1-G_jσ_j)) B_{k,R} (this is ∂(final_pixel_R)/∂(sh_coeff_R_for_gaussian_k))
    //    If ∂²c_R/∂c_{k,R}² (direct part) = 0, then Hessian is J_sh^T * (d2L/dc2) * J_sh
    // Note: The actual call to compute_sh_hessian_gradient_components_kernel_launcher needs many more params from model_
    // For now, this is a simplified call reflecting the stub's current state.
    // The launcher expects many torch::Tensor arguments for model parts.
    // We need to pass them correctly if the kernel were fully implemented.
    // The current stub only zeros H_ck_diag and g_ck, so these details are deferred.
    if (num_vis_gaussians > 0) {
        NewtonKernels::compute_sh_hessian_gradient_components_kernel_launcher(
            render_output.height, render_output.width, static_cast<int>(render_output.image.size(-1)), // C_img
            static_cast<int>(model_.size()), // P_total
            model_.get_means(), model_.get_scaling(), model_.get_rotation(), model_.get_opacity(), model_.get_shs(),
            model_.get_active_sh_degree(),
            sh_bases_values, // Pass evaluated SH bases
            camera.world_view_transform().to(device), // view_matrix
            camera.K().to(device), // K_matrix
            render_output,
            visible_indices, // visible_indices from the function argument
            loss_derivs.dL_dc,
            loss_derivs.d2L_dc2_diag,
            H_ck_diag, // Output (e.g., diagonal of Hessian)
            g_ck       // Output
        );
    }
    // For now, H_ck_diag remains ones and g_ck remains zeros due to stub.


    // --- Solve the linear system H_ck * Δc_k = -g_ck ---
    // If H_ck is diagonal: Δc_k_i = -g_ck_i / (H_ck_diag_i + damping)
    torch::Tensor delta_shs_flat = torch::zeros_like(g_ck);
    if (g_ck.numel() > 0) {
        delta_shs_flat = -g_ck / (H_ck_diag + options_.damping); // Element-wise for diagonal Hessian
        delta_shs_flat = options_.step_scale * delta_shs_flat;
        delta_shs_flat.nan_to_num_(0.0, 0.0, 0.0);
    }

    // Reshape delta_shs_flat [N_vis, sh_dim_flat] back to [N_vis, (deg+1)^2, 3]
    torch::Tensor delta_shs = delta_shs_flat.reshape(current_shs_for_opt.sizes());

    return AttributeUpdateOutput(delta_shs, true); // Return the (currently zero) delta
}
