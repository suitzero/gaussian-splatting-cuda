// src/newton_optimizer.cpp
#include "newton_optimizer.hpp" // Moved from core/
#include "kernels/newton_kernels.cuh" // Path relative to include paths, or needs adjustment
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
NewtonOptimizer::PositionHessianOutput NewtonOptimizer::compute_position_hessian_components_cuda(
    const SplatData& model_snapshot,
    const torch::Tensor& visibility_mask_for_model,
    const Camera& camera,
    const gs::RenderOutput& render_output, // Contains data for rasterizer-culled Gaussians
    const LossDerivatives& loss_derivs,
    int num_visible_gaussians_in_total_model // Number of Gaussians to produce output for
) {
    // Use const getter for SplatData when model_snapshot is const
    torch::Tensor means_tensor = model_snapshot.get_means();
    auto dev = means_tensor.device();
    auto dtype = means_tensor.dtype();
    auto tensor_opts = torch::TensorOptions().device(dev).dtype(dtype);

    // Output tensors for the *num_visible_gaussians_in_total_model*
    torch::Tensor H_p_output_packed = torch::zeros({num_visible_gaussians_in_total_model, 6}, tensor_opts);
    torch::Tensor grad_p_output = torch::zeros({num_visible_gaussians_in_total_model, 3}, tensor_opts);

    // Prepare camera parameters
    torch::Tensor view_mat_tensor_orig = camera.world_view_transform().to(dev).to(dtype); // Corrected method
    torch::Tensor view_mat_tensor = view_mat_tensor_orig.contiguous(); // Ensure contiguity
    torch::Tensor K_matrix = camera.K().to(dev).to(dtype).contiguous(); // Also ensure K_matrix is contiguous

    // Compute camera center C_w = -R_wc^T * t_wc from world_view_transform V = [R_wc | t_wc]
    // V is typically [4,4] or [3,4]. Assuming [4,4] world-to-camera.
    // Corrected slicing:
    torch::Tensor view_mat_2d = view_mat_tensor.select(0, 0); // Get [4,4] matrix assuming batch size is 1
    torch::Tensor R_wc_2d = view_mat_2d.slice(0, 0, 3).slice(1, 0, 3); // Slice to [3,3]
    torch::Tensor t_wc_2d = view_mat_2d.slice(0, 0, 3).slice(1, 3, 4); // Slice to [3,1]
    torch::Tensor R_wc = R_wc_2d.unsqueeze(0); // Add batch dim -> [1,3,3]
    torch::Tensor t_wc = t_wc_2d.unsqueeze(0); // Add batch dim -> [1,3,1]

    // Debug prints for shapes and strides
    if (options_.debug_print_shapes) { // Assuming an option to enable/disable prints
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
        print_tensor_info("loss_derivs.dL_dc", loss_derivs.dL_dc);
        print_tensor_info("loss_derivs.d2L_dc2_diag", loss_derivs.d2L_dc2_diag);
        print_tensor_info("H_p_output_packed", H_p_output_packed);
        print_tensor_info("grad_p_output", grad_p_output);
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
            std::cout << "[VERBOSE_CHECK] Tensor: " << name << std::endl;
            if (!tensor.defined()) {
                std::cout << "  - Defined: No" << std::endl;
                return;
            }
            std::cout << "  - Defined: Yes" << std::endl;
            std::cout << "  - Device: " << tensor.device() << std::endl;
            std::cout << "  - Dtype: " << tensor.scalar_type() << " (Expected: " << expected_type_str << ")" << std::endl;
            std::cout << "  - Contiguous: " << tensor.is_contiguous() << std::endl;
            std::cout << "  - Sizes: " << tensor.sizes() << std::endl;
            std::cout << "  - Numel: " << tensor.numel() << std::endl;
            try {
                if (tensor.numel() > 0) {
                    // Attempt to access data_ptr to see if it crashes here for this specific type
                    // This is a bit risky as it might crash, but that's what we are debugging.
                    // We are not actually using the pointer, just testing the call.
                    if (expected_type_str == "float") tensor.data_ptr<float>();
                    else if (expected_type_str == "bool") tensor.data_ptr<bool>();
                    else if (expected_type_str == "int") tensor.data_ptr<int>();
                    // Add other types if needed
                    std::cout << "  - data_ptr<" << expected_type_str << "> call: OK (or returned nullptr for empty)" << std::endl;
                } else {
                    std::cout << "  - data_ptr<" << expected_type_str << "> call: Skipped (numel is 0)" << std::endl;
                }
            } catch (const c10::Error& e) {
                std::cout << "  - data_ptr<" << expected_type_str << "> call: FAILED (c10::Error): " << e.what() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "  - data_ptr<" << expected_type_str << "> call: FAILED (std::exception): " << e.what() << std::endl;
            } catch (...) {
                std::cout << "  - data_ptr<" << expected_type_str << "> call: FAILED (unknown exception)" << std::endl;
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
        verbose_tensor_check_lambda("loss_derivs.dL_dc", loss_derivs.dL_dc, "float");
        verbose_tensor_check_lambda("loss_derivs.d2L_dc2_diag", loss_derivs.d2L_dc2_diag, "float");
        verbose_tensor_check_lambda("H_p_output_packed", H_p_output_packed, "float");
        verbose_tensor_check_lambda("grad_p_output", grad_p_output, "float");
    }


    // Prepare arguments for kernel launcher by getting data pointers
    // Storing tensors locally before checking and getting data_ptr

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
        verbose_tensor_check_lambda("loss_derivs.dL_dc", loss_derivs.dL_dc, "float");
        verbose_tensor_check_lambda("loss_derivs.d2L_dc2_diag", loss_derivs.d2L_dc2_diag, "float");
        verbose_tensor_check_lambda("H_p_output_packed", H_p_output_packed, "float"); // Already local
        verbose_tensor_check_lambda("grad_p_output", grad_p_output, "float"); // Already local
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
    const float* dL_dc_pixelwise_ptr = gs::torch_utils::get_const_data_ptr<float>(loss_derivs.dL_dc, "loss_derivs.dL_dc");
    const float* d2L_dc2_diag_pixelwise_ptr = gs::torch_utils::get_const_data_ptr<float>(loss_derivs.d2L_dc2_diag, "loss_derivs.d2L_dc2_diag");
    float* H_p_output_packed_ptr = gs::torch_utils::get_data_ptr<float>(H_p_output_packed, "H_p_output_packed");
    float* grad_p_output_ptr = gs::torch_utils::get_data_ptr<float>(grad_p_output, "grad_p_output");

    NewtonKernels::compute_position_hessian_components_kernel_launcher(
        render_output.height, render_output.width, render_output.image.size(-1), // Image: H, W, C
        p_total_for_kernel, // Total P Gaussians in model
        means_3d_all_ptr,
        scales_all_ptr,
        rotations_all_ptr,
        opacities_all_ptr,
        shs_all_ptr,
        model_snapshot.get_active_sh_degree(),
        static_cast<int>(model_snapshot.get_shs().size(1)), // sh_coeffs_dim
        view_matrix_ptr,
        K_matrix_ptr,
        cam_pos_world_ptr,
        means_2d_render_ptr,
        depths_render_ptr,
        radii_render_ptr,
        static_cast<int>(render_output.means2d.defined() ? render_output.means2d.size(0) : 0), // P_render
        visibility_mask_for_model, // Pass the tensor object directly
        dL_dc_pixelwise_ptr,
        d2L_dc2_diag_pixelwise_ptr,
        num_visible_gaussians_in_total_model, // Number of Gaussians to produce output for
        H_p_output_packed_ptr,
        grad_p_output_ptr,
        options_.debug_print_shapes // Pass the flag
    );

    return {H_p_output_packed, grad_p_output};
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

    torch::Tensor view_mat_tensor_orig = camera.world_view_transform().to(tensor_opts.device()); // Corrected
    torch::Tensor view_mat_tensor = view_mat_tensor_orig.contiguous(); // Ensure contiguity
    // Compute camera center C_w = -R_wc^T * t_wc
    // Corrected slicing:
    torch::Tensor view_mat_2d_proj = view_mat_tensor.select(0, 0); // Get [4,4] matrix assuming batch size is 1
    torch::Tensor R_wc_2d_proj = view_mat_2d_proj.slice(0, 0, 3).slice(1, 0, 3); // Slice to [3,3]
    torch::Tensor t_wc_2d_proj = view_mat_2d_proj.slice(0, 0, 3).slice(1, 3, 4); // Slice to [3,1]
    torch::Tensor R_wc_proj = R_wc_2d_proj.unsqueeze(0); // Add batch dim -> [1,3,3]
    torch::Tensor t_wc_proj = t_wc_2d_proj.unsqueeze(0); // Add batch dim -> [1,3,1]

    // Debug prints for shapes and strides
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
        static_cast<float>(step_scale), // step_scale is applied inside kernel: delta_v = -step_scale * H_inv * g
        gs::torch_utils::get_data_ptr<float>(delta_v)
    );

    torch::Tensor delta_p = torch::zeros({num_visible_gaussians, 3}, tensor_opts);
    torch::Tensor view_mat_tensor_orig = camera.world_view_transform().to(tensor_opts.device()); // Corrected
    torch::Tensor view_mat_tensor = view_mat_tensor_orig.contiguous(); // Ensure contiguity
    // Compute camera center C_w = -R_wc^T * t_wc
    // Corrected slicing:
    torch::Tensor view_mat_2d_solve = view_mat_tensor.select(0, 0); // Get [4,4] matrix assuming batch size is 1
    torch::Tensor R_wc_2d_solve = view_mat_2d_solve.slice(0, 0, 3).slice(1, 0, 3); // Slice to [3,3]
    torch::Tensor t_wc_2d_solve = view_mat_2d_solve.slice(0, 0, 3).slice(1, 3, 4); // Slice to [3,1]
    torch::Tensor R_wc_solve = R_wc_2d_solve.unsqueeze(0); // Add batch dim -> [1,3,3]
    torch::Tensor t_wc_solve = t_wc_2d_solve.unsqueeze(0); // Add batch dim -> [1,3,1]

    // Debug prints for shapes and strides
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


// Main step function (partial implementation for position)
void NewtonOptimizer::step(int iteration,
                           const torch::Tensor& visibility_mask_for_model, // Boolean mask for model_.means() [Total_N]
                           const gs::RenderOutput& current_render_output, // From primary target
                           const Camera& primary_camera,
                           const torch::Tensor& primary_gt_image, // Already on device [H,W,C]
                           const std::vector<std::pair<const Camera*, torch::Tensor>>& knn_secondary_targets_data) {

    if (!options_.optimize_means) {
        return;
    }

    torch::NoGradGuard no_grad;

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
        return; // Early exit
    }

    torch::Tensor means_visible_from_model = model_.means().detach().index_select(0, visible_indices);

    // I. Compute Loss Derivatives for primary target
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

    // II. Compute Hessian components (H_p, g_p) for primary target
    if (options_.debug_print_shapes) {
        std::cout << "[NewtonOpt STEP] Checking this->model_ BEFORE call to compute_position_hessian_components_cuda:" << std::endl;
        const SplatData& model_ref_in_step = this->model_;
        std::cout << "  - model_ref_in_step.get_means().defined(): " << model_ref_in_step.get_means().defined() << std::endl;
        if (model_ref_in_step.get_means().defined()) {
            std::cout << "  - model_ref_in_step.get_means().sizes(): " << model_ref_in_step.get_means().sizes() << std::endl;
            try {
                std::cout << "  - model_ref_in_step.size() direct call: " << model_ref_in_step.size() << std::endl;
            } catch (const c10::Error& e) {
                std::cout << "  - model_ref_in_step.size() direct call: FAILED with c10::Error: " << e.what_without_backtrace() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "  - model_ref_in_step.size() direct call: FAILED with std::exception: " << e.what() << std::endl;
            } catch (...) {
                std::cout << "  - model_ref_in_step.size() direct call: FAILED with unknown exception." << std::endl;
            }
        } else {
            std::cout << "  - model_ref_in_step.get_means() is UNDEFINED." << std::endl;
        }
    }
    PositionHessianOutput primary_hess_output = compute_position_hessian_components_cuda(
        model_, visibility_mask_for_model, primary_camera, current_render_output, primary_loss_derivs, num_visible_gaussians_in_model
    );

    torch::Tensor H_p_total_packed = primary_hess_output.H_p_packed.clone(); // [N_vis_model, 6]
    torch::Tensor g_p_total_visible = primary_hess_output.grad_p.clone(); // [N_vis_model, 3]

    // III. Handle Secondary Targets for Overshoot Prevention
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

            // 5. Compute Hessian and gradient components for secondary view
            //    Using primary view's visibility_mask_for_model and num_visible_gaussians_in_model
            //    as per paper's "sparsely evaluated" idea.
            PositionHessianOutput secondary_hess_output = compute_position_hessian_components_cuda(
                model_,
                visibility_mask_for_model, // Re-use primary visibility mask
                *secondary_camera,
                secondary_render_output,
                secondary_loss_derivs,
                num_visible_gaussians_in_model // Re-use count from primary visibility
            );

            // 6. Accumulate
            if (secondary_hess_output.H_p_packed.defined() && secondary_hess_output.H_p_packed.numel() > 0) {
                 H_p_total_packed.add_(secondary_hess_output.H_p_packed);
            }
            if (secondary_hess_output.grad_p.defined() && secondary_hess_output.grad_p.numel() > 0) {
                g_p_total_visible.add_(secondary_hess_output.grad_p);
            }
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

    // === 2. SCALING OPTIMIZATION ===
    if (options_.optimize_scales) {
        if (options_.debug_print_shapes) std::cout << "[NewtonOpt] Calling compute_scale_updates_newton (Placeholder)..." << std::endl;
        AttributeUpdateOutput scale_update = compute_scale_updates_newton(
            /* model_, */ visible_indices, primary_loss_derivs, primary_camera,
            current_render_output
        );
        if (scale_update.success && scale_update.delta.defined() && scale_update.delta.numel() > 0) {
            model_.get_scaling().index_add_(0, visible_indices, scale_update.delta);
        }
    }

    // === 3. ROTATION OPTIMIZATION ===
    if (options_.optimize_rotations) {
        if (options_.debug_print_shapes) std::cout << "[NewtonOpt] Calling compute_rotation_updates_newton (Placeholder)..." << std::endl;
         AttributeUpdateOutput rot_update = compute_rotation_updates_newton(
            visible_indices, primary_loss_derivs, primary_camera,
            current_render_output
        );
        if (rot_update.success && rot_update.delta.defined() && rot_update.delta.numel() > 0) {
            // Placeholder: actual rotation update is q_new = delta_q * q_old
            // model_.get_rotation().index_add_(0, visible_indices, rot_update.delta); // Not for quaternions
        }
    }

    // === 4. OPACITY OPTIMIZATION ===
    if (options_.optimize_opacities) {
        if (options_.debug_print_shapes) std::cout << "[NewtonOpt] Calling compute_opacity_updates_newton (Placeholder)..." << std::endl;
        AttributeUpdateOutput opacity_update = compute_opacity_updates_newton(
            visible_indices, primary_loss_derivs, primary_camera,
            current_render_output
        );
        if (opacity_update.success && opacity_update.delta.defined() && opacity_update.delta.numel() > 0) {
            // Placeholder: actual update might be in logit space + sigmoid, or handle barriers
            model_.get_opacity().index_add_(0, visible_indices, opacity_update.delta);
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
    /* const SplatData& model_snapshot, */ // model_ is a member
    const torch::Tensor& visible_indices,
    const LossDerivatives& loss_derivs,
    const Camera& camera,
    const gs::RenderOutput& render_output) {

    if (options_.debug_print_shapes) std::cout << "[NewtonOpt] STUB: compute_scale_updates_newton called for " << visible_indices.numel() << " Gaussians." << std::endl;

    if (visible_indices.numel() == 0) {
        return AttributeUpdateOutput(torch::empty({0}, model_.get_scaling().options()), true); // Success, but no work
    }

    // --- Get necessary data ---
    const torch::Tensor current_scales_for_opt = model_.get_scaling().index_select(0, visible_indices).detach(); // Renamed
    const torch::Tensor current_rotations = model_.get_rotation().index_select(0, visible_indices).detach();
    const torch::Tensor current_means = model_.get_means().index_select(0, visible_indices).detach();
    // Other model params like opacity, SHs might be needed if ∂c/∂s_k depends on them for full color C.

    int num_vis_gaussians = static_cast<int>(visible_indices.numel());
    auto tensor_opts_float = current_scales_for_opt.options(); // Use renamed variable

    // --- Placeholder outputs from conceptual CUDA kernels ---
    // These would compute ∂c/∂s_k and ∂²c/∂s_k² per pixel, then sum over pixels.
    // For simplicity, let's assume they directly output per-Gaussian H_s and g_s.
    // H_s_k : [num_vis_gaussians, 6] (for 3x3 symmetric Hessian of scales)
    // g_s_k : [num_vis_gaussians, 3] (gradient w.r.t. scales)
    torch::Tensor H_s_packed = torch::zeros({num_vis_gaussians, 6}, tensor_opts_float);
    torch::Tensor g_s = torch::zeros({num_vis_gaussians, 3}, tensor_opts_float);

    // Conceptual kernel call to compute per-Gaussian Hessian and gradient for scales
    // This kernel would be extremely complex, involving:
    // - Projecting Gaussians (like in position solve)
    // - Calculating ∂Σ_k/∂s_k (how 3D scale affects 2D covariance)
    // - Calculating ∂G_k/∂Σ_k (how Gaussian PDF changes with 2D covariance)
    // - Calculating ∂c/∂G_k (how color changes with Gaussian PDF value - depends on blending)
    // - Chaining these for ∂c/∂s_k and its second derivative ∂²c/∂s_k²
    // - Summing contributions over pixels using loss_derivs.dL_dc and loss_derivs.d2L_dc2_diag
    //   to form H_s_k and g_s_k using equations like:
    //   g_s_k = sum_pixels [ (∂c(pixel)/∂s_k)ᵀ * (dL/dc(pixel)) ]
    //   H_s_k = sum_pixels [ (∂c(pixel)/∂s_k)ᵀ * (d²L/dc²(pixel)) * (∂c(pixel)/∂s_k) + (dL/dc(pixel)) ⋅ (∂²c(pixel)/∂s_k²) ]
    /*
    NewtonKernels::compute_scale_hessian_gradient_components_kernel_launcher(
        render_output.height, render_output.width, render_output.image.size(-1), // C_img
        model_, // Pass relevant parts of model or specific tensors
        visible_indices,
        view_mat_tensor, // Need view_mat from primary_camera
        K_matrix,        // Need K from primary_camera
        cam_pos_world,   // Need cam_pos from primary_camera
        render_output,   // For tile iterators, etc.
        loss_derivs.dL_dc,
        loss_derivs.d2L_dc2_diag,
        H_s_packed, // Output
        g_s         // Output
    );
    */
    // For now, H_s_packed and g_s are zeros.

    // Paper mentions projection to eigenvalue space (lambda_min, lambda_max) for robustness.
    // Σ_k = V_k Λ_k V_kᵀ ; E_2ᵀ : (V_k J_k W_k R_k) : E_3 s_k = λ_k
    // This implies a transformation T_k such that Δs_k = T_k' Δλ_k or similar.
    // And g_λ = T_kᵀ g_s, H_λ = T_kᵀ H_s T_k.
    // This is a change of variables for the optimization.
    // For this structural stub, we'll proceed as if solving directly for Δs_k.
    // A full implementation would need the projection logic.

    // --- Solve the linear system H_s * Δs = -g_s ---
    // This would be a batch 3x3 solve for each Gaussian.
    // For simplicity, placeholder for delta_s (actual solve needed).
    torch::Tensor delta_s = torch::zeros_like(g_s);
    if (g_s.numel() > 0) {
        // Conceptual:
        // delta_s = NewtonKernels::batch_solve_3x3_system_kernel_launcher(H_s_packed, g_s, options_.damping);
        // delta_s = -options_.step_scale * delta_s; // Apply step scale

        // Placeholder: simplified update (e.g., gradient descent on scales for testing)
        // This is NOT the Newton step.
        delta_s = -options_.step_scale * g_s * 0.01; // Small learning rate for placeholder
    }


    // TODO: Implement paper's "Scaling solve"
    // 1. Get current scales: model_.get_scaling().index_select(0, visible_indices)
    // 2. Compute ∂c/∂s_k, ∂²c/∂s_k² (VERY COMPLEX - requires new CUDA kernels & use of supplement)
    //    - ∂c/∂s_k = (∂c/∂G_k) * (∂G_k/∂Σ_k) : (∂Σ_k/∂λ_k) * (∂λ_k/∂s_k)
    //    - Uses opt_params_ref_ for loss parameters, options_ for Newton parameters
    // 3. Assemble Hessian H_s_k and gradient g_s_k for scales
    // 4. Project to T_k subspace (optional, from paper)
    // 5. Solve Δs_k = -H_s_k⁻¹ g_s_k (or for Δλ_k)
    if (visible_indices.numel() == 0) return AttributeUpdateOutput(torch::empty({0}), false);
    torch::Tensor current_scales = model_.get_scaling().index_select(0, visible_indices);
    if (current_scales.numel() > 0) {
        return AttributeUpdateOutput(torch::zeros_like(current_scales)); // Return zero delta
    }
    return AttributeUpdateOutput(torch::empty({0}, model_.get_scaling().options()));
}

NewtonOptimizer::AttributeUpdateOutput NewtonOptimizer::compute_rotation_updates_newton(
    const torch::Tensor& visible_indices,
    const LossDerivatives& loss_derivs,
    const Camera& camera,
    const gs::RenderOutput& render_output) {

    if (options_.debug_print_shapes) std::cout << "[NewtonOpt] STUB: compute_rotation_updates_newton called for " << visible_indices.numel() << " Gaussians." << std::endl;

    if (visible_indices.numel() == 0) {
        return AttributeUpdateOutput(torch::empty({0}, model_.get_rotation().options().dtype(torch::kFloat).device(model_.get_rotation().device())), true); // Return empty delta if no visible gaussians
    }

    // --- Get necessary data ---
    const torch::Tensor current_rotations_quat = model_.get_rotation().index_select(0, visible_indices).detach(); // [N_vis, 4]
    const torch::Tensor current_means = model_.get_means().index_select(0, visible_indices).detach(); // Needed for r_k
    // Other model params (scales, opacity, SHs) might be needed for full ∂c/∂θ_k

    int num_vis_gaussians = static_cast<int>(visible_indices.numel());
    auto tensor_opts_float = current_rotations_quat.options(); // Should be float

    // Paper parameterizes rotation by an angle θ_k around axis r_k (view vector)
    // r_k = p_k - C_w (world space vector from camera center to Gaussian mean)
    // This r_k needs to be computed for each visible Gaussian.
    // cam_pos_world can be obtained similarly to how it's done in position optimization.
    torch::Tensor view_mat_tensor = camera.world_view_transform().to(tensor_opts_float.device()).contiguous();
    torch::Tensor view_mat_2d = view_mat_tensor.select(0,0);
    torch::Tensor R_wc_2d = view_mat_2d.slice(0,0,3).slice(1,0,3);
    torch::Tensor t_wc_2d = view_mat_2d.slice(0,0,3).slice(1,3,4);
    torch::Tensor cam_pos_world = -torch::matmul(R_wc_2d.t(), t_wc_2d).squeeze(); // [3]

    torch::Tensor r_k_vecs = current_means - cam_pos_world.unsqueeze(0); // [N_vis, 3]
    // r_k should be normalized, but paper might use unnormalized in some places for axis.
    // The axis for Δq_k is r_k (normalized).

    // --- Placeholder outputs from conceptual CUDA kernels ---
    // H_theta_k : [num_vis_gaussians, 1] (scalar Hessian for angle theta_k)
    // g_theta_k : [num_vis_gaussians, 1] (scalar gradient w.r.t. theta_k)
    torch::Tensor H_theta = torch::zeros({num_vis_gaussians, 1}, tensor_opts_float);
    torch::Tensor g_theta = torch::zeros({num_vis_gaussians, 1}, tensor_opts_float);

    // Conceptual kernel call to compute per-Gaussian Hessian and gradient for rotation angle theta_k
    // This kernel would be very complex:
    // - For each Gaussian, determine r_k.
    // - Compute ∂c/∂θ_k and ∂²c/∂θ_k² (paper: (∂c/∂G_k)*(∂G_k/∂Σ_k)*(∂Σ_k/∂θ_k) and its second derivative).
    //   This involves derivatives of 2D covariance Σ_k w.r.t. θ_k.
    // - Sum contributions over pixels using loss_derivs.dL_dc and loss_derivs.d2L_dc2_diag.
    /*
    NewtonKernels::compute_rotation_hessian_gradient_components_kernel_launcher(
        render_output.height, render_output.width, render_output.image.size(-1),
        model_, // or specific tensors: means, scales, rotations, opacities, shs
        visible_indices,
        r_k_vecs, // Axis of rotation for each Gaussian
        primary_camera, // For full view, projection matrices if needed by ∂Σ_k/∂θ_k
        render_output,
        loss_derivs.dL_dc,
        loss_derivs.d2L_dc2_diag,
        H_theta, // Output
        g_theta  // Output
    );
    */
    // For now, H_theta and g_theta are zeros.

    // --- Solve the linear system H_theta * Δtheta = -g_theta ---
    // This is a batch 1x1 solve: Δtheta = -g_theta / H_theta
    torch::Tensor delta_theta = torch::zeros_like(g_theta);
    if (g_theta.numel() > 0) {
        // Conceptual:
        // delta_theta = NewtonKernels::batch_solve_1x1_system_kernel_launcher(H_theta, g_theta, options_.damping);
        // delta_theta = -options_.step_scale * delta_theta; // Apply step scale

        // Placeholder: simplified update (e.g., gradient descent on theta for testing)
        // This is NOT the Newton step.
        // H_theta would need to be regularized (H_theta + damping).
        // delta_theta = -options_.step_scale * g_theta / (H_theta.abs().clamp_min(1e-6) + options_.damping);
        delta_theta = -options_.step_scale * g_theta * 0.01; // Small learning rate for placeholder
    }

    // The paper states update is Δq_k = [cos(θ_k_update/2), sin(θ_k_update/2) * normalized_r_k]^T
    // where θ_k_update is our delta_theta.
    // This delta_theta is the actual angle of rotation for the update.
    // So, the output 'delta' of this function should represent these delta_thetas.
    // The application q_new = delta_q * q_old will be handled in the main step() function.

    // TODO: Implement paper's "Rotation solve" (update as Δθ_k around r_k)
    // 1. Get current rotations: model_.get_rotation().index_select(0, visible_indices)
    // 2. Compute ∂c/∂θ_k, ∂²c/∂θ_k²
    // 3. Assemble H_θ_k, g_θ_k
    // 4. Solve Δθ_k = -H_θ_k⁻¹ g_θ_k
    if (visible_indices.numel() == 0) return AttributeUpdateOutput(torch::empty({0}), false);
    // Placeholder delta for θ_k might be [N_vis, 1]
    torch::Tensor current_rotations = model_.get_rotation().index_select(0, visible_indices);
     if (current_rotations.numel() > 0) {
        return AttributeUpdateOutput(torch::zeros({visible_indices.size(0), 1}, current_rotations.options()));
    }
    return AttributeUpdateOutput(torch::empty({0}, model_.get_rotation().options()));
}

NewtonOptimizer::AttributeUpdateOutput NewtonOptimizer::compute_opacity_updates_newton(
    const torch::Tensor& visible_indices,
    const LossDerivatives& loss_derivs,
    const Camera& camera,
    const gs::RenderOutput& render_output) {

    if (options_.debug_print_shapes) std::cout << "[NewtonOpt] STUB: compute_opacity_updates_newton called for " << visible_indices.numel() << " Gaussians." << std::endl;

    if (visible_indices.numel() == 0) {
        return AttributeUpdateOutput(torch::empty({0}, model_.get_opacity().options()), true);
    }

    // --- Get necessary data & parameters ---
    // Note: get_opacity() applies sigmoid. For barrier terms, we need raw σ_k in (0,1).
    // The paper's barrier derivatives are in terms of σ_k, not logit(σ_k).
    // We should use the direct output of get_opacity() which is already sigmoided.
    const torch::Tensor current_opacities_sigma = model_.get_opacity().index_select(0, visible_indices).detach(); // [N_vis] (already in range [0,1])

    int num_vis_gaussians = static_cast<int>(visible_indices.numel());
    auto tensor_opts_float = current_opacities_sigma.options();
    auto device = current_opacities_sigma.device();

    // Barrier parameters (alpha_sigma from paper, though paper's derivatives don't show it explicitly)
    // Assuming opt_params_ref_.log_barrier_alpha_opacity is the alpha_sigma.
    // If the paper's given derivatives for barrier already include alpha, then we don't multiply again.
    // The paper states: "Hessian and gradient w.r.t. L^t should also incorporate barrier loss i.e., -1/σ_k - 1/(1-σ_k) and 1/(1-σ_k)^2 - 1/σ_k^2."
    // This implies these ARE the additions to g_L and H_L respectively.
    float alpha_sigma = 1.0f; // Default if not in params, or assume it's baked into paper's derivative forms.
    // Example: if (opt_params_ref_.defined_log_barrier_alpha_opacity) alpha_sigma = opt_params_ref_.log_barrier_alpha_opacity;


    // --- Calculate derivatives of Log Barrier terms ---
    // Ensure opacities are clamped slightly away from 0 and 1 for barrier stability
    torch::Tensor sigma_k = current_opacities_sigma.clamp(1e-7f, 1.0f - 1e-7f);
    torch::Tensor g_barrier = -(1.0f / sigma_k + 1.0f / (1.0f - sigma_k)); // Paper's g_barrier term [N_vis]
    torch::Tensor H_barrier = 1.0f / torch::pow(1.0f - sigma_k, 2) - 1.0f / torch::pow(sigma_k, 2); // Paper's H_barrier term [N_vis]
    // If alpha_sigma is a separate multiplier for the barrier *loss term itself*:
    // g_barrier *= alpha_sigma; H_barrier *= alpha_sigma; // (This depends on precise definition)

    // --- Placeholder for base Hessian and Gradient from color terms ---
    // H_sigma_base_k : [num_vis_gaussians, 1] (scalar Hessian for opacity sigma_k from color rendering)
    // g_sigma_base_k : [num_vis_gaussians, 1] (scalar gradient w.r.t. sigma_k from color rendering)
    torch::Tensor H_sigma_base = torch::zeros({num_vis_gaussians}, tensor_opts_float); // scalar, so [N_vis]
    torch::Tensor g_sigma_base = torch::zeros({num_vis_gaussians}, tensor_opts_float); // scalar, so [N_vis]

    // Conceptual CUDA kernel calls:
    // 1. Kernel to compute dc_dopacity = ∂c/∂σ_k for each visible Gaussian & affected pixel.
    //    Paper: ∂c/∂σ_k = G_k (Π(1-α_j)) (c_gauss_k - C_contrib_behind)
    //    Paper also states ∂²c/∂σ_k² = 0. This simplifies H_sigma_base.
    //    This dc_dopacity would be [N_vis, Num_pixels_affected_by_k, C_channels]
    //    For now, this is a major missing piece.
    /*
    torch::Tensor dc_dopacity_packed; // Complex output
    NewtonKernels::compute_dc_dopacity_kernel_launcher(
        model_, visible_indices, camera, render_output, dc_dopacity_packed);
    */

    // 2. Kernel to accumulate H_sigma_base and g_sigma_base using dc_dopacity.
    //    g_sigma_base_k = sum_pixels [ (∂c/∂σ_k)ᵀ ⋅ (dL/dc) ]
    //    H_sigma_base_k = sum_pixels [ (∂c/∂σ_k)ᵀ ⋅ (d²L/dc²) ⋅ (∂c/∂σ_k) ] (since ∂²c/∂σ_k² = 0)
    /*
    NewtonKernels::accumulate_opacity_hessian_gradient_kernel_launcher(
        dc_dopacity_packed, // From previous kernel
        loss_derivs.dL_dc,
        loss_derivs.d2L_dc2_diag,
        render_output, // For pixel mapping if needed
        H_sigma_base, // Output
        g_sigma_base  // Output
    );
    */
    // For now, H_sigma_base and g_sigma_base are zeros from initialization.

    // --- Combine with barrier terms ---
    torch::Tensor H_sigma_total = H_sigma_base + H_barrier; // [N_vis]
    torch::Tensor g_sigma_total = g_sigma_base + g_barrier; // [N_vis]

    // --- Solve the linear system H_sigma * Δsigma = -g_sigma ---
    // Δsigma = -g_sigma / (H_sigma_total + damping)
    torch::Tensor delta_sigma = torch::zeros_like(g_sigma_total);
    if (g_sigma_total.numel() > 0) {
        delta_sigma = -g_sigma_total / (H_sigma_total + options_.damping); // Element-wise
        delta_sigma = options_.step_scale * delta_sigma; // Apply step scale
        // NaN/inf guard
        delta_sigma.nan_to_num_(0.0, 0.0, 0.0);
    }

    // The delta_sigma is an update to sigma_k directly.
    // The barrier terms in H and g are meant to keep sigma_k within (0,1).
    // Clamping might still be needed if updates are too large.
    // delta_sigma = torch::clamp(current_opacities_sigma + delta_sigma, 1e-7f, 1.0f - 1e-7f) - current_opacities_sigma;


    // TODO: Implement paper's "Opacity solve" (with log barriers)
    // 1. Get current opacities: model_.get_opacity().index_select(0, visible_indices)
    // 2. Compute ∂c/∂σ_k (paper says ∂²c/∂σ_k² = 0)
    // 3. Assemble H_σ_k, g_σ_k including barrier terms.
    //    Barrier loss: -alpha_sigma * ( log(σ_k) + log(1-σ_k) ) based on common form, paper has typo?
    //    Paper: L_local <- L_local - alpha_sigma * (sigma_k + ln(1-sigma_k)) -> this barrier form is unusual.
    //    Typically: -alpha * (log(x) + log(1-x)). Gradient: -alpha * (1/x - 1/(1-x)). Hessian: -alpha * (-1/x^2 - 1/(1-x)^2)
    //    Paper's barrier derivatives: -1/σ_k - 1/(1-σ_k) (for grad) and 1/(1-σ_k)^2 - 1/σ_k^2 (for hessian part) - these match -alpha*(log+log) if alpha=1.
    //    float alpha_sigma = opt_params_ref_.log_barrier_alpha_opacity; // Get from params
    // 4. Solve Δσ_k = -H_σ_k⁻¹ g_σ_k
    if (visible_indices.numel() == 0) return AttributeUpdateOutput(torch::empty({0}), false);
    torch::Tensor current_opacities = model_.get_opacity().index_select(0, visible_indices);
    if (current_opacities.numel() > 0) {
        return AttributeUpdateOutput(torch::zeros_like(current_opacities));
    }
    return AttributeUpdateOutput(torch::empty({0}, model_.get_opacity().options()));
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

    int sh_coeffs_per_channel = static_cast<int>(current_shs.size(1)); // (deg+1)^2
    // For H_ck, if decoupled per channel, it's 3 blocks of [N_vis, sh_coeffs_per_channel, sh_coeffs_per_channel]
    // Or, if solving all SH coeffs together: [N_vis, sh_dim_flat, sh_dim_flat]
    // For simplicity in stub, let's assume we get a flattened gradient and a diagonal Hessian.
    torch::Tensor H_ck_diag = torch::ones({num_vis_gaussians, sh_dim_flat}, tensor_opts_float); // Placeholder: Identity Hessian
    torch::Tensor g_ck = torch::zeros({num_vis_gaussians, sh_dim_flat}, tensor_opts_float);


    // Conceptual CUDA kernel calls:
    // 1. Kernel to compute SH basis functions B_k(r_k) for each visible Gaussian.
    //    Output: sh_bases [N_vis, (deg+1)^2]
    /*
    torch::Tensor sh_bases = NewtonKernels::compute_sh_bases_kernel_launcher(
        model_.get_active_sh_degree(), r_k_vecs_normalized);
    */

    // 2. Kernel to compute Jacobian J_sh = ∂c_pixel/∂c_k and then accumulate H_ck_base and g_ck_base.
    //    J_sh_pixel_channel = G_k * σ_k * (Π_alpha_front) * B_k_channel_coeff
    //    Paper: ∂c_R/∂c_{k,R} = sum_{gaussians} G_k σ_k (Π(1-G_jσ_j)) B_{k,R} (this is ∂(final_pixel_R)/∂(sh_coeff_R_for_gaussian_k))
    //    If ∂²c_R/∂c_{k,R}² (direct part) = 0, then Hessian is J_sh^T * (d2L/dc2) * J_sh
    /*
    NewtonKernels::compute_sh_hessian_gradient_components_kernel_launcher(
        render_output.height, render_output.width, render_output.image.size(-1),
        model_, visible_indices, sh_bases, // Pass evaluated SH bases
        camera, render_output, // For view info, tile iterators, accumulated alpha etc.
        loss_derivs.dL_dc,
        loss_derivs.d2L_dc2_diag,
        H_ck_diag, // Output (e.g., diagonal of Hessian)
        g_ck       // Output
    );
    */
    // For now, H_ck_diag and g_ck are zeros/ones.


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
