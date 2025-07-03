// kernels/newton_kernels.cuh
#pragma once
#include <torch/torch.h>
#include "core/torch_utils.hpp" // For gs::torch_utils like get_data_ptr

namespace NewtonKernels {

// Launcher for computing dL/dc and d2L/dc2
void compute_loss_derivatives_kernel_launcher(
    const torch::Tensor& rendered_image_tensor, // [H, W, C]
    const torch::Tensor& gt_image_tensor,       // [H, W, C]
    float lambda_dssim,
    bool use_l2_loss_term, // If true, L2+SSIM for L, else L1+SSIM
    torch::Tensor& out_dL_dc_tensor,      // [H, W, C]
    torch::Tensor& out_d2L_dc2_diag_tensor // [H, W, C] (diagonal elements)
);

// Launcher for computing Hessian components for position
// H_p = J_c^T * H_L_c * J_c + G_L_c * H_c_y
// g_p = J_c^T * G_L_c
// J_c = 𝜕𝒄 / 𝜕𝒑𝑘 (Jacobian of final pixel color w.r.t. p_k)
// H_L_c = 𝜕²L/𝜕c² (Hessian of loss w.r.t. final pixel color)
// G_L_c = 𝜕L/𝜕c (Gradient of loss w.r.t. final pixel color)
// H_c_y = 𝜕²𝒄/𝜕p² (Hessian of final pixel color w.r.t. p_k)
void compute_position_hessian_components_kernel_launcher(
    // Image dimensions
    int H, int W, int C_img, // C_img is number of channels in image (e.g. 3 for RGB)
    // Gaussian properties (all Gaussians in the model)
    int P_total,
    const float* means_3d_all,
    const float* scales_all,
    const float* rotations_all,
    const float* opacities_all,
    const float* shs_all,
    int sh_degree,
    int sh_coeffs_dim, // total dimension of SH coeffs per channel (e.g., (sh_degree+1)^2)
    // Camera properties
    const float* view_matrix,
    const float* projection_matrix_for_jacobian, // Typically K matrix [3,3] or [4,4]
    const float* cam_pos_world,
    // Data from RenderOutput (for Gaussians processed by rasterizer, potentially culled)
    const float* means_2d_render,  // [P_render, 2]
    const float* depths_render,   // [P_render]
    const float* radii_render,    // [P_render]
    // visibility_indices_in_render_output (ranks) removed, P_render is the size of above arrays.
    int P_render,
    // Visibility mask for *all* Gaussians in the model [P_total]. True if Gaussian k is visible on screen.
    const torch::Tensor& visibility_mask_for_model_tensor, // Changed from const bool*
    // Loss derivatives (pixel-wise)
    const float* dL_dc_pixelwise,          // [H, W, C_img]
    const float* d2L_dc2_diag_pixelwise,   // [H, W, C_img]
    // Output arrays are for Gaussians where visibility_mask_for_model is true.
    // num_output_gaussians is the count of true in visibility_mask_for_model.
    int num_output_gaussians,
    // Output arrays (dense, for visible Gaussians from model)
    float* H_p_output_packed, // [num_output_gaussians, 6] (symmetric 3x3)
    float* grad_p_output,     // [num_output_gaussians, 3]
    bool debug_prints_enabled // For conditional printing inside launcher
);


// Launcher for projecting Hessian and Gradient to camera plane
void project_position_hessian_gradient_kernel_launcher(
    int num_visible_gaussians,
    const float* H_p_packed_input,
    const float* grad_p_input,
    const float* means_3d_visible,
    const float* view_matrix,
    const float* cam_pos_world,
    float* out_H_v_packed,
    float* out_grad_v
);

// Launcher for solving batch 2x2 linear systems H_v * delta_v = -g_v
void batch_solve_2x2_system_kernel_launcher(
    int num_systems,
    const float* H_v_packed,
    const float* g_v,
    float damping,
    float step_scale, // Applied as: delta_v = -step_scale * H_inv * g
    float* out_delta_v
);

// Launcher for re-projecting delta_v to 3D delta_p = U_k * delta_v
void project_update_to_3d_kernel_launcher(
    int num_updates,
    const float* delta_v,
    const float* means_3d_visible,
    const float* view_matrix,
    const float* cam_pos_world,
    float* out_delta_p
);

// --- Launchers for Scale Optimization (Placeholders) ---

// Computes Hessian and gradient components for scale parameters
void compute_scale_hessian_gradient_components_kernel_launcher(
    // Image dimensions
    int H_img, int W_img, int C_img,
    // Model data (need access to means, scales, rotations, opacities, SHs for full ∂c/∂s_k)
    // For simplicity, let's assume these are passed via some context or specific tensors
    int P_total, // Total number of Gaussians in model
    const torch::Tensor& means_all,     // [P_total, 3]
    const torch::Tensor& scales_all,    // [P_total, 3]
    const torch::Tensor& rotations_all, // [P_total, 4]
    const torch::Tensor& opacities_all, // [P_total]
    const torch::Tensor& shs_all,       // [P_total, K, 3]
    int sh_degree,
    // Camera
    const torch::Tensor& view_matrix,   // [4,4] or [1,4,4]
    const torch::Tensor& K_matrix,      // [3,3] or [1,3,3]
    const torch::Tensor& cam_pos_world, // [3]
    // Render output from primary view (e.g., for tile information if applicable)
    const gs::RenderOutput& render_output, // May not be fully needed if we re-evaluate coverage
    // Visibility
    const torch::Tensor& visible_indices, // Indices of visible Gaussians [N_vis]
    // Loss derivatives (pixel-wise from primary view)
    const torch::Tensor& dL_dc_pixelwise,        // [H_img, W_img, C_img]
    const torch::Tensor& d2L_dc2_diag_pixelwise, // [H_img, W_img, C_img]
    // Output arrays (for visible Gaussians)
    torch::Tensor& out_H_s_packed, // [N_vis, 6] (for 3x3 symmetric Hessian of scales)
    torch::Tensor& out_g_s         // [N_vis, 3] (gradient w.r.t. scales)
    // bool debug_prints_enabled (already added to position launcher, could add here too)
);

// Solves batch 3x3 linear systems H_s * delta_s = -g_s
// (Note: paper might project scales to 2D eigenvalues, then it'd be a 2x2 solve)
void batch_solve_3x3_system_kernel_launcher(
    int num_systems,
    const torch::Tensor& H_s_packed, // [N_vis, 6]
    const torch::Tensor& g_s,        // [N_vis, 3]
    float damping,
    // float step_scale is not here, applied in C++ after solve
    torch::Tensor& out_delta_s       // [N_vis, 3]
);

// --- Launchers for Rotation Optimization (Placeholders) ---

// Computes Hessian and gradient components for rotation angle theta_k
void compute_rotation_hessian_gradient_components_kernel_launcher(
    // Image dimensions
    int H_img, int W_img, int C_img,
    // Model data
    int P_total,
    const torch::Tensor& means_all,
    const torch::Tensor& scales_all,
    const torch::Tensor& rotations_all,
    const torch::Tensor& opacities_all,
    const torch::Tensor& shs_all,
    int sh_degree,
    // Camera & View related
    const torch::Tensor& view_matrix,   // World to Camera
    const torch::Tensor& K_matrix,
    const torch::Tensor& cam_pos_world,
    const torch::Tensor& r_k_vecs,      // [N_vis, 3] view vectors (rotation axes)
    // Render output (if needed for tile iterators, etc.)
    const gs::RenderOutput& render_output,
    // Visibility
    const torch::Tensor& visible_indices, // [N_vis]
    // Loss derivatives
    const torch::Tensor& dL_dc_pixelwise,
    const torch::Tensor& d2L_dc2_diag_pixelwise,
    // Output arrays (for visible Gaussians)
    torch::Tensor& out_H_theta, // [N_vis, 1] (scalar Hessian for angle theta_k)
    torch::Tensor& out_g_theta  // [N_vis, 1] (scalar gradient for angle theta_k)
);

// Solves batch 1x1 linear systems H_theta * delta_theta = -g_theta
void batch_solve_1x1_system_kernel_launcher(
    int num_systems,
    const torch::Tensor& H_theta, // [N_vis, 1] or [N_vis]
    const torch::Tensor& g_theta, // [N_vis, 1] or [N_vis]
    float damping,
    torch::Tensor& out_delta_theta // [N_vis, 1] or [N_vis]
);


} // namespace NewtonKernels
