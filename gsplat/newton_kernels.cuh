// kernels/newton_kernels.cuh
#pragma once
#include <torch/torch.h>
#include "core/torch_utils.hpp" // For gs::torch_utils like get_data_ptr
#include "core/rasterizer.hpp" // For gs::RenderOutput

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
    int H, int W, int C_img, // Matched with .cuh
    int P_total,
    const float* means_3d_all, const float* scales_all, const float* rotations_all,
    const float* opacities_all, const float* shs_all,
    int sh_degree,
    int sh_coeffs_dim,                           // Matched with .cuh
    const float* view_matrix,                    // Matched with .cuh
    const float* projection_matrix_for_jacobian, // Matched with .cuh
    const float* cam_pos_world,                  // Matched with .cuh
    const torch::Tensor& visibility_mask_for_model_tensor,
    const torch::Tensor& visibility_mask_for_model_tensor,
    // dL_dc_pixelwise is removed as autograd gradients will be used directly for 'g'.
    const float* d2L_dc2_diag_pixelwise_for_hessian, // Still needed for H calculation.
    int num_output_gaussians,
    float* H_p_output_packed, // Matched with .cuh
    // grad_p_output is removed as it's no longer computed by this kernel.
    bool debug_prints_enabled);


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

// Computes Hessian components for scale parameters. Gradient comes from autograd.
void compute_scale_hessian_components_kernel_launcher(
    // Image dimensions
    int H_img, int W_img, int C_img,
    // Model data
    int P_total,
    const torch::Tensor& means_all,
    const torch::Tensor& scales_all,    // These are raw log_scales
    const torch::Tensor& rotations_all,
    const torch::Tensor& opacities_all, // These are raw logits
    const torch::Tensor& shs_all,
    int sh_degree,
    // Camera
    const torch::Tensor& view_matrix,
    const torch::Tensor& K_matrix,
    const torch::Tensor& cam_pos_world,
    // Render output
    const gs::RenderOutput& render_output,
    // Visibility
    const torch::Tensor& visible_indices, // Indices of visible Gaussians [N_vis]
    // Second derivative of loss w.r.t. pixel color (for Hessian)
    const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian, // [H_img, W_img, C_img]
    // Output array (for visible Gaussians)
    torch::Tensor& out_H_s_packed // [N_vis, 6] (for 3x3 symmetric Hessian of log_scales)
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

// Computes Hessian components for rotation parameters. Gradient comes from autograd.
// Rotation can be parameterized e.g. by axis-angle (theta_k around r_k) or directly quaternion components.
// For axis-angle (theta_k), H is scalar per Gaussian. For quaternions, it's more complex (e.g. 3x3 or 4x4).
// Let's assume for now a simplified scalar Hessian for an angle parameterization as in the paper.
void compute_rotation_hessian_components_kernel_launcher(
    // Image dimensions
    int H_img, int W_img, int C_img,
    // Model data
    int P_total,
    const torch::Tensor& means_all,
    const torch::Tensor& scales_all_raw,    // raw log_scales
    const torch::Tensor& rotations_all_raw, // raw quaternions
    const torch::Tensor& opacities_all_raw, // raw logits
    const torch::Tensor& shs_all,
    int sh_degree,
    // Camera & View related
    const torch::Tensor& view_matrix,
    const torch::Tensor& K_matrix,
    const torch::Tensor& cam_pos_world,
    const torch::Tensor& r_k_vecs,      // [N_vis, 3] view vectors (rotation axes for angle parameterization)
    // Render output
    const gs::RenderOutput& render_output,
    // Visibility
    const torch::Tensor& visible_indices, // [N_vis]
    // Second derivative of loss w.r.t. pixel color (for Hessian)
    const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian, // [H_img, W_img, C_img]
    // Output array for visible Gaussians
    torch::Tensor& out_H_theta // [N_vis, 1] (scalar Hessian for angle theta_k)
);

// Solves batch 1x1 linear systems H_theta * delta_theta = -g_theta
void batch_solve_1x1_system_kernel_launcher(
    int num_systems,
    const torch::Tensor& H_theta, // [N_vis, 1] or [N_vis]
    const torch::Tensor& g_theta, // [N_vis, 1] or [N_vis]
    float damping,
    torch::Tensor& out_delta_theta // [N_vis, 1] or [N_vis]
);

// --- Launchers for Opacity Optimization (Placeholders) ---

// Computes Hessian components for opacity parameter (logits). Gradient comes from autograd.
// The paper mentions a barrier term for opacity, which is added in C++. This kernel computes H_base.
// Paper states ∂²c/∂σ_k² = 0, simplifying H_σ_base.
void compute_opacity_hessian_components_kernel_launcher(
    // Image dimensions
    int H_img, int W_img, int C_img,
    // Model data
    int P_total,
    const torch::Tensor& means_all,
    const torch::Tensor& scales_all_raw,    // raw log_scales
    const torch::Tensor& rotations_all_raw, // raw quaternions
    const torch::Tensor& opacities_all_raw, // raw logits
    const torch::Tensor& shs_all,
    int sh_degree,
    // Camera
    const torch::Tensor& view_matrix,
    const torch::Tensor& K_matrix,
    const torch::Tensor& cam_pos_world,
    // Render output
    const gs::RenderOutput& render_output,
    // Visibility
    const torch::Tensor& visible_indices, // [N_vis]
    // Second derivative of loss w.r.t. pixel color (for Hessian)
    const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian, // [H_img, W_img, C_img]
    // Output array for visible Gaussians
    torch::Tensor& out_H_opacity_base // [N_vis] (scalar base Hessian for opacity logits)
);

// --- Launchers for SH (Color) Optimization (Placeholders) ---

// Evaluates SH basis functions for given view directions
// Output: sh_bases_values [N_vis, num_sh_coeffs]
torch::Tensor compute_sh_bases_kernel_launcher(
    int sh_degree,
    const torch::Tensor& normalized_view_vectors // [N_vis, 3] (r_k_normalized)
);

// Computes Hessian components for SH coefficients. Gradient comes from autograd.
// Assumes Hessian is diagonal per SH coefficient (simplification from paper: ∂²c_R/∂c_{k,R}² = 0).
void compute_sh_hessian_components_kernel_launcher(
    // Image dimensions
    int H_img, int W_img, int C_img,
    // Model data
    int P_total,
    const torch::Tensor& means_all,
    const torch::Tensor& scales_all_raw,    // raw log_scales
    const torch::Tensor& rotations_all_raw, // raw quaternions
    const torch::Tensor& opacities_all_raw, // raw logits
    const torch::Tensor& shs_all_raw,       // All SH coeffs (sh0, shN combined) [P_total, (deg+1)^2, 3]
    int sh_degree,
    const torch::Tensor& sh_bases_values, // Precomputed SH basis values for visible Gaussians [N_vis, (deg+1)^2]
    // Camera
    const torch::Tensor& view_matrix,
    const torch::Tensor& K_matrix, // May not be needed if geometric effects on SH basis are ignored for Hessian
    // Render output
    const gs::RenderOutput& render_output,
    // Visibility
    const torch::Tensor& visible_indices, // [N_vis]
    // Second derivative of loss w.r.t. pixel color (for Hessian)
    const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian, // [H_img, W_img, C_img]
    // Output array for visible Gaussians
    torch::Tensor& out_H_sh_diag // [N_vis, num_sh_coeffs_flat] (diagonal of Hessian for SH coeffs)
);


} // namespace NewtonKernels
