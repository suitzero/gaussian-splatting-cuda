// include/newton_optimizer.hpp
#pragma once

#include "core/splat_data.hpp"
#include "core/camera.hpp" // Included for Camera&
#include "core/rasterizer.hpp" // Included for RenderOutput
#include "core/parameters.hpp" // For gs::param::OptimizationParameters
#include "newton_kernels.cuh" // Corrected path to newton_kernels.cuh
#include <torch/torch.h>
#include <vector>
#include <string> // For std::string in Options

// Forward declaration
namespace gs {
    struct RenderOutput;
    namespace param {
        struct OptimizationParameters;
    }
}

class NewtonOptimizer {
public:
    struct Options {
        double step_scale = 1.0;
        double damping = 1e-6;
        int knn_k = 3;
        float secondary_target_downsample = 0.5;
        float lambda_dssim_for_hessian = 0.2f; // Assuming similar to main loss lambda

        bool optimize_means = true;
        bool optimize_scales = true;
        bool optimize_rotations = true;
        bool optimize_opacities = true;
        bool optimize_shs = true;

        // For L2 loss part in Hessian computation (as per paper)
        bool use_l2_for_hessian_L_term = true;
        bool debug_print_shapes = false;         // Defaulted to false now
    };

    NewtonOptimizer(SplatData& splat_data,
                    const gs::param::OptimizationParameters& opt_params,
                    Options options = {});

    void step(int iteration,
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
              const std::vector<std::pair<const Camera*, torch::Tensor>>& knn_secondary_targets_data
              );

private:
    SplatData& model_;
    const gs::param::OptimizationParameters& opt_params_ref_;
    Options options_;

    // --- Loss Derivatives ---
    struct LossDerivatives {
        torch::Tensor dL_dc;     // [H, W, 3] or [B, H, W, 3]
        torch::Tensor d2L_dc2_diag; // [H, W, 3] or [B, H, W, 3] (diagonal of the 3x3 Hessian block for each pixel)
    };
    LossDerivatives compute_loss_derivatives_cuda(
        const torch::Tensor& rendered_image, // [H, W, 3]
        const torch::Tensor& gt_image,       // [H, W, 3]
        float lambda_dssim,
        bool use_l2_loss_term
    );


    // --- Position (Means) ---
    struct PositionHessianOutput {
        torch::Tensor H_p_packed; // Packed symmetric 3x3 Hessian per Gaussian [N_vis, 6]
        // grad_p is removed, it will come from autograd_grad_means passed to step()
    };
    // Computes only the Hessian components for position. Gradient comes from autograd.
    PositionHessianOutput compute_position_hessian_components_cuda(
        const SplatData& model_snapshot,
        const torch::Tensor& visibility_mask_for_model, // [Total_N] bool tensor
        const Camera& camera,
        const gs::RenderOutput& render_output,
        const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian, // [H, W, 3]
        int num_visible_gaussians_in_total_model
    );

    // This function projects both H and g. g will be derived from autograd_grad_means.
    torch::Tensor compute_projected_position_hessian_and_gradient(
        const torch::Tensor& H_p_packed, // [N_vis, 6]
        const torch::Tensor& grad_p,     // [N_vis, 3]
        const torch::Tensor& means_3d_visible, // [N_vis, 3] (only visible means from model)
        const Camera& camera,
        torch::Tensor& out_grad_v       // Output for projected gradient [N_vis, 2]
                                         // Returns H_v_packed [N_vis, 3] (for symmetric 2x2)
    );

    torch::Tensor solve_and_project_position_updates(
        const torch::Tensor& H_v_projected_packed, // [N_vis, 3]
        const torch::Tensor& grad_v_projected,     // [N_vis, 2]
        const torch::Tensor& means_3d_visible,     // [N_vis, 3]
        const Camera& camera,
        double damping,
        double step_scale
    ); // Returns delta_p [N_vis, 3]

    // Placeholder for other parameter groups
    // These will be the actual Newton update computation functions for each attribute
    struct AttributeUpdateOutput {
        torch::Tensor delta;
        bool success = true;
        // Default constructor for placeholder returns
        AttributeUpdateOutput(torch::Tensor d = torch::empty({0}), bool s = true) : delta(d), success(s) {}
    };

    AttributeUpdateOutput compute_scale_updates_newton(
        const torch::Tensor& visible_indices, // Indices of visible Gaussians [N_vis]
        const torch::Tensor& autograd_grad_scales_raw, // Autograd gradient for raw log_scales [N_vis, 3]
        const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian, // [H, W, 3]
        const Camera& camera,
        const gs::RenderOutput& render_output
        // model_ (SplatData) is a member, options_ is a member
    );

    AttributeUpdateOutput compute_rotation_updates_newton(
        const torch::Tensor& visible_indices, // Indices of visible Gaussians [N_vis]
        const torch::Tensor& autograd_grad_rotation_raw, // Autograd gradient for raw quaternions [N_vis, 4]
                                                         // (or for angle if using angle parameterization for g)
        const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian, // [H, W, 3]
        const Camera& camera,
        const gs::RenderOutput& render_output
    );

    AttributeUpdateOutput compute_opacity_updates_newton(
        const torch::Tensor& visible_indices, // Indices of visible Gaussians [N_vis]
        const torch::Tensor& autograd_grad_opacity_raw, // Autograd gradient for raw logits [N_vis, 1]
        const torch::Tensor& d2L_dc2_diag_pixelwise_for_hessian, // [H, W, 3]
        const Camera& camera,
        const gs::RenderOutput& render_output
    );

    AttributeUpdateOutput compute_sh_updates_newton(
        const torch::Tensor& visible_indices,
        const LossDerivatives& loss_derivs,
        const Camera& camera,
        const gs::RenderOutput& render_output);
};
