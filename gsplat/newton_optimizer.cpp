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

// Main step function
void NewtonOptimizer::step(int iteration,
                           const torch::Tensor& visibility_mask_for_model, // Boolean mask for model_.means() [N_total]
                           const torch::Tensor& grad_means,                // [N_total, 3]
                           const torch::Tensor& grad_scales_raw,           // [N_total, 3]
                           const torch::Tensor& grad_rotation_raw,         // [N_total, 4]
                           const torch::Tensor& grad_opacity_raw,          // [N_total, 1]
                           const torch::Tensor& grad_sh0,                  // [N_total, 1, 3]
                           const torch::Tensor& grad_shN,                  // [N_total, K-1, 3]
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
        std::cout << "means:"  << grad_means.sizes() << std::endl;
    }

    if (num_visible_gaussians_in_model == 0) {
        if (options_.debug_print_shapes) {
             std::cout << "[NewtonOpt] Step: No visible Gaussians based on mask at iteration " << iteration << ". Skipping Newton update." << std::endl;
        }
        return; // Early exit if no Gaussians are visible
    }

    // === 4. OPACITY OPTIMIZATION ===
    if (options_.optimize_opacities) {
    }

    // === 5. SH COEFFICIENTS (COLOR) OPTIMIZATION ===
    if (options_.optimize_shs) {
    }
}


