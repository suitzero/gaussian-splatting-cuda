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
        int knn_k = 3;
        bool optimize_means = true;
        bool optimize_scales = true;
        bool optimize_rotations = true;
        bool optimize_opacities = true;
        bool optimize_shs = true;
        bool debug_print_shapes = true;         // Defaulted to false now
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
};
