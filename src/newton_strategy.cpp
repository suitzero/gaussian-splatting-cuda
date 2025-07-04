#include "core/newton_strategy.hpp"
#include "core/rasterizer.hpp" // For gs::rasterize if needed for secondary targets
#include "core/torch_utils.hpp" // For get_bg_color_from_image

#include <algorithm> // for std::sort, std::nth_element
#include <limits>    // for std::numeric_limits
#include <torch/torch.h> // Ensure torch is included for tensor operations
#include <unordered_map> // For uid_to_camera_cache_

// Note: spherical_distance helper function is removed as KNNs are now precomputed.

NewtonStrategy::NewtonStrategy(
    std::unique_ptr<SplatData> splat_data_owner,
    std::shared_ptr<CameraDataset> train_dataset_for_knn)
: splat_data_(std::move(splat_data_owner)),
  train_dataset_ref_(train_dataset_for_knn) {
    TORCH_CHECK(splat_data_, "NewtonStrategy: SplatData owner cannot be null.");
    TORCH_CHECK(train_dataset_ref_, "NewtonStrategy: CameraDataset reference cannot be null.");

    // optim_params_cache_ will be set in initialize() by the Trainer.
    // Caching camera references can be done here or in initialize,
    // but it depends on train_dataset_ref_ which is available now.
    cache_camera_references();
}

void NewtonStrategy::initialize(const gs::param::OptimizationParameters& optimParams) {
    optim_params_cache_ = optimParams;

    if (optim_params_cache_.use_newton_optimizer) {
        NewtonOptimizer::Options newton_opts;
        newton_opts.step_scale = optim_params_cache_.newton_step_scale;
        newton_opts.damping = optim_params_cache_.newton_damping;
        newton_opts.knn_k = optim_params_cache_.newton_knn_k; // This K is for overshoot prevention in Newton step
                                                       // The K for finding secondary targets comes from SplatData's KNNs
        newton_opts.secondary_target_downsample = optim_params_cache_.newton_secondary_target_downsample_factor;
        newton_opts.lambda_dssim_for_hessian = optim_params_cache_.newton_lambda_dssim_for_hessian;
        newton_opts.use_l2_for_hessian_L_term = optim_params_cache_.newton_use_l2_for_hessian_L_term;

        // Attribute-specific optimization flags
        newton_opts.optimize_means = optim_params_cache_.newton_optimize_means;
        newton_opts.optimize_scales = optim_params_cache_.newton_optimize_scales;
        newton_opts.optimize_rotations = optim_params_cache_.newton_optimize_rotations;
        newton_opts.optimize_opacities = optim_params_cache_.newton_optimize_opacities;
        newton_opts.optimize_shs = optim_params_cache_.newton_optimize_shs;

        optimizer_ = std::make_unique<NewtonOptimizer>(*splat_data_, optim_params_cache_, newton_opts);

        // No need to call cache_camera_references() again if called in constructor,
        // unless dataset could change, which is not typical after strategy construction.
        // If optim_params_cache_ was needed for caching, then it should be here.
        // Since it's just caching Camera*, constructor is fine.

    } else {
        // Fallback or error if this strategy is used when use_newton_optimizer is false
        // Or, this strategy should only be created if use_newton_optimizer is true.
        TORCH_CHECK(false, "NewtonStrategy initialized but use_newton_optimizer is false in params!");
    }
}

void NewtonStrategy::compute_visibility_mask_for_model(const gs::RenderOutput& render_output, const SplatData& model) {
    // Based on investigation, render_output.visibility is already a boolean mask
    // of shape [P_total] (total number of Gaussians in the model),
    // indicating which Gaussians have a projected radius > 0.
    // This is suitable for use as visibility_mask_for_model.

    if (!render_output.visibility.defined()) {
        std::cerr << "Warning: render_output.visibility is not defined in NewtonStrategy. Defaulting to all-false mask." << std::endl;
        current_visibility_mask_for_model_ = torch::zeros({model.size()},
            torch::TensorOptions().dtype(torch::kBool).device(model.get_means().device()));
        return;
    }

    TORCH_CHECK(render_output.visibility.dim() == 1 && render_output.visibility.size(0) == model.size(),
                "NewtonStrategy: render_output.visibility shape mismatch. Expected [P_total], got ",
                render_output.visibility.sizes());
    TORCH_CHECK(render_output.visibility.scalar_type() == torch::kBool,
                "NewtonStrategy: render_output.visibility dtype mismatch. Expected Bool, got ",
                render_output.visibility.scalar_type());

    current_visibility_mask_for_model_ = render_output.visibility.to(model.get_means().device()); // Ensure it's on the same device
}


void NewtonStrategy::post_backward(int iter, gs::RenderOutput& render_output) {
    // This method is called by Trainer after loss.backward()
    // Cache necessary data for the NewtonOptimizer::step() call
    current_iter_ = iter;
    current_render_output_cache_ = render_output; // This is a shallow copy of Tensors if RenderOutput holds Tensors directly

    // Compute and cache the full visibility mask.
    // This is where the 'ranks' tensor from gsplat projection would be essential.
    // Since RenderOutput doesn't expose it, this will be a placeholder.
    compute_visibility_mask_for_model(render_output, *splat_data_);

    // Capture gradients from autograd
    // Ensure that tensors exist and have gradients. Clone to be safe.
    if (splat_data_->means().grad().defined()) {
        autograd_grad_means_ = splat_data_->means().grad().clone();
    } else {
        // Create a zero tensor of the same shape if grad is not defined.
        // This might happen if a parameter isn't part of the computation graph leading to the loss.
        std::cerr << "Warning: NewtonStrategy::post_backward - splat_data_->means().grad() is not defined. Using zeros." << std::endl;
        autograd_grad_means_ = torch::zeros_like(splat_data_->means());
    }

    // TODO: Capture gradients for other parameters (scales, rotations, opacities, shs)
    // For now, creating zero tensors as placeholders if they are needed by NewtonOptimizer::step
    // This assumes NewtonOptimizer::step will be modified to take all these grads.
    // If a parameter is not optimized by Newton, its grad might not be needed.
    if (optim_params_cache_.newton_optimize_scales && splat_data_->scaling_raw().grad().defined()) {
         autograd_grad_scales_raw_ = splat_data_->scaling_raw().grad().clone();
    } else if (optim_params_cache_.newton_optimize_scales) {
        std::cerr << "Warning: NewtonStrategy::post_backward - splat_data_->scaling_raw().grad() is not defined while newton_optimize_scales is true. Using zeros." << std::endl;
        autograd_grad_scales_raw_ = torch::zeros_like(splat_data_->scaling_raw());
    }

    if (optim_params_cache_.newton_optimize_rotations && splat_data_->rotation_raw().grad().defined()) {
        autograd_grad_rotation_raw_ = splat_data_->rotation_raw().grad().clone();
    } else if (optim_params_cache_.newton_optimize_rotations) {
        std::cerr << "Warning: NewtonStrategy::post_backward - splat_data_->rotation_raw().grad() is not defined while newton_optimize_rotations is true. Using zeros." << std::endl;
        autograd_grad_rotation_raw_ = torch::zeros_like(splat_data_->rotation_raw());
    }

    if (optim_params_cache_.newton_optimize_opacities && splat_data_->opacity_raw().grad().defined()) {
        autograd_grad_opacity_raw_ = splat_data_->opacity_raw().grad().clone();
    } else if (optim_params_cache_.newton_optimize_opacities) {
        std::cerr << "Warning: NewtonStrategy::post_backward - splat_data_->opacity_raw().grad() is not defined while newton_optimize_opacities is true. Using zeros." << std::endl;
        autograd_grad_opacity_raw_ = torch::zeros_like(splat_data_->opacity_raw());
    }

    if (optim_params_cache_.newton_optimize_shs && splat_data_->sh0().grad().defined() && splat_data_->shN().grad().defined()) {
        autograd_grad_sh0_ = splat_data_->sh0().grad().clone();
        autograd_grad_shN_ = splat_data_->shN().grad().clone();
    } else if (optim_params_cache_.newton_optimize_shs) {
         std::cerr << "Warning: NewtonStrategy::post_backward - SH grads not defined while newton_optimize_shs is true. Using zeros." << std::endl;
        if (!autograd_grad_sh0_.defined() || autograd_grad_sh0_.numel() == 0) autograd_grad_sh0_ = torch::zeros_like(splat_data_->sh0());
        if (!autograd_grad_shN_.defined() || autograd_grad_shN_.numel() == 0) autograd_grad_shN_ = torch::zeros_like(splat_data_->shN());
    }
}

void NewtonStrategy::step(int iter) {
    if (!optimizer_ || !optim_params_cache_.use_newton_optimizer) {
        TORCH_CHECK(false, "NewtonStrategy::step called without optimizer or when not enabled.");
        return;
    }
    if (!current_primary_camera_ || !current_primary_gt_image_.defined()) {
         TORCH_CHECK(false, "NewtonStrategy::step called without primary camera/GT image. Call set_current_view_data first.");
        return;
    }

    // Ensure autograd_grad_means_ is defined before passing
    if (!autograd_grad_means_.defined()) {
        TORCH_CHECK(false, "NewtonStrategy::step - autograd_grad_means_ is not defined. Ensure post_backward was called after loss.backward().");
        // Or, alternatively, provide a zero tensor if this case should be handled gracefully,
        // though it indicates a logical error in the training loop.
        // autograd_grad_means_ = torch::zeros_like(splat_data_->means());
    }
    // Ensure all necessary autograd gradients are defined before passing
    TORCH_CHECK(autograd_grad_means_.defined(), "NewtonStrategy::step - autograd_grad_means_ is not defined.");
    TORCH_CHECK(autograd_grad_scales_raw_.defined(), "NewtonStrategy::step - autograd_grad_scales_raw_ is not defined.");
    TORCH_CHECK(autograd_grad_rotation_raw_.defined(), "NewtonStrategy::step - autograd_grad_rotation_raw_ is not defined.");
    TORCH_CHECK(autograd_grad_opacity_raw_.defined(), "NewtonStrategy::step - autograd_grad_opacity_raw_ is not defined.");
    TORCH_CHECK(autograd_grad_sh0_.defined(), "NewtonStrategy::step - autograd_grad_sh0_ is not defined.");
    TORCH_CHECK(autograd_grad_shN_.defined(), "NewtonStrategy::step - autograd_grad_shN_ is not defined.");

    optimizer_->step(
        iter,
        current_visibility_mask_for_model_,
        autograd_grad_means_,
        autograd_grad_scales_raw_,
        autograd_grad_rotation_raw_,
        autograd_grad_opacity_raw_,
        autograd_grad_sh0_,
        autograd_grad_shN_,
        current_render_output_cache_,
        *current_primary_camera_,
        current_primary_gt_image_,
        current_knn_targets_gpu_
    );
}

bool NewtonStrategy::is_refining(int iter) const {
    // Basic refinement logic, can be adapted from standard 3DGS
    if (optim_params_cache_.refine_every > 0 && iter % optim_params_cache_.refine_every == 0) {
        return iter >= optim_params_cache_.start_refine && iter <= optim_params_cache_.stop_refine;
    }
    return false;
}

void NewtonStrategy::set_current_view_data(
    const Camera* primary_camera,
    const torch::Tensor& primary_gt_image,
    const gs::RenderOutput& render_output,
    const gs::param::OptimizationParameters& opt_params,
    int iteration
) {
    current_primary_camera_ = primary_camera;
    current_primary_gt_image_ = primary_gt_image;     // Assumed on device
    current_render_output_cache_ = render_output;     // Shallow copy of Tensors
    current_iter_ = iteration;
    // optim_params_cache_ should already be set by initialize()
    // Re-assert or update if dynamic changes are possible (unlikely for opt_params during training)
    // optim_params_cache_ = opt_params;

    compute_visibility_mask_for_model(render_output, *splat_data_);

    if (optim_params_cache_.use_newton_optimizer && optim_params_cache_.newton_knn_k > 0) {
        find_knn_for_current_primary(primary_camera);
    } else {
        current_knn_targets_gpu_.clear();
    }
}


void NewtonStrategy::cache_camera_references() {
    if (!train_dataset_ref_ || train_dataset_ref_->size().value_or(0) == 0) {
        std::cerr << "Warning: NewtonStrategy::cache_camera_references skipped: no training dataset provided." << std::endl;
        return;
    }
    if (!uid_to_camera_cache_.empty()){
        return; // Already initialized
    }

    uid_to_camera_cache_.clear();
    const auto& cameras_from_dataset = train_dataset_ref_->get_cameras();
    if (cameras_from_dataset.empty()) {
         std::cerr << "Warning: NewtonStrategy::cache_camera_references skipped: training dataset has no cameras." << std::endl;
        return;
    }

    for (const auto& cam_shared_ptr : cameras_from_dataset) {
        if (cam_shared_ptr) {
            uid_to_camera_cache_[cam_shared_ptr->uid()] = cam_shared_ptr.get();
        }
    }
    std::cout << "NewtonStrategy: Cached " << uid_to_camera_cache_.size() << " camera references." << std::endl;
}

void NewtonStrategy::find_knn_for_current_primary(const Camera* primary_cam_in) {
    current_knn_targets_gpu_.clear();
    if (!splat_data_ || !primary_cam_in) {
        TORCH_CHECK(false, "NewtonStrategy::find_knn_for_current_primary: SplatData or primary_cam_in is null.");
        return;
    }
    // K for KNN (number of secondary views) is determined by the precomputed KNNs in SplatData.
    // optim_params_cache_.newton_knn_k is for NewtonOptimizer's internal use (e.g. Hessian neighborhood), not for this.

    const std::vector<int>& neighbor_uids = splat_data_->get_knns_for_camera_uid(primary_cam_in->uid());

    if (neighbor_uids.empty()) {
        // No precomputed KNNs for this camera, or K was 0 during precomputation.
        return;
    }

    current_knn_targets_gpu_.reserve(neighbor_uids.size());

    for (int neighbor_uid : neighbor_uids) {
        if (neighbor_uid == primary_cam_in->uid()) continue; // Skip self

        auto it = uid_to_camera_cache_.find(neighbor_uid);
        if (it == uid_to_camera_cache_.end()) {
            std::cerr << "Warning: NewtonStrategy: KNN UID " << neighbor_uid
                      << " not found in cached camera references. Skipping." << std::endl;
            continue;
        }
        const Camera* secondary_cam = it->second;

        // Load GT image for this secondary_cam
        // This assumes Camera has a method to load its image.
        // We need to determine the target resolution for secondary GT images.
        int target_height = static_cast<int>(secondary_cam->image_height() * optim_params_cache_.newton_secondary_target_downsample_factor);
        int target_width = static_cast<int>(secondary_cam->image_width() * optim_params_cache_.newton_secondary_target_downsample_factor);

        // Create a temporary camera with new dimensions for loading if resolution param in load_and_get_image is not enough
        // Or, assume load_and_get_image handles downsampling if resolution is different from native.
        // For now, let's assume load_and_get_image can take a target resolution, or we post-process.
        // The Camera class provided doesn't show a way to change its internal H/W for loading.
        // So, we load full and then downsample.

        torch::Tensor secondary_gt_cpu = const_cast<Camera*>(secondary_cam)->load_and_get_image(); // Load full res CPU

        if (secondary_gt_cpu.defined() && secondary_gt_cpu.numel() > 0) {
             if (optim_params_cache_.newton_secondary_target_downsample_factor < 1.0f &&
                 optim_params_cache_.newton_secondary_target_downsample_factor > 0.0f) {

                // Ensure it's float and on CPU for interpolate if needed, then permute
                secondary_gt_cpu = secondary_gt_cpu.to(torch::kFloat32); // Ensure float for interpolate
                if (secondary_gt_cpu.is_cuda()) secondary_gt_cpu = secondary_gt_cpu.cpu();

                torch::Tensor input_for_interpolate = secondary_gt_cpu.permute({2,0,1}).unsqueeze(0); // HWC to 1CHW

                long new_H = static_cast<long>(secondary_gt_cpu.size(0) * optim_params_cache_.newton_secondary_target_downsample_factor);
                long new_W = static_cast<long>(secondary_gt_cpu.size(1) * optim_params_cache_.newton_secondary_target_downsample_factor);

                if (new_H > 0 && new_W > 0) {
                    secondary_gt_cpu = torch::nn::functional::interpolate(
                        input_for_interpolate,
                        torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>{static_cast<int64_t>(new_H), static_cast<int64_t>(new_W)}).mode(torch::kArea)
                    ).squeeze(0).permute({1,2,0}); // 1CHW -> CHW -> HWC
                } else {
                    std::cerr << "Warning: KNN downsampled GT image for cam " << secondary_cam->uid()
                              << " resulted in zero dimension. Original H/W: "
                              << secondary_gt_cpu.size(0) << "/" << secondary_gt_cpu.size(1)
                              << ", Factor: " << optim_params_cache_.newton_secondary_target_downsample_factor << std::endl;
                    continue; // Skip this problematic one
                }
            }
            current_knn_targets_gpu_.emplace_back(secondary_cam, secondary_gt_cpu.to(splat_data_->get_means().device())); // Corrected model_ to splat_data_
        } else {
            std::cerr << "Warning: Could not load GT image for secondary KNN camera UID " << secondary_cam->uid() << std::endl;
        }
    }
}
