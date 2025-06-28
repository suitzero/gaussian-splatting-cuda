#include "core/trainer.hpp"
#include "core/rasterizer.hpp"
#include "kernels/fused_ssim.cuh"
#include "visualizer/detail.hpp"
#include <chrono>
#include <iostream>
#include <numeric>
#include <torch/torch.h>

namespace gs {

    static inline torch::Tensor ensure_4d(const torch::Tensor& image) {
        return image.dim() == 3 ? image.unsqueeze(0) : image;
    }

    void Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return;
        }

        bilateral_grid_ = std::make_unique<gs::BilateralGrid>(
            train_dataset_size_,
            device_, // Pass the configured device
            params_.optimization.bilateral_grid_X,
            params_.optimization.bilateral_grid_Y,
            params_.optimization.bilateral_grid_W);

        // The optimizer will work with parameters on the correct device
        bilateral_grid_optimizer_ = std::make_unique<torch::optim::Adam>(
            std::vector<torch::Tensor>{bilateral_grid_->parameters()},
            torch::optim::AdamOptions(params_.optimization.bilateral_grid_lr)
                .eps(1e-15));
    }

    torch::Tensor Trainer::compute_loss(const RenderOutput& render_output,
                                        const torch::Tensor& gt_image,
                                        const SplatData& splatData,
                                        const param::OptimizationParameters& opt_params) {
        // Ensure images have same dimensions
        torch::Tensor rendered = render_output.image;
        torch::Tensor gt = gt_image;

        // Ensure both tensors are 4D (batch, height, width, channels)
        rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
        gt = gt.dim() == 3 ? gt.unsqueeze(0) : gt;

        TORCH_CHECK(rendered.sizes() == gt.sizes(), "ERROR: size mismatch – rendered ", rendered.sizes(), " vs. ground truth ", gt.sizes());

        // Base loss: L1 + SSIM
        auto l1_loss = torch::l1_loss(rendered, gt);

        // fused_ssim expects NCHW format [Batch, Channels, Height, Width]
        // Assuming 'rendered' and 'gt' are currently NHWC [Batch, Height, Width, Channels]
        // (Common from image loading/rasterization)
        torch::Tensor rendered_nchw = rendered.permute({0, 3, 1, 2}).contiguous();
        torch::Tensor gt_nchw = gt.permute({0, 3, 1, 2}).contiguous();

        auto ssim_loss = 1.f - fused_ssim(rendered_nchw, gt_nchw, "valid", /*train=*/true);
        torch::Tensor loss = (1.f - opt_params.lambda_dssim) * l1_loss +
                             opt_params.lambda_dssim * ssim_loss;

        // Regularization terms
        if (opt_params.opacity_reg > 0.0f) {
            auto opacity_l1 = torch::abs(splatData.get_opacity()).mean();
            loss += opt_params.opacity_reg * opacity_l1;
        }

        if (opt_params.scale_reg > 0.0f) {
            auto scale_l1 = torch::abs(splatData.get_scaling()).mean();
            loss += opt_params.scale_reg * scale_l1;
        }
        // Total variation loss for bilateral grid
        if (params_.optimization.use_bilateral_grid) {
            loss += params_.optimization.tv_loss_weight * bilateral_grid_->tv_loss();
        }

        return loss;
    }

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     const param::TrainingParameters& params)
        : strategy_(std::move(strategy)),
          params_(params) {

        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        // Handle dataset split based on evaluation flag
        if (params.optimization.enable_eval) {
            // Create train/val split
            train_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(), params.dataset, CameraDataset::Split::TRAIN);
            val_dataset_ = std::make_shared<CameraDataset>(
                dataset->get_cameras(), params.dataset, CameraDataset::Split::VAL);

            std::cout << "Created train/val split: "
                      << train_dataset_->size().value() << " train, "
                      << val_dataset_->size().value() << " val images" << std::endl;
        } else {
            // Use all images for training
            train_dataset_ = dataset;
            val_dataset_ = nullptr;

            std::cout << "Using all " << train_dataset_->size().value()
                      << " images for training (no evaluation)" << std::endl;
        }

        train_dataset_size_ = train_dataset_->size().value();

        // Initialize and validate the CUDA device
        int device_id = params.optimization.cuda_device_id;
        if (torch::cuda::is_available()) {
            int num_cuda_devices = torch::cuda::device_count();
            if (device_id < 0 || device_id >= num_cuda_devices) {
                std::cerr << "Warning: Invalid cuda_device_id " << device_id
                          << ". Available devices: 0-" << (num_cuda_devices - 1)
                          << ". Defaulting to device 0." << std::endl;
                device_id = 0;
            }
            device_ = torch::Device(torch::kCUDA, device_id);
            std::cout << "Using CUDA device: " << device_id << " (" << torch::cuda::get_device_name(device_id) << ")" << std::endl;
        } else {
            // This was already checked at the beginning of the constructor, but for safety:
            throw std::runtime_error("CUDA is not available – aborting. (Secondary check)");
        }

        strategy_->initialize(params.optimization);

        // Initialize bilateral grid if enabled
        initialize_bilateral_grid(); // This might need to be device-aware if it creates CUDA tensors

        background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32));
        background_ = background_.to(device_); // Use configured device

        progress_ = std::make_unique<TrainingProgress>(
            params.optimization.iterations,
            /*bar_width=*/100);

        // Initialize the evaluator - it handles all metrics internally
        evaluator_ = std::make_unique<metrics::MetricsEvaluator>(params);

        // Print render mode configuration
        std::cout << "Render mode: " << params.optimization.render_mode << std::endl;

        std::cout << "Visualization: " << (params.optimization.enable_viz ? "enabled" : "disabled") << std::endl;
    }

    Trainer::~Trainer() {
        // Ensure training is stopped
        stop_requested_ = true;
    }

    GSViewer* Trainer::create_and_get_viewer() {
        if (!params_.optimization.enable_viz) {
            return nullptr;
        }

        if (!viewer_) {
            viewer_ = std::make_unique<GSViewer>("GS-CUDA", 1280, 720);
            viewer_->setTrainer(this);
        }

        return viewer_.get();
    }

    void Trainer::handle_control_requests(int iter) {
        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            progress_->pause();
            std::cout << "\nTraining paused at iteration " << iter << std::endl;
            std::cout << "Click 'Resume Training' to continue." << std::endl;
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            progress_->resume(iter, current_loss_, static_cast<int>(strategy_->get_model().size()));
            std::cout << "\nTraining resumed at iteration " << iter << std::endl;
        }

        // Handle save request
        if (save_requested_.load()) {
            save_requested_ = false;
            std::cout << "\nSaving checkpoint at iteration " << iter << "..." << std::endl;
            strategy_->get_model().save_ply(params_.dataset.output_path / "checkpoints", iter, /*join=*/true);
            std::cout << "Checkpoint saved to " << (params_.dataset.output_path / "checkpoints").string() << std::endl;
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            std::cout << "\nStopping training permanently at iteration " << iter << "..." << std::endl;
            std::cout << "Saving final model..." << std::endl;
            strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    // bool Trainer::train_step(int iter, Camera* cam, torch::Tensor gt_image, RenderMode render_mode) {
    bool Trainer::train_step(int iter, std::vector<CameraWithImage>& batch_data, RenderMode render_mode) {
        current_iteration_ = iter;

        if (batch_data.empty()) {
            // Or log a warning, or return true to not stop training if this is recoverable
            std::cerr << "Warning: train_step received empty batch_data at iteration " << iter << std::endl;
            return true;
        }

        // Check control requests at the beginning
        handle_control_requests(iter);

        // If stop requested, return false to end training
        if (stop_requested_) {
            return false;
        }

        // If paused, wait
        while (is_paused_ && !stop_requested_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            handle_control_requests(iter);
        }

        // Check stop again after potential pause
        if (stop_requested_) {
            return false;
        }

        std::vector<torch::Tensor> rendered_images_list;
        rendered_images_list.reserve(batch_data.size());
        std::vector<torch::Tensor> gt_images_list;
        gt_images_list.reserve(batch_data.size());

        RenderOutput last_r_output; // For strategy and potentially other single-instance needs

        for (CameraWithImage& item : batch_data) {
            Camera* cam = item.camera;
            torch::Tensor current_gt_image = std::move(item.image);

            if (params_.optimization.accelerate_data_loading) {
                current_gt_image = current_gt_image.to(device_, /*non_blocking=*/true);
            } else {
                current_gt_image = current_gt_image.to(device_); // Ensure on configured CUDA device
            }

            auto render_fn_item = [this, &cam, render_mode]() {
                return gs::rasterize(
                    *cam,
                    strategy_->get_model(),
                    background_,
                    1.0f,
                    false,
                    false,
                    render_mode);
            };

            RenderOutput r_output_item;
            if (viewer_) {
                std::lock_guard<std::mutex> lock(viewer_->splat_mtx_);
                r_output_item = render_fn_item();
            } else {
                r_output_item = render_fn_item();
            }

            if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                r_output_item.image = bilateral_grid_->apply(r_output_item.image, cam->uid());
            }

            rendered_images_list.push_back(r_output_item.image);
            gt_images_list.push_back(current_gt_image);
            if (&item == &batch_data.back()) { // Check if it's the last item
                last_r_output = r_output_item;
            }
        }

        if (rendered_images_list.empty()) {
             std::cerr << "Warning: No images rendered in batch at iteration " << iter << std::endl;
            return true; // Continue training, maybe log this
        }

        torch::Tensor batched_rendered_images = torch::stack(rendered_images_list);
        torch::Tensor batched_gt_images = torch::stack(gt_images_list);

        // Construct a RenderOutput for compute_loss. It mainly uses .image.
        // Other fields like valid_mask, depth, alpha are not used by compute_loss directly.
        // If they were, this would need more careful handling for batching.
        RenderOutput batched_r_output;
        batched_r_output.image = batched_rendered_images;
        // If other fields from last_r_output are needed by strategy or other parts,
        // they are available in last_r_output. For compute_loss, only .image is used.

        torch::Tensor loss = compute_loss(batched_r_output, // Pass RenderOutput with batched image
                                          batched_gt_images,
                                          strategy_->get_model(),
                                          params_.optimization);

        current_loss_ = loss.item<float>();
        loss.backward();

        {
            torch::NoGradGuard no_grad;

            // Evaluation: Note that val_dataset_ is not batched here.
            // evaluator_->evaluate might need adjustment if it expects batching or if we want to eval full batches.
            // For now, it will use its internal dataloading which is likely single image.
            if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                evaluator_->print_evaluation_header(iter);
                auto metrics = evaluator_->evaluate(iter,
                                                    strategy_->get_model(),
                                                    val_dataset_, // val_dataset_ is not changed by batching train data
                                                    background_);
                std::cout << metrics.to_string() << std::endl;
            }

            // Save model at specified steps
            for (size_t save_step : params_.optimization.save_steps) {
                if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                    const bool join_threads = (iter == params_.optimization.save_steps.back());
                    strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/join_threads);
                }
            }

            // The strategy part uses last_r_output from the loop.
            // This might be an approximation if the strategy needs batched info.
            auto do_strategy = [&]() {
                strategy_->post_backward(iter, last_r_output);
                strategy_->step(iter);
            };

            if (viewer_) {
                std::lock_guard<std::mutex> lock(viewer_->splat_mtx_);
                do_strategy();
            } else {
                do_strategy();
            }

            if (params_.optimization.use_bilateral_grid) {
                // This optimizer is for the bilateral_grid_ parameters itself,
                // which are shared across all items. So, one step is correct.
                bilateral_grid_optimizer_->step();
                bilateral_grid_optimizer_->zero_grad(true);
            }
        }

        progress_->update(iter, loss.item<float>(),
                          static_cast<int>(strategy_->get_model().size()),
                          strategy_->is_refining(iter));

        if (viewer_) {
            if (viewer_->info_) {
                auto& info = viewer_->info_;
                std::lock_guard<std::mutex> lock(viewer_->info_->mtx);
                info->updateProgress(iter, params_.optimization.iterations);
                info->updateNumSplats(static_cast<size_t>(strategy_->get_model().size()));
                info->updateLoss(loss.item<float>());
            }

            if (viewer_->notifier_) {
                auto& notifier = viewer_->notifier_;
                std::unique_lock<std::mutex> lock(notifier->mtx);
                notifier->cv.wait(lock, [&notifier] { return notifier->ready; });
            }
        }

        // Return true if we should continue training
        return iter < params_.optimization.iterations && !stop_requested_;
    }

    void Trainer::train() {
        is_running_ = false; // Don't start running until notified
        training_complete_ = false;

        // Wait for the start signal from GUI if visualization is enabled
        if (viewer_ && viewer_->notifier_) {
            auto& notifier = viewer_->notifier_;
            std::unique_lock<std::mutex> lock(notifier->mtx);
            notifier->cv.wait(lock, [&notifier] { return notifier->ready; });
        }

        is_running_ = true; // Now we can start

        int iter = 1;
        const int epochs_needed = (params_.optimization.iterations + train_dataset_size_ - 1) / train_dataset_size_;

        const int num_workers = 4; // This could also be made configurable later if needed
        const int batch_size = params_.optimization.batch_size;

        const RenderMode render_mode = stringToRenderMode(params_.optimization.render_mode);

        bool should_continue = true;

        for (int epoch = 0; epoch < epochs_needed && should_continue; ++epoch) {
            auto train_dataloader = create_dataloader_from_dataset(train_dataset_, batch_size, num_workers);

            for (auto& batched_examples : *train_dataloader) {
                // batched_examples is std::vector<torch::data::Example<CameraWithImage, torch::Tensor>>
                std::vector<CameraWithImage> current_batch_data;
                current_batch_data.reserve(batched_examples.size());
                for (auto& example : batched_examples) {
                    current_batch_data.push_back(std::move(example.data));
                }

                // The gt_image.to(torch::kCUDA) call is removed from here.
                // It will be handled inside train_step for each image if accelerate_data_loading is true.

                // train_step will be modified in the next step to accept std::vector<CameraWithImage>
                // For now, this call will be a compile error until train_step's signature is updated.
                should_continue = train_step(iter, current_batch_data, render_mode);

                if (!should_continue) {
                    break;
                }

                ++iter;
            }
        }

        // Final save if not already saved by stop request
        if (!stop_requested_) {
            strategy_->get_model().save_ply(params_.dataset.output_path, iter, /*join=*/true);
        }

        progress_->complete();
        evaluator_->save_report();
        progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));

        is_running_ = false;
        training_complete_ = true;
    }

} // namespace gs