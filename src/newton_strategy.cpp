#include "core/newton_strategy.hpp"
#include "Ops.h"
#include "core/debug_utils.hpp"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include "kernels/fused_ssim.cuh"
#include <c10/cuda/CUDACachingAllocator.h>
#include <exception>
#include <iostream>
#include <random>

void NewtonStrategy::ExponentialLR::step() {
    if (param_group_index_ >= 0) {
        auto& group = optimizer_.param_groups()[param_group_index_];

        // Try to cast to our custom Options first
        if (auto* adam_options = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
            double current_lr = adam_options->lr();
            adam_options->lr(current_lr * gamma_);
        }
    } else {
        // Update all param groups
        for (auto& group : optimizer_.param_groups()) {
            if (auto* adam_options = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
                double current_lr = adam_options->lr();
                adam_options->lr(current_lr * gamma_);
            }
        }
    }
}

NewtonStrategy::NewtonStrategy(SplatData&& splat_data)
    : _splat_data(std::move(splat_data)) {
}

torch::Tensor NewtonStrategy::multinomial_sample(const torch::Tensor& weights, int n, bool replacement) {
    const int64_t num_elements = weights.size(0);

    // PyTorch's multinomial has a limit of 2^24 elements
    if (num_elements <= (1 << 24)) {
        return torch::multinomial(weights, n, replacement);
    } else {
        // For larger arrays, we need to implement sampling manually
        auto weights_normalized = weights / weights.sum();
        auto weights_cpu = weights_normalized.cpu();

        std::vector<int64_t> sampled_indices;
        sampled_indices.reserve(n);

        // Create cumulative distribution
        auto cumsum = weights_cpu.cumsum(0);
        auto cumsum_data = cumsum.accessor<float, 1>();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0, 1.0);

        for (int i = 0; i < n; ++i) {
            float u = dis(gen);
            // Binary search for the index
            int64_t idx = 0;
            int64_t left = 0, right = num_elements - 1;
            while (left <= right) {
                int64_t mid = (left + right) / 2;
                if (cumsum_data[mid] < u) {
                    left = mid + 1;
                } else {
                    idx = mid;
                    right = mid - 1;
                }
            }
            sampled_indices.push_back(idx);
        }

        auto result = torch::tensor(sampled_indices, torch::kLong);
        return result.to(weights.device());
    }
}

void NewtonStrategy::update_optimizer_for_relocate(torch::optim::Optimizer* optimizer,
                                                   const torch::Tensor& sampled_indices,
                                                   const torch::Tensor& dead_indices,
                                                   int param_position) {
    // Get the parameter
    auto& param = optimizer->param_groups()[param_position].params()[0];
    void* param_key = param.unsafeGetTensorImpl();

    // Check if optimizer state exists
    auto state_it = optimizer->state().find(param_key);
    if (state_it == optimizer->state().end()) {
        // No state exists yet - this can happen if optimizer.step() hasn't been called
        // In this case, there's nothing to reset, so we can safely return
        return;
    }

    // Get the optimizer state - handle both Adam types
    auto& param_state = *state_it->second;

    if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(&param_state)) {
        // Standard Adam
        adam_state->exp_avg().index_put_({sampled_indices}, 0);
        adam_state->exp_avg_sq().index_put_({sampled_indices}, 0);

        if (adam_state->max_exp_avg_sq().defined()) {
            adam_state->max_exp_avg_sq().index_put_({sampled_indices}, 0);
        }
    }
}

int NewtonStrategy::relocate_gs() {
    // Get opacities and handle both [N] and [N, 1] shapes
    torch::NoGradGuard no_grad;
    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto dead_mask = opacities <= _params->min_opacity;
    auto dead_indices = dead_mask.nonzero().squeeze(-1);
    int n_dead = dead_indices.numel();

    if (n_dead == 0)
        return 0;

    auto alive_mask = ~dead_mask;
    auto alive_indices = alive_mask.nonzero().squeeze(-1);

    if (alive_indices.numel() == 0)
        return 0;

    // Sample from alive Gaussians based on opacity
    auto probs = opacities.index_select(0, alive_indices);
    auto sampled_idxs_local = multinomial_sample(probs, n_dead, true);
    auto sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);

    // Get parameters for sampled Gaussians
    auto sampled_opacities = opacities.index_select(0, sampled_idxs);
    auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

    // Count occurrences of each sampled index
    auto ratios = torch::zeros({opacities.size(0)}, torch::kFloat32).to(torch::kCUDA);
    ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kFloat32));
    ratios = ratios.index_select(0, sampled_idxs) + 1;

    // IMPORTANT: Clamp and convert to int as in Python implementation
    const int n_max = static_cast<int>(_binoms.size(0));
    ratios = torch::clamp(ratios, 1, n_max);
    ratios = ratios.to(torch::kInt32).contiguous(); // Convert to int!

    // Call the CUDA relocation function from gsplat
    auto relocation_result = gsplat::relocation(
        sampled_opacities,
        sampled_scales,
        ratios,
        _binoms,
        n_max);

    auto new_opacities = std::get<0>(relocation_result);
    auto new_scales = std::get<1>(relocation_result);

    // Clamp new opacities
    new_opacities = torch::clamp(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

    // Update parameters for sampled indices
    // Handle opacity shape properly
    if (_splat_data.opacity_raw().dim() == 2) {
        _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                             torch::logit(new_opacities).unsqueeze(-1));
    } else {
        _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
    }
    _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

    // Copy from sampled to dead indices
    _splat_data.means().index_put_({dead_indices}, _splat_data.means().index_select(0, sampled_idxs));
    _splat_data.sh0().index_put_({dead_indices}, _splat_data.sh0().index_select(0, sampled_idxs));
    _splat_data.shN().index_put_({dead_indices}, _splat_data.shN().index_select(0, sampled_idxs));
    _splat_data.scaling_raw().index_put_({dead_indices}, _splat_data.scaling_raw().index_select(0, sampled_idxs));
    _splat_data.rotation_raw().index_put_({dead_indices}, _splat_data.rotation_raw().index_select(0, sampled_idxs));
    _splat_data.opacity_raw().index_put_({dead_indices}, _splat_data.opacity_raw().index_select(0, sampled_idxs));

    // Update optimizer states for sampled indices
    for (int i = 0; i < 6; ++i) {
        update_optimizer_for_relocate(_optimizer.get(), sampled_idxs, dead_indices, i);
    }

    return n_dead;
}

int NewtonStrategy::add_new_gs() {
    // Add this check at the beginning
    torch::NoGradGuard no_grad;
    if (!_optimizer) {
        std::cerr << "Warning: add_new_gs called but optimizer not initialized" << std::endl;
        return 0;
    }

    const int current_n = _splat_data.size();
    const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
    const int n_new = std::max(0, n_target - current_n);

    if (n_new == 0)
        return 0;

    // Get opacities and handle both [N] and [N, 1] shapes
    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto probs = opacities.flatten();
    auto sampled_idxs = multinomial_sample(probs, n_new, true);

    // Get parameters for sampled Gaussians
    auto sampled_opacities = opacities.index_select(0, sampled_idxs);
    auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

    // Count occurrences
    auto ratios = torch::zeros({opacities.size(0)}, torch::kFloat32).to(torch::kCUDA);
    ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kFloat32));
    ratios = ratios.index_select(0, sampled_idxs) + 1;

    // IMPORTANT: Clamp and convert to int as in Python implementation
    const int n_max = static_cast<int>(_binoms.size(0));
    ratios = torch::clamp(ratios, 1, n_max);
    ratios = ratios.to(torch::kInt32).contiguous(); // Convert to int!

    // Call the CUDA relocation function from gsplat
    auto relocation_result = gsplat::relocation(
        sampled_opacities,
        sampled_scales,
        ratios,
        _binoms,
        n_max);

    auto new_opacities = std::get<0>(relocation_result);
    auto new_scales = std::get<1>(relocation_result);

    // Clamp new opacities
    new_opacities = torch::clamp(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

    // Update existing Gaussians FIRST (before concatenation)
    if (_splat_data.opacity_raw().dim() == 2) {
        _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                             torch::logit(new_opacities).unsqueeze(-1));
    } else {
        _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
    }
    _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

    // Prepare new Gaussians to concatenate
    auto new_means = _splat_data.means().index_select(0, sampled_idxs);
    auto new_sh0 = _splat_data.sh0().index_select(0, sampled_idxs);
    auto new_shN = _splat_data.shN().index_select(0, sampled_idxs);
    auto new_scaling = _splat_data.scaling_raw().index_select(0, sampled_idxs);
    auto new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs);
    auto new_opacity = _splat_data.opacity_raw().index_select(0, sampled_idxs);

    // Step 1: Concatenate all parameters
    auto concat_means = torch::cat({_splat_data.means(), new_means}, 0).set_requires_grad(true);
    auto concat_sh0 = torch::cat({_splat_data.sh0(), new_sh0}, 0).set_requires_grad(true);
    auto concat_shN = torch::cat({_splat_data.shN(), new_shN}, 0).set_requires_grad(true);
    auto concat_scaling = torch::cat({_splat_data.scaling_raw(), new_scaling}, 0).set_requires_grad(true);
    auto concat_rotation = torch::cat({_splat_data.rotation_raw(), new_rotation}, 0).set_requires_grad(true);
    auto concat_opacity = torch::cat({_splat_data.opacity_raw(), new_opacity}, 0).set_requires_grad(true);

    // Step 2: SAFER optimizer state update
    // Store the new parameters in a temporary array first
    std::array<torch::Tensor*, 6> new_params = {
        &concat_means, &concat_sh0, &concat_shN,
        &concat_scaling, &concat_rotation, &concat_opacity};

    // Collect old parameter keys and states
    std::vector<void*> old_param_keys;
    std::vector<std::unique_ptr<torch::optim::OptimizerParamState>> saved_states;

    for (int i = 0; i < 6; ++i) {
        auto& old_param = _optimizer->param_groups()[i].params()[0];
        void* old_key = old_param.unsafeGetTensorImpl();
        old_param_keys.push_back(old_key);

        // Check if state exists
        auto state_it = _optimizer->state().find(old_key);
        if (state_it != _optimizer->state().end()) {
            // Clone the state before modifying - handle both optimizer types
            if (auto* adam_state = dynamic_cast<torch::optim::AdamParamState*>(state_it->second.get())) {
                // Standard Adam state
                torch::IntArrayRef new_shape;
                if (i == 0)
                    new_shape = new_means.sizes();
                else if (i == 1)
                    new_shape = new_sh0.sizes();
                else if (i == 2)
                    new_shape = new_shN.sizes();
                else if (i == 3)
                    new_shape = new_scaling.sizes();
                else if (i == 4)
                    new_shape = new_rotation.sizes();
                else
                    new_shape = new_opacity.sizes();

                auto zeros_to_add = torch::zeros(new_shape, adam_state->exp_avg().options());
                auto new_exp_avg = torch::cat({adam_state->exp_avg(), zeros_to_add}, 0);
                auto new_exp_avg_sq = torch::cat({adam_state->exp_avg_sq(), zeros_to_add}, 0);

                // Create new state
                auto new_state = std::make_unique<torch::optim::AdamParamState>();
                new_state->step(adam_state->step());
                new_state->exp_avg(new_exp_avg);
                new_state->exp_avg_sq(new_exp_avg_sq);
                if (adam_state->max_exp_avg_sq().defined()) {
                    auto new_max_exp_avg_sq = torch::cat({adam_state->max_exp_avg_sq(), zeros_to_add}, 0);
                    new_state->max_exp_avg_sq(new_max_exp_avg_sq);
                }

                saved_states.push_back(std::move(new_state));
            } else {
                saved_states.push_back(nullptr);
            }
        } else {
            saved_states.push_back(nullptr);
        }
    }

    // Now remove all old states
    for (auto key : old_param_keys) {
        _optimizer->state().erase(key);
    }

    // Update parameters and add new states
    for (int i = 0; i < 6; ++i) {
        _optimizer->param_groups()[i].params()[0] = *new_params[i];

        if (saved_states[i]) {
            void* new_param_key = new_params[i]->unsafeGetTensorImpl();
            _optimizer->state()[new_param_key] = std::move(saved_states[i]);
        }
    }

    // Step 3: Finally update the model's parameters
    _splat_data.means() = concat_means;
    _splat_data.sh0() = concat_sh0;
    _splat_data.shN() = concat_shN;
    _splat_data.scaling_raw() = concat_scaling;
    _splat_data.rotation_raw() = concat_rotation;
    _splat_data.opacity_raw() = concat_opacity;

    return n_new;
}

void NewtonStrategy::inject_noise(int iter) {
    // Get opacities and handle both [N] and [N, 1] shapes
    torch::NoGradGuard no_grad;

    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto scales = _splat_data.get_scaling();
    auto quats = _splat_data.get_rotation();

    // Use gsplat's quat_scale_to_covar_preci function
    auto covar_result = gsplat::quat_scale_to_covar_preci_fwd(
        quats,
        scales,
        true,  // compute_covar
        false, // compute_preci
        false  // triu
    );
    auto covars = std::get<0>(covar_result); // [N, 3, 3]

    // Opacity sigmoid function: 1 / (1 + exp(-k * (x - x0)))
    const float k = 100.0f;
    const float x0 = 0.995f;
    auto op_sigmoid = 1.0f / (1.0f + torch::exp(-k * ((1.0f - opacities) - x0)));

    // Get current learning rate from optimizer (after scheduler has updated it)
    float current_lr = 0.0f;
    auto& group = _optimizer->param_groups()[0];
    if (auto* adam_options = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
        current_lr = static_cast<float>(adam_options->lr());
    }

    // Generate noise
    auto noise = torch::randn_like(_splat_data.means()) * op_sigmoid.unsqueeze(-1) * current_lr * _noise_lr;

    // Transform noise by covariance
    noise = torch::bmm(covars, noise.unsqueeze(-1)).squeeze(-1);

    // Add noise to positions
    _splat_data.means().add_(noise);
}

void NewtonStrategy::post_backward(int iter, gs::RenderOutput& render_output) {
    std::vector<torch::Tensor> delta_p = conjugate_gradient();

    // Log grad and update norms for diagnosis
    float grad_norm = 0.0f;
    float update_norm = 0.0f;
    for (size_t i = 0; i < _current_grads.size(); ++i) {
        if (_current_grads[i].defined()) {
            grad_norm += _current_grads[i].norm().item<float>();
        }
        if (delta_p[i].defined()) {
            update_norm += delta_p[i].norm().item<float>();
        }
    }
    //std::cout << "Iter " << iter << ": Grad norm = " << grad_norm << ", Update norm = " << update_norm << std::endl;

    // Temporarily set gradients to delta_p for optimizer update
    std::vector<torch::Tensor> original_grads;
    for (size_t i = 0; i < _current_params_list.size(); ++i) {
        if (_current_params_list[i].grad().defined()) {
            original_grads.push_back(_current_params_list[i].grad().clone());
            _current_params_list[i].grad().copy_(delta_p[i]);
        } else {
            original_grads.push_back(torch::Tensor());
        }
    }

    torch::NoGradGuard no_grad;
    if (iter % _params->sh_degree_interval == 0) {
        _splat_data.increment_sh_degree();
    }

    // Moved densification after optimizer step to apply updates before modifying parameters
    // Now called in step() after optimizer.step()

    inject_noise(iter);

    // Refresh _current_params_list after potential densification
    // But since densification is moved, refresh if needed
    _current_params_list.clear();
    for (auto& group : _optimizer->param_groups()) {
        if (!group.params().empty()) {
            _current_params_list.push_back(group.params()[0]);
        }
    }
}

void NewtonStrategy::step(int iter) {
    if (iter < _params->iterations) {
        _optimizer->step();
        _optimizer->zero_grad(true);
        _scheduler->step();

        // Perform densification after optimizer step and scheduler
        torch::NoGradGuard no_grad;
        if (is_refining(iter)) {
            relocate_gs();
            if (add_new_gs() > 0) {
                // After adding new GS, refresh param list and attach any pending grads if needed
                // But since we zero_grad above, and densify under no_grad, no grads to attach
                // If you have pending delta_p or something, handle here - but in this case, it's post-update
                c10::cuda::CUDACachingAllocator::emptyCache();

                // Refresh param list after densification
                _current_params_list.clear();
                for (auto& group : _optimizer->param_groups()) {
                    if (!group.params().empty()) {
                        _current_params_list.push_back(group.params()[0]);
                    }
                }
            }
        }
    }
}
void NewtonStrategy::initialize(const gs::param::OptimizationParameters& optimParams) {
    _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

    const auto dev = torch::kCUDA;
    _splat_data.means() = _splat_data.means().to(dev).set_requires_grad(true);
    _splat_data.scaling_raw() = _splat_data.scaling_raw().to(dev).set_requires_grad(true);
    _splat_data.rotation_raw() = _splat_data.rotation_raw().to(dev).set_requires_grad(true);
    _splat_data.opacity_raw() = _splat_data.opacity_raw().to(dev).set_requires_grad(true);
    _splat_data.sh0() = _splat_data.sh0().to(dev).set_requires_grad(true);
    _splat_data.shN() = _splat_data.shN().to(dev).set_requires_grad(true);

    // Initialize binomial coefficients
    const int n_max = 51;
    _binoms = torch::zeros({n_max, n_max}, torch::kFloat32);
    auto binoms_accessor = _binoms.accessor<float, 2>();
    for (int n = 0; n < n_max; ++n) {
        for (int k = 0; k <= n; ++k) {
            // Compute binomial coefficient C(n,k)
            float binom = 1.0f;
            for (int i = 0; i < k; ++i) {
                binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
            }
            binoms_accessor[n][k] = binom;
        }
    }
    _binoms = _binoms.to(dev);

    // Initialize optimizer

    using torch::optim::AdamOptions;
    std::vector<torch::optim::OptimizerParamGroup> groups;

    // Calculate initial learning rate for position
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.means()},
                                                          std::make_unique<AdamOptions>(_params->means_lr * _splat_data.get_scene_scale())));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.sh0()},
                                                          std::make_unique<AdamOptions>(_params->shs_lr)));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.shN()},
                                                          std::make_unique<AdamOptions>(_params->shs_lr / 20.f)));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.scaling_raw()},
                                                          std::make_unique<AdamOptions>(_params->scaling_lr)));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.rotation_raw()},
                                                          std::make_unique<AdamOptions>(_params->rotation_lr)));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.opacity_raw()},
                                                          std::make_unique<AdamOptions>(_params->opacity_lr)));

    for (auto& g : groups)
        static_cast<AdamOptions&>(g.options()).eps(1e-15);

    _optimizer = std::make_unique<torch::optim::Adam>(groups, AdamOptions(0.f).eps(1e-15));

    // Initialize exponential scheduler
    // Python: gamma = 0.01^(1/max_steps)
    // This means after max_steps, lr will be 0.01 * initial_lr
    const double gamma = std::pow(0.01, 1.0 / _params->iterations);
    _scheduler = std::make_unique<ExponentialLR>(*_optimizer, gamma, 0);
}

bool NewtonStrategy::is_refining(int iter) const {
    return (iter < _params->stop_refine &&
            iter > _params->start_refine &&
            iter % _params->refine_every == 0);
}

std::vector<torch::Tensor> NewtonStrategy::conjugate_gradient() {
    if (_current_grads.empty() || !_current_grads[0].defined())
        throw std::runtime_error("Gradients not computed");
    auto b = _current_grads;
    int max_iter = 50;
    float tol = 1e-5;
    float damping = 0.1;
    auto Hv_func = [&](const std::vector<torch::Tensor>& v) -> std::vector<torch::Tensor> {
        float eps = 1e-6f; // Reduced eps for better numerical stability
        auto grad_p = compute_perturbed_grad(v, eps);

        std::vector<torch::Tensor> hvp;
        for (int i = 0; i < v.size(); ++i) {
            auto diff = (grad_p[i] - _current_grads[i]) / eps;
            diff += damping * v[i];
            hvp.push_back(diff);
        }
        return hvp;
    };
    std::vector<torch::Tensor> x;
    for (const auto& bi : b) {
        x.push_back(torch::zeros_like(bi).to(torch::kCUDA));
    }

    auto r = b;
    auto p = r;
    auto rs_old = torch::tensor(0.0, b[0].options());
    for (const auto& ri : r) {
        if (ri.defined()) {
            rs_old += torch::norm(ri).pow(2).sum();
        }
    }

    int cg_iters = 0;
    for (int i = 0; i < max_iter; ++i) {
        cg_iters++;
        auto Ap = Hv_func(p);
        auto pAp = torch::tensor(0.0, b[0].options());
        for (size_t j = 0; j < p.size(); ++j) {
            if (p[j].defined() && Ap[j].defined()) {
                pAp += torch::sum(p[j] * Ap[j]);
            }
        }

        if (pAp.item<float>() <= 0) {
            damping *= 2.0f;
            continue;
        }

        auto alpha = rs_old / pAp;

        for (size_t j = 0; j < x.size(); ++j) {
            if (x[j].defined()) {
                x[j] += alpha * p[j];
                r[j] -= alpha * Ap[j];
            }
        }

        auto rs_new = torch::tensor(0.0, b[0].options());
        for (const auto& ri : r) {
            if (ri.defined()) {
                rs_new += torch::sum(ri * ri);
            }
        }

        if (rs_new.sqrt().item<double>() < tol) {
            break;
        }

        auto beta = rs_new / rs_old;
        for (size_t j = 0; j < p.size(); ++j) {
            if (p[j].defined()) {
                p[j] = r[j] + beta * p[j];
            }
        }

        rs_old = rs_new;
    }

    // Log CG iters for diagnosis
    //std::cout << "CG iterations: " << cg_iters << ", Final damping: " << damping << std::endl;

    return x;
}

torch::Tensor NewtonStrategy::compute_loss(Camera* viewpoint_camera, const gs::RenderOutput& render_output, const torch::Tensor& gt_image, bool update_members) {
    _cam = viewpoint_camera;
    _gt_image = gt_image;

    torch::Tensor rendered = render_output.image;
    rendered = rendered.dim() == 3 ? rendered.unsqueeze(0) : rendered;
    torch::Tensor gt = gt_image.dim() == 3 ? gt_image.unsqueeze(0) : gt_image;
    TORCH_CHECK(rendered.sizes() == gt.sizes(), "ERROR: size mismatch – rendered ", rendered.sizes(), " vs. ground truth ", gt.sizes());

    auto ssim_loss = 1.f - fused_ssim(rendered, gt, "valid", /*train=*/true);
    auto l2_loss = torch::mse_loss(rendered, gt);
    torch::Tensor loss = (1.f - _params->lambda_dssim) * l2_loss + _params->lambda_dssim * ssim_loss;

    if (_params->opacity_reg > 0.0f) {
        auto opacity_l1 = torch::abs(_splat_data.get_opacity()).mean();
        loss += _params->opacity_reg * opacity_l1;
    }

    if (_params->scale_reg > 0.0f) {
        auto scale_l1 = torch::abs(_splat_data.get_scaling()).mean();
        loss += _params->scale_reg * scale_l1;
    }

    if (update_members) {
        _current_loss = loss;
        loss.backward();
        _current_params_list.clear();
        for (auto& group : _optimizer->param_groups()) {
            if (!group.params().empty()) {
                _current_params_list.push_back(group.params()[0]);
            }
        }
        _current_grads.clear();
        for (const auto& param : _current_params_list) {
            _current_grads.push_back(param.grad());
        }
    }

    return loss;
}

std::vector<torch::Tensor> NewtonStrategy::compute_perturbed_grad(const std::vector<torch::Tensor>& perturb, float eps) {
    std::vector<torch::Tensor> original_data;
    std::vector<float> relative_eps_list;
    for (const auto& param : _current_params_list) {
        original_data.push_back(param.clone());
    }
    {
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < perturb.size(); ++i) {
            float param_norm = _current_params_list[i].norm().item<float>() + 1e-6f; // Avoid zero
            float relative_eps = eps * param_norm;
            relative_eps_list.push_back(relative_eps);
            auto& param = _current_params_list[i];
            param.add_(perturb[i] * relative_eps);
        }
    }

    for (size_t i = 0; i < perturb.size(); ++i) {
        auto& param = _current_params_list[i];
        if (param.grad().defined()) {
            param.grad().zero_();
        }
    }

    auto background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32)).to(torch::kCUDA);
    const gs::RenderMode render_mode = gs::stringToRenderMode(_params->render_mode);
    auto render = gs::rasterize(*_cam, _splat_data, background_, 1.0f, false, false, render_mode);

    auto loss = compute_loss(_cam, render, _gt_image, false);

    std::vector<torch::Tensor> perturbed_grads(perturb.size());
    try {
        loss.backward();
        for (size_t i = 0; i < perturb.size(); ++i) {
            perturbed_grads[i] = _current_params_list[i].grad().clone();
        }
    } catch (const c10::Error& e) {
        std::cerr << "Backward failed in perturbed grad: " << e.msg() << std::endl;
        // Assume zero grads if no computation graph (e.g., no visible gaussians)
        for (size_t i = 0; i < perturb.size(); ++i) {
            perturbed_grads[i] = torch::zeros_like(_current_params_list[i].grad());
        }
    }

    {
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < perturb.size(); ++i) {
            auto& param = _current_params_list[i];
            param.copy_(original_data[i]);
        }
    }

    // Return perturbed_grads; the division by relative_eps happens in Hv_func
    return perturbed_grads;
}