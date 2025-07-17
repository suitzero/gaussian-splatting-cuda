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

// Removed update_optimizer_for_relocate as no Adam state

int NewtonStrategy::relocate_gs() {
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

    auto probs = opacities.index_select(0, alive_indices);
    auto sampled_idxs_local = multinomial_sample(probs, n_dead, true);
    auto sampled_idxs = alive_indices.index_select(0, sampled_idxs_local);

    auto sampled_opacities = opacities.index_select(0, sampled_idxs);
    auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

    auto ratios = torch::zeros({opacities.size(0)}, torch::kFloat32).to(torch::kCUDA);
    ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kFloat32));
    ratios = ratios.index_select(0, sampled_idxs) + 1;

    const int n_max = static_cast<int>(_binoms.size(0));
    ratios = torch::clamp(ratios, 1, n_max);
    ratios = ratios.to(torch::kInt32).contiguous();

    auto relocation_result = gsplat::relocation(
        sampled_opacities, sampled_scales, ratios, _binoms, n_max);

    auto new_opacities = std::get<0>(relocation_result);
    auto new_scales = std::get<1>(relocation_result);

    new_opacities = torch::clamp(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

    if (_splat_data.opacity_raw().dim() == 2) {
        _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                             torch::logit(new_opacities).unsqueeze(-1));
    } else {
        _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
    }
    _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

    _splat_data.means().index_put_({dead_indices}, _splat_data.means().index_select(0, sampled_idxs));
    _splat_data.sh0().index_put_({dead_indices}, _splat_data.sh0().index_select(0, sampled_idxs));
    _splat_data.shN().index_put_({dead_indices}, _splat_data.shN().index_select(0, sampled_idxs));
    _splat_data.scaling_raw().index_put_({dead_indices}, _splat_data.scaling_raw().index_select(0, sampled_idxs));
    _splat_data.rotation_raw().index_put_({dead_indices}, _splat_data.rotation_raw().index_select(0, sampled_idxs));
    _splat_data.opacity_raw().index_put_({dead_indices}, _splat_data.opacity_raw().index_select(0, sampled_idxs));

    // Removed optimizer state update

    return n_dead;
}

int NewtonStrategy::add_new_gs() {
    torch::NoGradGuard no_grad;

    const int current_n = _splat_data.size();
    const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
    const int n_new = std::max(0, n_target - current_n);

    if (n_new == 0)
        return 0;

    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto probs = opacities.flatten();
    auto sampled_idxs = multinomial_sample(probs, n_new, true);

    auto sampled_opacities = opacities.index_select(0, sampled_idxs);
    auto sampled_scales = _splat_data.get_scaling().index_select(0, sampled_idxs);

    auto ratios = torch::zeros({opacities.size(0)}, torch::kFloat32).to(torch::kCUDA);
    ratios.index_add_(0, sampled_idxs, torch::ones_like(sampled_idxs, torch::kFloat32));
    ratios = ratios.index_select(0, sampled_idxs) + 1;

    const int n_max = static_cast<int>(_binoms.size(0));
    ratios = torch::clamp(ratios, 1, n_max);
    ratios = ratios.to(torch::kInt32).contiguous();

    auto relocation_result = gsplat::relocation(
        sampled_opacities, sampled_scales, ratios, _binoms, n_max);

    auto new_opacities = std::get<0>(relocation_result);
    auto new_scales = std::get<1>(relocation_result);

    new_opacities = torch::clamp(new_opacities, _params->min_opacity, 1.0f - 1e-7f);

    if (_splat_data.opacity_raw().dim() == 2) {
        _splat_data.opacity_raw().index_put_({sampled_idxs, torch::indexing::Slice()},
                                             torch::logit(new_opacities).unsqueeze(-1));
    } else {
        _splat_data.opacity_raw().index_put_({sampled_idxs}, torch::logit(new_opacities));
    }
    _splat_data.scaling_raw().index_put_({sampled_idxs}, torch::log(new_scales));

    auto new_means = _splat_data.means().index_select(0, sampled_idxs);
    auto new_sh0 = _splat_data.sh0().index_select(0, sampled_idxs);
    auto new_shN = _splat_data.shN().index_select(0, sampled_idxs);
    auto new_scaling = _splat_data.scaling_raw().index_select(0, sampled_idxs);
    auto new_rotation = _splat_data.rotation_raw().index_select(0, sampled_idxs);
    auto new_opacity = _splat_data.opacity_raw().index_select(0, sampled_idxs);

    _splat_data.means() = torch::cat({_splat_data.means(), new_means}, 0).set_requires_grad(true);
    _splat_data.sh0() = torch::cat({_splat_data.sh0(), new_sh0}, 0).set_requires_grad(true);
    _splat_data.shN() = torch::cat({_splat_data.shN(), new_shN}, 0).set_requires_grad(true);
    _splat_data.scaling_raw() = torch::cat({_splat_data.scaling_raw(), new_scaling}, 0).set_requires_grad(true);
    _splat_data.rotation_raw() = torch::cat({_splat_data.rotation_raw(), new_rotation}, 0).set_requires_grad(true);
    _splat_data.opacity_raw() = torch::cat({_splat_data.opacity_raw(), new_opacity}, 0).set_requires_grad(true);

    // Removed optimizer state update

    return n_new;
}

void NewtonStrategy::inject_noise() {
    torch::NoGradGuard no_grad;

    auto opacities = _splat_data.get_opacity();
    if (opacities.dim() == 2 && opacities.size(1) == 1) {
        opacities = opacities.squeeze(-1);
    }

    auto scales = _splat_data.get_scaling();
    auto quats = _splat_data.get_rotation();

    auto covar_result = gsplat::quat_scale_to_covar_preci_fwd(
        quats,
        scales,
        true,  // compute_covar
        false, // compute_preci
        false  // triu
    );
    auto covars = std::get<0>(covar_result); // [N, 3, 3]

    const float k = 100.0f;
    const float x0 = 0.995f;
    auto op_sigmoid = 1.0f / (1.0f + torch::exp(-k * ((1.0f - opacities) - x0)));

    float current_lr = 1e-3f; // Fixed reasonable value for noise, adjust if needed

    auto noise = torch::randn_like(_splat_data.means()) * op_sigmoid.unsqueeze(-1) * current_lr * _noise_lr;

    noise = torch::bmm(covars, noise.unsqueeze(-1)).squeeze(-1);

    _splat_data.means().add_(noise);
}

void NewtonStrategy::post_backward(int iter, gs::RenderOutput& render_output) {
    torch::NoGradGuard no_grad;
    if (iter % _params->sh_degree_interval == 0) {
        _splat_data.increment_sh_degree();
    }

    bool densified = false;
    if (is_refining(iter)) {
        relocate_gs();
        if (add_new_gs() > 0) {
            densified = true;
        }
        c10::cuda::CUDACachingAllocator::emptyCache();
    }

    inject_noise();

    // Refresh _current_params_list after potential densification
    _current_params_list = {
        _splat_data.means(),
        _splat_data.rotation_raw(),
        _splat_data.scaling_raw(),
        _splat_data.opacity_raw(),
        _splat_data.shN(),
        _splat_data.sh0()};

    // If densified, recompute gradients for the new state
    if (densified) {
        // Clear old grads
        for (auto& param : _current_params_list) {
            if (param.grad().defined()) {
                param.grad().zero_();
            }
        }
        // Re-render and recompute loss/grads
        auto background_ = torch::tensor({0.f, 0.f, 0.f}, torch::TensorOptions().dtype(torch::kFloat32)).to(torch::kCUDA);
        const gs::RenderMode render_mode = gs::stringToRenderMode(_params->render_mode);
        auto render = gs::rasterize(*_cam, _splat_data, background_, 1.0f, false, false, render_mode);
        auto loss = compute_loss(_cam, render, _gt_image, false);
        loss.backward();
        _current_grads.clear();
        for (const auto& param : _current_params_list) {
            _current_grads.push_back(param.grad().clone());
        }
    }
}

void NewtonStrategy::step(int iter) {
    if (iter >= _params->iterations)
        return;

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
    std::cout << "Iter " << iter << ": Grad norm = " << grad_norm << ", Update norm = " << update_norm << std::endl;

    {
        torch::NoGradGuard no_grad;
        if (_current_params_list.size() == delta_p.size()) {
            for (size_t i = 0; i < _current_params_list.size(); ++i) {
                if (_current_params_list[i].defined() && delta_p[i].defined()) {
                    _current_params_list[i].add_(delta_p[i], -1.0); // Optional: multiply by small LR like 0.1 if updates too large
                }
            }
        } else {
            std::cerr << "Warning: Mismatch between param list size and delta_p size. Skipping update." << std::endl;
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
        float eps = 1e-5f;
        auto grad_p1 = compute_perturbed_grad(v, eps);
        auto grad_m1 = compute_perturbed_grad(v, -eps);
        auto grad_p2 = compute_perturbed_grad(v, 2 * eps);
        auto grad_m2 = compute_perturbed_grad(v, -2 * eps);

        std::vector<torch::Tensor> hvp;
        for (int i = 0; i < v.size(); ++i) {
            auto diff = (-grad_p2[i] + 8 * grad_p1[i] - 8 * grad_m1[i] + grad_m2[i]) / (12 * eps);
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

    for (int i = 0; i < max_iter; ++i) {
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
    auto l1_loss = torch::l1_loss(rendered, gt);
    torch::Tensor loss = (1.f - _params->lambda_dssim) * l1_loss + _params->lambda_dssim * ssim_loss;

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
        _current_params_list = {
            _splat_data.means(),
            _splat_data.rotation_raw(),
            _splat_data.scaling_raw(),
            _splat_data.opacity_raw(),
            _splat_data.shN(),
            _splat_data.sh0()};
        _current_grads.clear();
        for (const auto& param : _current_params_list) {
            _current_grads.push_back(param.grad());
        }
    }

    return loss;
}

std::vector<torch::Tensor> NewtonStrategy::compute_perturbed_grad(const std::vector<torch::Tensor>& perturb, float eps) {
    std::vector<torch::Tensor> original_data;
    for (const auto& param : _current_params_list) {
        original_data.push_back(param.clone());
    }
    {
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < perturb.size(); ++i) {
            // Use relative eps per param for better numerical stability
            float param_norm = _current_params_list[i].norm().item<float>() + 1e-6f; // Avoid zero
            float relative_eps = eps * param_norm;
            auto& param = _current_params_list[i];
            param.add_(perturb[i] * relative_eps);
        }
    }

    // Debug print: compare perturbed to original for each param type
    /*
    std::vector<std::string> param_names = {"means [N,3]", "rotation_raw [N,4]", "scaling_raw [N,3]", "opacity_raw [N,1]", "shN [N,K-1,3]", "sh0 [N,1,3]"};
    for (size_t i = 0; i < _current_params_list.size(); ++i) {
        auto& param = _current_params_list[i];
        auto& orig = original_data[i];
        std::cout << "Param " << param_names[i] << " change:" << std::endl;

        int num_print = std::min(5LL, param.numel());
        auto param_slice = param.slice(0, 0, num_print).cpu();
        auto orig_slice = orig.slice(0, 0, num_print).cpu();

        for (int j = 0; j < num_print; ++j) {
            float p_val = param_slice.data_ptr<float>()[j];
            float o_val = orig_slice.data_ptr<float>()[j];
            std::cout << "Elem " << j << ": orig=" << o_val << ", perturbed=" << p_val << ", diff=" << (p_val - o_val) << std::endl;
        }

        auto diff = torch::abs(param - orig).mean().item<float>();
        std::cout << "Mean abs diff: " << diff << std::endl;
    }*/

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
            auto& param = _current_params_list[i];
            perturbed_grads[i] = param.grad().clone();
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

    return perturbed_grads;
}