#include "core/pico_strategy.hpp"
#include "Ops.h"
#include "core/debug_utils.hpp"
#include "core/parameters.hpp"
#include "core/rasterizer.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <exception>
#include <iostream>
#include <random>

PicoStrategy::PicoStrategy(SplatData&& splat_data)
    : _splat_data(std::move(splat_data)) {
}


void PicoStrategy::post_backward(int iter, gs::RenderOutput& render_output) {

    if (iter == 1000) {
        using torch::optim::AdamOptions;
        std::vector<torch::optim::OptimizerParamGroup> groups;

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
    }
}

void PicoStrategy::step(int iter) {
    if (iter < _params->iterations) {
	    _optimizer->step();
		_optimizer->zero_grad(true);
    }
}

void PicoStrategy::initialize(const gs::param::OptimizationParameters& optimParams) {

    _params = std::make_unique<const gs::param::OptimizationParameters>(optimParams);

    const auto dev = torch::kCUDA;
    _splat_data.means() = _splat_data.means().to(dev).set_requires_grad(true);
    _splat_data.scaling_raw() = _splat_data.scaling_raw().to(dev).set_requires_grad(true);
    _splat_data.rotation_raw() = _splat_data.rotation_raw().to(dev).set_requires_grad(true);
    _splat_data.opacity_raw() = _splat_data.opacity_raw().to(dev).set_requires_grad(true);
    _splat_data.sh0() = _splat_data.sh0().to(dev).set_requires_grad(true);
    _splat_data.shN() = _splat_data.shN().to(dev).set_requires_grad(true);

    // Initialize optimizer

	using torch::optim::AdamOptions;
	std::vector<torch::optim::OptimizerParamGroup> groups;

	// Calculate initial learning rate for position
    /* groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.means()},
														  std::make_unique<AdamOptions>(_params->means_lr * _splat_data.get_scene_scale())));*/
    /* groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.sh0()},
														  std::make_unique<AdamOptions>(_params->shs_lr)));
	groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.shN()},
														  std::make_unique<AdamOptions>(_params->shs_lr / 20.f)));*/
	groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.scaling_raw()},
														  std::make_unique<AdamOptions>(_params->scaling_lr)));
    groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.rotation_raw()},
														  std::make_unique<AdamOptions>(_params->rotation_lr)));
    /* groups.emplace_back(torch::optim::OptimizerParamGroup({_splat_data.opacity_raw()},
														  std::make_unique<AdamOptions>(_params->opacity_lr))); */

	for (auto& g : groups)
		static_cast<AdamOptions&>(g.options()).eps(1e-15);

	_optimizer = std::make_unique<torch::optim::Adam>(groups, AdamOptions(0.f).eps(1e-15));
}

bool PicoStrategy::is_refining(int iter) const {

    return false;
}
