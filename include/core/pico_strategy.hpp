#pragma once

#include "core/istrategy.hpp"
#include "core/selective_adam.hpp"
#include <memory>
#include <torch/torch.h>

class PicoStrategy : public IStrategy {
public:
    PicoStrategy() = delete;
    PicoStrategy(SplatData&& splat_data);

    PicoStrategy(const PicoStrategy&) = delete;
    PicoStrategy& operator=(const PicoStrategy&) = delete;
    PicoStrategy(PicoStrategy&&) = default;
    PicoStrategy& operator=(PicoStrategy&&) = default;

    // IStrategy interface implementation
    void initialize(const gs::param::OptimizationParameters& optimParams) override;
    void post_backward(int iter, gs::RenderOutput& render_output) override;
    bool is_refining(int iter) const override;
    void step(int iter) override;
    SplatData& get_model() override { return _splat_data; }
    const SplatData& get_model() const override { return _splat_data; }


private:
    // Member variables
    std::unique_ptr<torch::optim::Optimizer> _optimizer;
    SplatData _splat_data;
    std::unique_ptr<const gs::param::OptimizationParameters> _params;
};

