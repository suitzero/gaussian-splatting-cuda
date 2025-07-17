#pragma once

#include "core/istrategy.hpp"
#include "core/selective_adam.hpp"
#include "core/camera.hpp"
#include <memory>
#include <torch/torch.h>

class NewtonStrategy : public IStrategy {
public:
    NewtonStrategy() = delete;
    NewtonStrategy(SplatData&& splat_data);

    NewtonStrategy(const NewtonStrategy&) = delete;
    NewtonStrategy& operator=(const NewtonStrategy&) = delete;
    NewtonStrategy(NewtonStrategy&&) = default;
    NewtonStrategy& operator=(NewtonStrategy&&) = default;

    // IStrategy interface implementation
    void initialize(const gs::param::OptimizationParameters& optimParams) override;
    void post_backward(int iter, gs::RenderOutput& render_output) override;
    bool is_refining(int iter) const override;
    void step(int iter) override;
    SplatData& get_model() override { return _splat_data; }
    const SplatData& get_model() const override { return _splat_data; }

    torch::Tensor compute_loss(Camera* viewpoint_camera,const gs::RenderOutput& render_output,const torch::Tensor& gt_image,bool update_members = true);
    std::vector<torch::Tensor> compute_perturbed_grad(const std::vector<torch::Tensor>& p,float eps);
    std::vector<torch::Tensor> conjugate_gradient();

private:
    class ExponentialLR {
    public:
        ExponentialLR(torch::optim::Optimizer& optimizer, double gamma, int param_group_index = -1)
            : optimizer_(optimizer),
              gamma_(gamma),
              param_group_index_(param_group_index) {}

        void step();

    private:
        torch::optim::Optimizer& optimizer_;
        double gamma_;
        int param_group_index_;
    };
    // Helper functions
    torch::Tensor multinomial_sample(const torch::Tensor& weights, int n, bool replacement = true);
    int relocate_gs();
    int add_new_gs();
    void inject_noise(int iter);
    void update_optimizer_for_relocate(torch::optim::Optimizer* optimizer,
                                       const torch::Tensor& sampled_indices,
                                       const torch::Tensor& dead_indices,
                                       int param_position);


    Camera* _cam;
    // Member variables
    std::unique_ptr<torch::optim::Optimizer> _optimizer;
    std::unique_ptr<ExponentialLR> _scheduler;
    SplatData _splat_data;
    std::unique_ptr<const gs::param::OptimizationParameters> _params;

    // Member variables for storing results from loss_backward_and_hvp
    float _new_loss, _original_loss;
    torch::Tensor _current_loss;
    torch::Tensor _gt_image;
    std::vector<torch::Tensor> _current_grads;
    std::vector<torch::Tensor> _current_hvp_result;
    torch::autograd::variable_list _current_params_list;

    // MCMC specific parameters
    const float _noise_lr = 5e5;
    float _damping = 0.1;

    // State variables
    torch::Tensor _binoms;

    // SelectiveAdam support
    torch::Tensor _last_visibility_mask;
};
