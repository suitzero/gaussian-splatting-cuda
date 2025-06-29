#include "MortonCodes.h"
#include "Common.h" // For CHECK_CUDA, CHECK_CONTIGUOUS, etc.
#include <ATen/core/Tensor.h>
#include <torch/extension.h> // For pybind11

namespace gsplat {

// C++ wrapper to be exposed to Python
at::Tensor compute_morton_codes_tensor(
    const at::Tensor means3d,        // [N, 3]
    const at::Tensor world_min_tensor, // [3]
    const at::Tensor world_max_tensor  // [3]
) {
    CHECK_INPUT(means3d);
    CHECK_INPUT(world_min_tensor);
    CHECK_INPUT(world_max_tensor);

    TORCH_CHECK(means3d.dim() == 2 && means3d.size(1) == 3, "means3d must be [N, 3]");
    TORCH_CHECK(world_min_tensor.dim() == 1 && world_min_tensor.size(0) == 3, "world_min_tensor must be [3]");
    TORCH_CHECK(world_max_tensor.dim() == 1 && world_max_tensor.size(0) == 3, "world_max_tensor must be [3]");

    uint32_t N = means3d.size(0);
    auto options = at::TensorOptions().device(means3d.device()).dtype(at::kInt);
    at::Tensor morton_codes = at::empty({N}, options);

    if (N == 0) {
        return morton_codes;
    }

    // Convert world_min and world_max tensors to vec3
    // Ensure they are on CPU to access data easily, or access data_ptr directly if already float
    at::Tensor min_cpu = world_min_tensor.to(at::kCPU).contiguous();
    at::Tensor max_cpu = world_max_tensor.to(at::kCPU).contiguous();

    vec3 world_min_val(min_cpu.data_ptr<float>()[0], min_cpu.data_ptr<float>()[1], min_cpu.data_ptr<float>()[2]);
    vec3 world_max_val(max_cpu.data_ptr<float>()[0], max_cpu.data_ptr<float>()[1], max_cpu.data_ptr<float>()[2]);

    launch_compute_morton_codes_kernel(
        means3d,
        world_min_val,
        world_max_val,
        morton_codes
    );

    return morton_codes;
}

} // namespace gsplat
