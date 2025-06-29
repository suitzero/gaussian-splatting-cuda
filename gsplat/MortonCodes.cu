#include "MortonCodes.h"
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include "Common.h" // For CHECK_CUDA, CHECK_CONTIGUOUS, etc.

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void compute_morton_codes_kernel(
    const uint32_t N,                     // Number of Gaussians
    const scalar_t* __restrict__ means3d, // [N, 3] Gaussian means
    const vec3 world_min,                 // Minimum corner of the bounding box
    const vec3 world_max,                 // Maximum corner of the bounding box
    uint32_t* __restrict__ morton_codes  // [N] Output Morton codes
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    scalar_t x = means3d[idx * 3 + 0];
    scalar_t y = means3d[idx * 3 + 1];
    scalar_t z = means3d[idx * 3 + 2];

    // Normalize coordinates to [0, 1] within the bounding box
    scalar_t norm_x = (x - world_min.x) / (world_max.x - world_min.x);
    scalar_t norm_y = (y - world_min.y) / (world_max.y - world_min.y);
    scalar_t norm_z = (z - world_min.z) / (world_max.z - world_min.z);

    // Clamp and scale to [0, 1023] for 10-bit Morton codes
    // 1023 = 2^10 - 1
    uint32_t morton_x = static_cast<uint32_t>(clamp(norm_x * 1023.0f, 0.0f, 1023.0f));
    uint32_t morton_y = static_cast<uint32_t>(clamp(norm_y * 1023.0f, 0.0f, 1023.0f));
    uint32_t morton_z = static_cast<uint32_t>(clamp(norm_z * 1023.0f, 0.0f, 1023.0f));

    morton_codes[idx] = morton3D(morton_x, morton_y, morton_z);
}

void launch_compute_morton_codes_kernel(
    const at::Tensor means3d,        // [N, 3]
    const vec3 world_min,
    const vec3 world_max,
    at::Tensor morton_codes          // [N]
) {
    CHECK_INPUT(means3d);
    CHECK_INPUT(morton_codes);
    TORCH_CHECK(means3d.dim() == 2 && means3d.size(1) == 3, "means3d must be [N, 3]");
    TORCH_CHECK(morton_codes.dim() == 1 && morton_codes.size(0) == means3d.size(0), "morton_codes must be [N]");
    TORCH_CHECK(morton_codes.scalar_type() == at::kInt, "morton_codes must be of type int (uint32_t)");


    uint32_t N = means3d.size(0);
    if (N == 0) return;

    dim3 threads(256);
    dim3 grid((N + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(
        means3d.scalar_type(), "compute_morton_codes_kernel", ([&] {
            compute_morton_codes_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N,
                means3d.data_ptr<scalar_t>(),
                world_min,
                world_max,
                reinterpret_cast<uint32_t*>(morton_codes.data_ptr<int32_t>()) // PyTorch uses int32 for kInt
            );
        })
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace gsplat
