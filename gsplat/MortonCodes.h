#pragma once

#include <cstdint>
#include "Common.h" // For vec3

namespace gsplat {

// Expands a 10-bit integer into 30 bits by inserting two zeros after each bit.
inline __device__ uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates 3D Morton code for a point (x, y, z).
// Input coordinates are expected to be normalized to [0, 1023].
inline __device__ uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t xx = expandBits(x);
    uint32_t yy = expandBits(y);
    uint32_t zz = expandBits(z);
    return xx | (yy << 1) | (zz << 2);
}

// Kernel to compute Morton codes for a set of 3D points.
template <typename scalar_t>
__global__ void compute_morton_codes_kernel(
    const uint32_t N,                     // Number of Gaussians
    const scalar_t* __restrict__ means3d, // [N, 3] Gaussian means
    const vec3 world_min,                 // Minimum corner of the bounding box
    const vec3 world_max,                 // Maximum corner of the bounding box
    uint32_t* __restrict__ morton_codes  // [N] Output Morton codes
);

// Launcher function for the Morton coding kernel.
void launch_compute_morton_codes_kernel(
    const at::Tensor means3d,        // [N, 3]
    const vec3 world_min,
    const vec3 world_max,
    at::Tensor morton_codes          // [N]
);

// C++ wrapper function callable from other C++ code (e.g., tests or pipeline)
// This is implemented in MortonCodes.cpp
at::Tensor compute_morton_codes_tensor(
    const at::Tensor means3d,        // [N, 3]
    const at::Tensor world_min_tensor, // [3]
    const at::Tensor world_max_tensor  // [3]
);

} // namespace gsplat
