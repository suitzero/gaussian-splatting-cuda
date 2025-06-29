#pragma once

#include <cstdint> // For uint32_t
#include "Common.h" // For vec3, includes ATen/core/Tensor.h if this is how at::Tensor is made available generally
                   // Or, we need to include ATen headers directly for at::Tensor usage.

// Forward declaration for at::Tensor if its full definition isn't needed by this header alone for all users.
// However, function signatures returning or taking at::Tensor by value/reference need the full definition.
// Common.h or Ops.h should provide ATen/core/Tensor.h ideally.
// If not, uncomment below and include minimally.
// #include <ATen/core/Tensor.h>

// Ensure PyTorch types are available for function signatures
#ifndef AT_TENSOR_H
#include <ATen/core/Tensor.h>
#endif


namespace gsplat {

// For CUDA specific keywords
#ifdef __NVCC__
    #define GSPLAT_DEVICE __device__
    #define GSPLAT_GLOBAL __global__
#else
    #define GSPLAT_DEVICE
    #define GSPLAT_GLOBAL
#endif

// Expands a 10-bit integer into 30 bits by inserting two zeros after each bit.
inline GSPLAT_DEVICE uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates 3D Morton code for a point (x, y, z).
// Input coordinates are expected to be normalized to [0, 1023].
inline GSPLAT_DEVICE uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t xx = expandBits(x);
    uint32_t yy = expandBits(y);
    uint32_t zz = expandBits(z);
    return xx | (yy << 1) | (zz << 2);
}

#ifdef __NVCC__
// Kernel to compute Morton codes for a set of 3D points.
// Declaration only needed for NVCC compilation path if called from other .cu files,
// or if we want strict separation (implementation in .cu).
// The template definition is in MortonCodes.cu
template <typename scalar_t>
GSPLAT_GLOBAL void compute_morton_codes_kernel(
    const uint32_t N,                     // Number of Gaussians
    const scalar_t* __restrict__ means3d, // [N, 3] Gaussian means
    const vec3 world_min,                 // Minimum corner of the bounding box
    const vec3 world_max,                 // Maximum corner of the bounding box
    uint32_t* __restrict__ morton_codes  // [N] Output Morton codes
);
#endif

// Launcher function for the Morton coding kernel.
// This is callable from C++ code (that links with the CUDA object).
void launch_compute_morton_codes_kernel(
    const at::Tensor& means3d,        // [N, 3]
    const vec3& world_min,
    const vec3& world_max,
    at::Tensor& morton_codes          // [N]
);

// C++ wrapper function callable from other C++ code (e.g., tests or pipeline)
// This is implemented in MortonCodes.cpp
at::Tensor compute_morton_codes_tensor(
    const at::Tensor& means3d,        // [N, 3]
    const at::Tensor& world_min_tensor, // [3]
    const at::Tensor& world_max_tensor  // [3]
);

} // namespace gsplat

// Undefine macros if they are only for this header
#undef GSPLAT_DEVICE
#undef GSPLAT_GLOBAL
