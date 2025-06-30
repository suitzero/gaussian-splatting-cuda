#pragma once

#include "core/parameters.hpp" // For TrainingParameters
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <ATen/ATen.h> // For at::Tensor and at::optional<at::Tensor>

// Helper to check CUDA calls (can be moved to a common utils header if not already present)
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error in %s at line %d: %s (%d)\\n", __FILE__, \
                __LINE__, cudaGetErrorString(err), err);                     \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while (0)
#endif

// Forward declare CameraModelType and other enums if they are in gsplat namespace
// and used by kernel launchers (they are, from Rasterization.h)
namespace gsplat {
    enum CameraModelType;
    struct UnscentedTransformParameters;
    enum ShutterType;
}


namespace gs {
namespace benchmark {

// Structure to hold at::Tensor objects for the benchmark
struct BenchmarkData {
    // Common parameters that define tensor sizes (can also be stored/loaded from snapshot metadata)
    uint32_t N_gaussians = 0;
    uint32_t C_cameras = 1; // Number of cameras
    uint32_t CDIM_colors = 3;
    uint32_t image_width = 0;
    uint32_t image_height = 0;
    uint32_t tile_size = 16;
    uint32_t n_isects = 0;
    bool packed_format = false;

    // --- Inputs for rasterize_to_pixels_3dgs_bwd_kernel ---
    at::Tensor means2d_tensor;
    at::Tensor conics_tensor;
    at::Tensor colors_in_tensor;
    at::Tensor opacities_in_tensor;
    at::optional<at::Tensor> backgrounds_tensor;
    at::optional<at::Tensor> masks_tensor;

    at::Tensor tile_offsets_tensor;
    at::Tensor flatten_ids_tensor;

    at::Tensor render_alphas_tensor; // Forward output
    at::Tensor last_ids_tensor;      // Forward output

    at::Tensor v_render_colors_tensor; // Incoming gradient
    at::Tensor v_render_alphas_tensor; // Incoming gradient

    // Output gradient tensors (to be computed by the kernel)
    at::optional<at::Tensor> v_means2d_abs_tensor;
    at::Tensor v_means2d_tensor;
    at::Tensor v_conics_tensor;
    at::Tensor v_colors_tensor;
    at::Tensor v_opacities_tensor;

    // --- Inputs for rasterize_to_pixels_from_world_3dgs_bwd_kernel ---
    at::Tensor means3d_tensor;
    at::Tensor quats_tensor;
    at::Tensor scales_tensor;

    at::Tensor viewmats0_tensor;
    at::optional<at::Tensor> viewmats1_tensor;
    at::Tensor Ks_tensor;

    // Enum types for world kernel (need to be set based on snapshot/params)
    gsplat::CameraModelType camera_model_type;
    gsplat::UnscentedTransformParameters ut_params; // TODO: Populate this properly
    gsplat::ShutterType rs_type;

    at::optional<at::Tensor> radial_coeffs_tensor;
    at::optional<at::Tensor> tangential_coeffs_tensor;
    at::optional<at::Tensor> thin_prism_coeffs_tensor;

    // Output gradient tensors for world kernel
    at::Tensor v_means3d_tensor;
    at::Tensor v_quats_tensor;
    at::Tensor v_scales_tensor;
    // v_colors_tensor, v_opacities_tensor can be reused if CDIM is the same

    // No explicit destructor needed for at::Tensor members, they manage their own memory.
};

// Function to "load" data. Initially populates with mock at::Tensor objects.
/*
 * User Responsibilities for `load_benchmark_snapshot`:
 * 1. Saving Tensors:
 *    - During a normal training run, at the point *just before* the backward rasterization
 *      kernel (`launch_rasterize_to_pixels_3dgs_bwd_kernel` or `..._from_world_...`) is called,
 *      all its `at::Tensor` inputs MUST be saved to disk.
 *    - It's recommended to save each tensor as a separate `.pt` file (using `torch::save(tensor, "path/to/tensor_name.pt");`)
 *      within a dedicated directory. This directory path will be passed via `--benchmark_input_snapshot`.
 *    - Expected tensor names are used in `src/benchmark_utils.cpp` (e.g., "means2d.pt", "conics.pt", etc.).
 *      Match these names when saving.
 *    - Also save any necessary metadata (like N_gaussians, image_width, image_height, n_isects, packed_format,
 *      camera_model_type, rs_type for the world kernel, etc.) if they cannot be reliably
 *      derived from `TrainingParameters` or tensor shapes alone for the benchmark. A simple JSON or text file
 *      for metadata in the snapshot directory is a good approach.
 *
 * 2. Modifying `create_mock_gpu_at_tensor` (or `load_benchmark_snapshot` directly):
 *    - The current implementation of `create_mock_gpu_at_tensor` in `src/benchmark_utils.cpp`
 *      INITIALIZES TENSORS WITH MOCK DATA.
 *    - You MUST replace the mock data initialization with actual loading logic using `torch::load(file_path.string())`
 *      for each tensor.
 *    - Ensure that the loaded tensors are correctly moved to CUDA device, have the correct dtype,
 *      and their `requires_grad` status is set appropriately for the benchmark.
 *    - Handle potential errors during file loading (e.g., file not found, incorrect format).
 *
 * The quality and realism of the benchmark heavily depend on using representative data
 * captured from an actual training scenario.
 */
bool load_benchmark_snapshot(
    const std::string& snapshot_path,
    const gs::param::TrainingParameters& params,
    BenchmarkData& benchmark_data // Output parameter
);

} // namespace benchmark
} // namespace gs
