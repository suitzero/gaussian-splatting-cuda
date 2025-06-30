/*
 * =====================================================================================
 * USER ACTION REQUIRED FOR ACCURATE BENCHMARKING:
 * =====================================================================================
 * This file (`benchmark_utils.cpp`) and its header (`benchmark_utils.hpp`) provide
 * a framework for benchmarking specific CUDA kernels, particularly the backward
 * rasterization pass.
 *
 * CURRENT STATE:
 * - The `create_mock_gpu_at_tensor` function (called by `load_benchmark_snapshot`)
 *   populates tensors with MOCK (dummy) data.
 * - The `load_benchmark_snapshot` function uses this mock data generation.
 *
 * FOR MEANINGFUL PERFORMANCE ANALYSIS, YOU MUST:
 *
 * 1. CAPTURE REAL TENSOR DATA:
 *    - Identify a representative point in your normal training pipeline, just before
 *      the backward rasterization kernel (`launch_rasterize_to_pixels_3dgs_bwd_kernel`
 *      or `launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel`) is invoked.
 *    - Save ALL `at::Tensor` inputs to these kernels to disk.
 *    - Recommended method: Use `torch::save(tensor, "/path/to/snapshot_dir/tensor_name.pt");`
 *      for each tensor. The `tensor_name.pt` should match the names used as the
 *      `tensor_name` argument in calls to `create_mock_gpu_at_tensor` within
 *      `load_benchmark_snapshot` (e.g., "means2d.pt", "conics.pt", etc.).
 *    - Create a dedicated directory for these snapshot files. This directory's path
 *      will be passed to the executable via the `--benchmark_input_snapshot` argument.
 *    - CRITICAL: Also save essential metadata alongside your tensors. This includes values like:
 *        - `N_gaussians` (true number of Gaussians in the snapshot)
 *        - `image_width`, `image_height` (of the rendered image)
 *        - `n_isects` (true number of intersections for `flatten_ids`)
 *        - `packed_format` (boolean, true if using packed representation in the snapshot)
 *        - `CDIM_colors` (actual number of channels for the *rendered output*, typically 3 for RGB)
 *        - `actual_color_input_channels` (number of channels for the `colors_in_tensor` fed to the kernel,
 *          e.g., 3 for RGB, or `(SH_DEGREE+1)^2` for SH coefficients if `sh_degree > 0`).
 *        - `C_cameras` (number of cameras/views in the snapshot)
 *        - `tile_size` (used for the snapshot)
 *        - For "world" kernel: `CameraModelType`, `ShutterType`, and values for `UnscentedTransformParameters`.
 *      A simple JSON or text file (e.g., `snapshot_metadata.json`) in the snapshot
 *      directory is a good way to store this metadata.
 *
 * 2. MODIFY `load_benchmark_snapshot` (and `create_mock_gpu_at_tensor`):
 *    - In `load_benchmark_snapshot`:
 *        - Load the metadata from your saved metadata file first.
 *        - Use this metadata to determine the correct shapes, dtypes, and parameters
 *          (N_gaussians, image dimensions, n_isects, C_cameras, etc.) for creating tensors.
 *          Do NOT rely on `params.optimization.max_cap` or other training parameters
 *          for these values in benchmark mode; use the captured snapshot's metadata.
 *    - In `create_mock_gpu_at_tensor` (or directly in `load_benchmark_snapshot` if preferred):
 *        - Replace the mock data generation with actual tensor loading logic.
 *        - Use `at::Tensor loaded_tensor = torch::load(file_path.string());` (You'll need to ensure
 *          your project links correctly against libtorch for `torch::load` to be available).
 *        - Ensure the loaded tensor is moved to the CUDA device (`.to(torch::kCUDA)`),
 *          cast to the correct `dtype` if necessary, and `requires_grad` is set appropriately
 *          (usually `false` for inputs, and also `false` for output gradient buffers in this C++
 *          benchmark context as we are not backpropagating further from them here).
 *        - Perform error checking (file existence, shape/dtype consistency with metadata).
 *
 * 3. VERIFY KERNEL LAUNCH PARAMETERS IN `src/main.cpp`:
 *    - The `run_benchmark` function in `src/main.cpp` currently hardcodes `<3>` for
 *      the `CDIM` template parameter of the kernel launchers. This assumes the *rendered output*
 *      (and thus `v_render_colors_tensor` and `v_colors_tensor`) has 3 channels.
 *    - If your actual `colors_in_tensor` (input to the kernel, representing SH coefficients if sh_degree > 0)
 *      or other tensors imply a different `CDIM` for the kernel's internal processing or specific
 *      templated versions, ensure the correct kernel instantiation is called.
 *      The `bm_data.CDIM_colors` is intended for rendered output channels.
 *      The `actual_color_input_channels` in `load_benchmark_snapshot` calculates the
 *      channels for the input `colors_in_tensor`.
 *
 * By following these steps, the benchmark mode will use realistic data, providing
 * accurate performance insights for your specific use case.
 * =====================================================================================
 */
#include "core/benchmark_utils.hpp"
#include "gsplat/Rasterization.h" // For CameraModelType etc.
#include "gsplat/Common.h"        // For gsplat::CameraModelType, ShutterType etc. if not fully in Rasterization.h
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <ATen/ATen.h> // For at::Tensor, torch::from_blob, etc.


// Note: gsplat::CameraModelType, gsplat::ShutterType, and gsplat::UnscentedTransformParameters
// are expected to be defined from gsplat/Common.h (included via gsplat/Rasterization.h or directly).

namespace gs {
namespace benchmark {

// Helper template to allocate and fill GPU at::Tensor with mock data
template<typename T_scalar, typename T_idx = int64_t>
at::Tensor create_mock_gpu_at_tensor(
    const std::vector<T_idx>& shape,
    T_scalar default_value,
    const std::string& tensor_name,
    const std::string& snapshot_path,
    at::ScalarType dtype,
    bool requires_grad = false) {

    std::filesystem::path file_path = std::filesystem::path(snapshot_path) / (tensor_name + ".pt"); // Assuming .pt for torch tensors

    // USER TODO: Implement actual file reading here for torch::Tensor.
    // Example using torch::load (requires libtorch to be linked correctly for I/O ops):
    /*
    if (!snapshot_path.empty() && std::filesystem::exists(file_path)) {
        try {
            at::Tensor loaded_tensor = torch::load(file_path.string());
            // Optional: Check shape and dtype against expected if metadata is available
            // For example: if (loaded_tensor.sizes() == shape && loaded_tensor.scalar_type() == dtype)
            std::cout << "INFO: Successfully loaded " << tensor_name << " from " << file_path << std::endl;
            return loaded_tensor.to(torch::kCUDA).to(dtype).set_requires_grad(requires_grad);
        } catch (const c10::Error& e) {
            std::cerr << "WARNING: PyTorch error loading tensor " << tensor_name << " from " << file_path << ": " << e.what() << ". Using mock data." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "WARNING: Generic error loading tensor " << tensor_name << " from " << file_path << ": " << e.what() << ". Using mock data." << std::endl;
        }
    } else {
        if (!snapshot_path.empty()) { // Only warn if a path was given but file not found
             std::cerr << "WARNING: Snapshot file " << file_path << " for " << tensor_name << " not found. Using mock data." << std::endl;
        }
    }
    */

    std::cout << "INFO: Using MOCK data for " << tensor_name
              << (snapshot_path.empty() ? "." : ". Expected at: " + file_path.string()) << std::endl;

    int64_t num_elements = 1;
    for (T_idx s : shape) {
        num_elements *= s;
    }
    if (num_elements == 0) { // Handle case for empty tensors if shape has a 0
      return torch::empty(shape, torch::TensorOptions().dtype(dtype).device(torch::kCUDA).requires_grad(requires_grad));
    }

    std::vector<T_scalar> h_data(num_elements);
    for(int64_t i = 0; i < num_elements; ++i) {
        h_data[i] = static_cast<T_scalar>(((i % 100) + 1) * 0.01f * default_value);
    }

    at::Tensor tensor = torch::from_blob(h_data.data(), shape, torch::TensorOptions().dtype(dtype)).clone();
    return tensor.to(torch::kCUDA).set_requires_grad(requires_grad);
}


bool load_benchmark_snapshot(
    const std::string& snapshot_path,
    const gs::param::TrainingParameters& params,
    BenchmarkData& bm_data // Output parameter
) {
    std::cout << "Attempting to load benchmark snapshot from: " << snapshot_path << std::endl;
    if (snapshot_path.empty()) {
        std::cerr << "ERROR: Snapshot path is empty. Cannot load benchmark data." << std::endl;
        return false;
    }
    // Warning if path doesn't exist, but still proceed with mock data.
    // USER TODO: For production/real benchmarking, if snapshot_path is provided but invalid,
    // this should likely be a hard error unless a fallback to mock data is explicitly desired.
    if (!std::filesystem::exists(snapshot_path) || !std::filesystem::is_directory(snapshot_path)) {
        std::cerr << "WARNING: Snapshot path " << snapshot_path << " does not exist or is not a directory. Will proceed with MOCK data for all tensors." << std::endl;
    }

    // Populate dimensions from params or defaults
    // USER TODO: These should ideally be loaded from a metadata file within the snapshot_path
    // to ensure the benchmark runs with the exact parameters of the captured state.
    bm_data.N_gaussians = params.optimization.max_cap > 0 ? static_cast<uint32_t>(params.optimization.max_cap) : 100000;
    // CDIM_colors calculation:
    // If sh_degree is 0, colors are RGB (3 channels).
    // If sh_degree > 0, colors are SH coefficients. Number of SH coeffs is (sh_degree + 1)^2.
    // Each SH coefficient is typically a vec3 (for R,G,B), so total channels = (sh_degree + 1)^2 * 3.
    // However, the backward kernel template CDIM often refers to the output color channels (usually 3 for RGB after SH evaluation).
    // The `colors_in_tensor` for the backward pass would be the SH coefficients if sh_degree > 0.
    // For simplicity in mock data, we'll assume `bm_data.CDIM_colors` refers to the final rendered color channels (typically 3).
    // The actual `colors_in_tensor` for the kernel will have (sh_degree+1)^2 channels if sh_degree > 0, or 3 if sh_degree == 0.
    // This distinction needs to be handled carefully if not using mock data.
    // For the benchmark kernel call, we are hardcoding CDIM=3, assuming the v_render_colors is 3-channel.
    bm_data.CDIM_colors = 3; // This is for the *rendered* color dimension, used for v_render_colors.

    // Actual input color tensor dimension calculation:
    int input_color_channels = 3; // Default for RGB
    if (params.optimization.sh_degree > 0) {
        input_color_channels = (params.optimization.sh_degree + 1) * (params.optimization.sh_degree + 1);
    }


    bm_data.image_width = params.dataset.resolution > 0 ? static_cast<uint32_t>(params.dataset.resolution) : 800;
    // Basic aspect ratio handling for mock data setup. Load from metadata for real scenarios.
    float aspect_ratio = (bm_data.image_width > 0 && params.dataset.images.size() > 0 && !params.dataset.images.empty())
                       ? static_cast<float>(bm_data.image_width) / (bm_data.image_width * 0.75f) /* TODO: Fix this, needs actual image height from dataset if available */
                       : (800.0f/600.0f);
    // If params.dataset.resolution only sets width, try to infer height, or default.
    // This is very approximate; real metadata is better.
    uint32_t inferred_height = 600;
    if(params.dataset.resolution > 0 && params.dataset.images.empty()){ // if resolution is set but no images to get aspect from
         // Try to find if a common aspect ratio like 4:3 or 16:9 was intended for height.
         // This is difficult without more info. For now, use a common default or calculate based on a typical aspect if width is known.
         if(bm_data.image_width == 800) inferred_height = 600;
         else if (bm_data.image_width == 1920) inferred_height = 1080;
         else inferred_height = bm_data.image_width * 3/4; // Default to 4:3
    } else if (!params.dataset.images.empty()){
        // A better way would be to load one image from params.dataset.images and get its dimensions if possible.
        // This is out of scope for mock data generation here.
    }
    bm_data.image_height = params.dataset.resolution > 0 ? inferred_height : 600; // Fallback if logic is complex
    if (bm_data.image_width == 0) bm_data.image_width = 800; // Ensure not zero
    if (bm_data.image_height == 0) bm_data.image_height = 600; // Ensure not zero


    bm_data.tile_size = 16;
    bm_data.packed_format = false; // USER TODO: Load from snapshot metadata.
    bm_data.C_cameras = 1; // USER TODO: Load from snapshot metadata. For now, assume 1 camera.

    // Estimate n_isects if not loaded. USER TODO: Load from snapshot metadata.
    bm_data.n_isects = bm_data.N_gaussians * 5;
    if (bm_data.n_isects == 0 && bm_data.N_gaussians > 0) bm_data.n_isects = bm_data.N_gaussians;

    at::TensorOptions float_opts = torch::TensorOptions().dtype(torch::kFloat32);
    at::TensorOptions int32_opts = torch::TensorOptions().dtype(torch::kInt32);
    // at::TensorOptions bool_opts = torch::TensorOptions().dtype(torch::kBool); // For masks if used

    // --- Common Data ---
    int64_t C = bm_data.C_cameras;
    int64_t N = bm_data.N_gaussians;
    int64_t n_isects_long = bm_data.n_isects;

    // Number of channels for the input 'colors' tensor to the kernel might be different from rendered CDIM_colors if using SH
    int64_t actual_color_input_channels = (params.optimization.sh_degree == 0) ? 3 : ((params.optimization.sh_degree + 1) * (params.optimization.sh_degree + 1));

    std::vector<int64_t> colors_in_tensor_shape = bm_data.packed_format ? std::vector<int64_t>{n_isects_long, actual_color_input_channels} : std::vector<int64_t>{C, N, actual_color_input_channels};
    bm_data.colors_in_tensor = create_mock_gpu_at_tensor<float>(colors_in_tensor_shape, 0.5f, "colors_in", snapshot_path, torch::kFloat32);

    std::vector<int64_t> opacities_shape = bm_data.packed_format ? std::vector<int64_t>{n_isects_long} : std::vector<int64_t>{C, N};
    bm_data.opacities_in_tensor = create_mock_gpu_at_tensor<float>(opacities_shape, 0.7f, "opacities_in", snapshot_path, torch::kFloat32);

    // Optional: backgrounds, masks
    // bm_data.backgrounds_tensor = create_mock_gpu_at_tensor<float>({C, bm_data.CDIM_colors}, 0.1f, "backgrounds", snapshot_path, torch::kFloat32);
    // uint32_t tile_w_for_mask = (bm_data.image_width + bm_data.tile_size - 1) / bm_data.tile_size;
    // uint32_t tile_h_for_mask = (bm_data.image_height + bm_data.tile_size - 1) / bm_data.tile_size;
    // bm_data.masks_tensor = create_mock_gpu_at_tensor<bool>({C, tile_h_for_mask, tile_w_for_mask}, true, "masks", snapshot_path, torch::kBool);

    uint32_t tile_w_dim = (bm_data.image_width + bm_data.tile_size - 1) / bm_data.tile_size;
    uint32_t tile_h_dim = (bm_data.image_height + bm_data.tile_size - 1) / bm_data.tile_size;
    if (tile_w_dim == 0) tile_w_dim = 1; // Avoid zero dimensions
    if (tile_h_dim == 0) tile_h_dim = 1;

    bm_data.tile_offsets_tensor = create_mock_gpu_at_tensor<int32_t>({C, tile_h_dim, tile_w_dim}, 0, "tile_offsets", snapshot_path, torch::kInt32);
    // Mock tile_offsets realistically - this is complex and snapshot data is much preferred.
    if (C > 0 && tile_h_dim > 0 && tile_w_dim > 0 && bm_data.tile_offsets_tensor.numel() > 0) {
        at::Tensor h_tile_offsets_cpu = bm_data.tile_offsets_tensor.cpu(); // Work on CPU copy
        auto accessor = h_tile_offsets_cpu.accessor<int32_t, 3>();
        int32_t current_offset_val = 0;
        size_t total_tiles = C * tile_h_dim * tile_w_dim;
        size_t avg_isects_per_tile = (total_tiles > 0 && bm_data.n_isects > 0) ? (bm_data.n_isects / total_tiles) : 1;
        if (avg_isects_per_tile == 0) avg_isects_per_tile = 1;

        for(int64_t cam_idx = 0; cam_idx < C; ++cam_idx) {
            for(int64_t th = 0; th < tile_h_dim; ++th) {
                for(int64_t tw = 0; tw < tile_w_dim; ++tw) {
                    accessor[cam_idx][th][tw] = current_offset_val;
                    if ( (cam_idx * tile_h_dim * tile_w_dim + th * tile_w_dim + tw) < total_tiles -1 ) { // Not the very last tile overall
                         current_offset_val += avg_isects_per_tile + ((th * tile_w_dim + tw) % 3); // Add some variation
                         if (current_offset_val > (int32_t)bm_data.n_isects) current_offset_val = bm_data.n_isects;
                    }
                }
            }
        }
        // The last element of tile_offsets (conceptually, if flattened) should point to n_isects
        // or be n_isects itself if it's the start of a non-existent next tile range.
        // This mock logic is imperfect. For correct ranges, the last tile_offset should allow reading up to n_isects.
        // A common way is to have an extra element in tile_offsets[max_tile_id+1] = n_isects.
        // For now, this simplified mock should be sufficient for basic structure.
        bm_data.tile_offsets_tensor.copy_(h_tile_offsets_cpu);
    }


    bm_data.flatten_ids_tensor = create_mock_gpu_at_tensor<int32_t>({n_isects_long}, 0, "flatten_ids", snapshot_path, torch::kInt32);
    if (n_isects_long > 0 && N > 0) { // Ensure N is positive before modulo
        at::Tensor h_flatten_ids_cpu = bm_data.flatten_ids_tensor.cpu();
        auto accessor = h_flatten_ids_cpu.accessor<int32_t, 1>();
        for(int64_t i = 0; i < n_isects_long; ++i) {
            accessor[i] = i % (bm_data.packed_format ? n_isects_long : N);
        }
        bm_data.flatten_ids_tensor.copy_(h_flatten_ids_cpu);
    }

    std::vector<int64_t> image_shape_suffix = {(int64_t)bm_data.image_height, (int64_t)bm_data.image_width};
    std::vector<int64_t> render_alpha_shape = {C}; std::copy(image_shape_suffix.begin(), image_shape_suffix.end(), std::back_inserter(render_alpha_shape)); render_alpha_shape.push_back(1);
    std::vector<int64_t> last_ids_shape = {C}; std::copy(image_shape_suffix.begin(), image_shape_suffix.end(), std::back_inserter(last_ids_shape));
    std::vector<int64_t> v_render_colors_shape = {C}; std::copy(image_shape_suffix.begin(), image_shape_suffix.end(), std::back_inserter(v_render_colors_shape)); v_render_colors_shape.push_back(bm_data.CDIM_colors);

    bm_data.render_alphas_tensor = create_mock_gpu_at_tensor<float>(render_alpha_shape, 0.5f, "render_alphas", snapshot_path, torch::kFloat32);
    bm_data.last_ids_tensor = create_mock_gpu_at_tensor<int32_t>(last_ids_shape, N > 0 ? N -1 : 0, "last_ids", snapshot_path, torch::kInt32);
    bm_data.v_render_colors_tensor = create_mock_gpu_at_tensor<float>(v_render_colors_shape, 0.01f, "v_render_colors", snapshot_path, torch::kFloat32);
    bm_data.v_render_alphas_tensor = create_mock_gpu_at_tensor<float>(render_alpha_shape, 0.01f, "v_render_alphas", snapshot_path, torch::kFloat32);

    // Output Gradients (allocated zeroed, requires_grad=true for autograd in PyTorch, but for C++ benchmark not strictly needed unless testing further C++ autograd)
    // For the kernel itself, requires_grad on output buffers is irrelevant.
    bm_data.v_colors_tensor = create_mock_gpu_at_tensor<float>(colors_in_tensor_shape, 0.0f, "v_colors_output_buffer", snapshot_path, torch::kFloat32, false);
    bm_data.v_opacities_tensor = create_mock_gpu_at_tensor<float>(opacities_shape, 0.0f, "v_opacities_output_buffer", snapshot_path, torch::kFloat32, false);

    if (params.benchmark.kernel_type == "2d") {
        std::cout << "INFO: Loading mock data for '2d' kernel type." << std::endl;
        std::vector<int64_t> means2d_shape = bm_data.packed_format ? std::vector<int64_t>{n_isects_long, 2} : std::vector<int64_t>{C, N, 2};
        bm_data.means2d_tensor = create_mock_gpu_at_tensor<float>(means2d_shape, 0.5f, "means2d", snapshot_path, torch::kFloat32);

        std::vector<int64_t> conics_shape = bm_data.packed_format ? std::vector<int64_t>{n_isects_long, 3} : std::vector<int64_t>{C, N, 3};
        bm_data.conics_tensor = create_mock_gpu_at_tensor<float>(conics_shape, 0.1f, "conics", snapshot_path, torch::kFloat32);

        // bm_data.v_means2d_abs_tensor = create_mock_gpu_at_tensor<float>(means2d_shape, 0.0f, "v_means2d_abs_output_buffer", snapshot_path, torch::kFloat32, false); // Optional
        bm_data.v_means2d_tensor = create_mock_gpu_at_tensor<float>(means2d_shape, 0.0f, "v_means2d_output_buffer", snapshot_path, torch::kFloat32, false);
        bm_data.v_conics_tensor = create_mock_gpu_at_tensor<float>(conics_shape, 0.0f, "v_conics_output_buffer", snapshot_path, torch::kFloat32, false);

    } else if (params.benchmark.kernel_type == "world") {
        std::cout << "INFO: Loading mock data for 'world' kernel type." << std::endl;
        bm_data.means3d_tensor = create_mock_gpu_at_tensor<float>({N, 3}, 0.0f, "means3d", snapshot_path, torch::kFloat32);
        bm_data.quats_tensor = create_mock_gpu_at_tensor<float>({N, 4}, 0.0f, "quats", snapshot_path, torch::kFloat32);
        if (N > 0 && bm_data.quats_tensor.numel() > 0) {
            at::Tensor h_quats_cpu = bm_data.quats_tensor.cpu();
            auto h_quats_acc = h_quats_cpu.accessor<float,2>();
            for(int64_t i=0; i < N; ++i) h_quats_acc[i][3] = 1.0f; // w component for identity-like quaternion
            bm_data.quats_tensor.copy_(h_quats_cpu);
        }
        bm_data.scales_tensor = create_mock_gpu_at_tensor<float>({N, 3}, 0.1f, "scales", snapshot_path, torch::kFloat32);

        bm_data.viewmats0_tensor = create_mock_gpu_at_tensor<float>({C, 4, 4}, 0.0f, "viewmats0", snapshot_path, torch::kFloat32);
        if (C > 0 && bm_data.viewmats0_tensor.numel() > 0) {
            at::Tensor h_vm_cpu = bm_data.viewmats0_tensor.cpu();
            auto h_vm_acc = h_vm_cpu.accessor<float,3>();
            for(int64_t cam = 0; cam < C; ++cam) {
                h_vm_acc[cam][0][0] = 1.0f; h_vm_acc[cam][1][1] = 1.0f;
                h_vm_acc[cam][2][2] = 1.0f; h_vm_acc[cam][3][3] = 1.0f; // Identity matrix
            }
            bm_data.viewmats0_tensor.copy_(h_vm_cpu);
        }
        // bm_data.viewmats1_tensor = create_mock_gpu_at_tensor<float>({C, 4, 4}, 0.0f, "viewmats1", snapshot_path, torch::kFloat32); // Optional

        bm_data.Ks_tensor = create_mock_gpu_at_tensor<float>({C, 3, 3}, 0.0f, "Ks", snapshot_path, torch::kFloat32);
        if (C > 0 && bm_data.Ks_tensor.numel() > 0) {
             at::Tensor h_Ks_cpu = bm_data.Ks_tensor.cpu();
             auto h_Ks_acc = h_Ks_cpu.accessor<float,3>();
            for(int64_t cam = 0; cam < C; ++cam) {
                h_Ks_acc[cam][0][0] = bm_data.image_width / 2.0f;    // fx
                h_Ks_acc[cam][1][1] = bm_data.image_height / 2.0f;   // fy
                h_Ks_acc[cam][0][2] = bm_data.image_width / 2.0f - 0.5f;    // cx
                h_Ks_acc[cam][1][2] = bm_data.image_height / 2.0f - 0.5f;   // cy
                h_Ks_acc[cam][2][2] = 1.0f;
            }
            bm_data.Ks_tensor.copy_(h_Ks_cpu);
        }

        // Set default enum values for world kernel - USER TODO: Load these from snapshot metadata
        bm_data.camera_model_type = gsplat::CameraModelType::PINHOLE;
        bm_data.ut_params = {};
        bm_data.rs_type = gsplat::ShutterType::GLOBAL;

        // bm_data.radial_coeffs_tensor = create_mock_gpu_at_tensor<float>({C, 6}, 0.0f, "radial_coeffs", snapshot_path, torch::kFloat32); // Example for Pinhole
        // bm_data.tangential_coeffs_tensor = create_mock_gpu_at_tensor<float>({C, 2}, 0.0f, "tangential_coeffs", snapshot_path, torch::kFloat32);
        // bm_data.thin_prism_coeffs_tensor = create_mock_gpu_at_tensor<float>({C, 4}, 0.0f, "thin_prism_coeffs", snapshot_path, torch::kFloat32);


        // Output Gradients for world kernel
        bm_data.v_means3d_tensor = create_mock_gpu_at_tensor<float>({N, 3}, 0.0f, "v_means3d_output_buffer", snapshot_path, torch::kFloat32, false);
        bm_data.v_quats_tensor = create_mock_gpu_at_tensor<float>({N, 4}, 0.0f, "v_quats_output_buffer", snapshot_path, torch::kFloat32, false);
        bm_data.v_scales_tensor = create_mock_gpu_at_tensor<float>({N, 3}, 0.0f, "v_scales_output_buffer", snapshot_path, torch::kFloat32, false);
    } else {
        std::cerr << "ERROR: Invalid benchmark_kernel_type: " << params.benchmark.kernel_type << std::endl;
        return false;
    }

    std::cout << "INFO: Finished (mock) loading of benchmark data using at::Tensor." << std::endl;
    std::cout << "      USER ACTION REQUIRED: Replace mock data loading in src/benchmark_utils.cpp with actual snapshot loading for meaningful results." << std::endl;
    return true;
}

} // namespace benchmark
} // namespace gs
