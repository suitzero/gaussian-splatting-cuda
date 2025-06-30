#include "core/argument_parser.hpp"
#include "core/dataset.hpp"
#include "core/mcmc.hpp"
#include "core/argument_parser.hpp"
#include "core/dataset.hpp"
#include "core/mcmc.hpp"
#include "core/parameters.hpp"
#include "core/trainer.hpp"
#include "visualizer/detail.hpp"
#include "core/benchmark_utils.hpp" // Added for benchmarking
#include "gsplat/Rasterization.h"    // For kernel launchers
#include "gsplat/Common.h"           // For CameraModelType etc.


#include <iostream>
#include <memory>
#include <thread>
#include <vector>      // For std::vector
#include <chrono>      // For std::chrono
#include <numeric>     // For std::accumulate, std::inner_product
#include <algorithm>   // For std::min_element, std::max_element
#include <cuda_runtime.h> // For cudaEvent_t

// Helper to check CUDA calls (can be moved to a common utils header if not already present)
#ifndef MAIN_CUDA_CHECK
#define MAIN_CUDA_CHECK(call)                                                \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s at line %d: %s (%d)\\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err), err);       \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)
#endif


// BENCHMARKING FUNCTION
/*
 * Runs a focused benchmark on the specified backward rasterization kernel.
 *
 * To use benchmark mode, run the executable with the following arguments:
 *   --benchmark_backward_kernel <num_iterations>
 *       Enables benchmark mode and specifies the number of timed iterations.
 *   --benchmark_input_snapshot <path_to_snapshot_dir>
 *       (Required if benchmark_backward_kernel is set)
 *       Path to a directory containing the pre-saved input tensors for the kernel.
 *       The program expects specific .pt (PyTorch tensor) files in this directory.
 *       Refer to `src/benchmark_utils.cpp` for expected filenames and instructions
 *       on how to save these snapshots from your training pipeline.
 *   --benchmark_kernel_type <"2d" | "world">
 *       (Optional, default: "2d") Specifies which kernel to benchmark:
 *       - "2d": for `launch_rasterize_to_pixels_3dgs_bwd_kernel`
 *       - "world": for `launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel`
 *   --benchmark_warmup_iterations <num_iterations>
 *       (Optional, default: 10) Number of warmup iterations before timing starts.
 *
 * The `gs::benchmark::load_benchmark_snapshot` function in `benchmark_utils.cpp`
 * currently loads MOCK data. You MUST modify it to load your actual saved tensors
 * for meaningful benchmark results.
 */
void run_benchmark(const gs::param::TrainingParameters& params) {
    std::cout << "--- BENCHMARK MODE ---" << std::endl;
    std::cout << "Kernel type: " << params.benchmark.kernel_type << std::endl;
    std::cout << "Iterations: " << params.benchmark.backward_kernel_iterations
              << " (Warmup: " << params.benchmark.warmup_iterations << ")" << std::endl;
    std::cout << "Snapshot path: " << params.benchmark.input_snapshot_path << std::endl;

    gs::benchmark::BenchmarkData bm_data;

    if (!gs::benchmark::load_benchmark_snapshot(params.benchmark.input_snapshot_path, params, bm_data)) {
        std::cerr << "Benchmark data loading failed. Exiting." << std::endl;
        return;
    }

    cudaEvent_t start_event, stop_event;
    MAIN_CUDA_CHECK(cudaEventCreate(&start_event));
    MAIN_CUDA_CHECK(cudaEventCreate(&stop_event));
    cudaStream_t stream = 0; // Default stream

    // Warm-up runs
    std::cout << "Running " << params.benchmark.warmup_iterations << " warmup iterations for "
              << params.benchmark.kernel_type << " kernel..." << std::endl;
    for (int i = 0; i < params.benchmark.warmup_iterations; ++i) {
        if (params.benchmark.kernel_type == "2d") {
            // Assuming CDIM = 3 for v_render_colors and v_colors for benchmark simplicity
            gsplat::launch_rasterize_to_pixels_3dgs_bwd_kernel<3>(
                bm_data.means2d_tensor, bm_data.conics_tensor, bm_data.colors_in_tensor, bm_data.opacities_in_tensor,
                bm_data.backgrounds_tensor, bm_data.masks_tensor,
                bm_data.image_width, bm_data.image_height, bm_data.tile_size,
                bm_data.tile_offsets_tensor, bm_data.flatten_ids_tensor,
                bm_data.render_alphas_tensor, bm_data.last_ids_tensor,
                bm_data.v_render_colors_tensor, bm_data.v_render_alphas_tensor,
                bm_data.v_means2d_abs_tensor, bm_data.v_means2d_tensor, bm_data.v_conics_tensor,
                bm_data.v_colors_tensor, bm_data.v_opacities_tensor
            );
        } else if (params.benchmark.kernel_type == "world") {
            gsplat::launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<3>(
                bm_data.means3d_tensor, bm_data.quats_tensor, bm_data.scales_tensor,
                bm_data.colors_in_tensor, bm_data.opacities_in_tensor,
                bm_data.backgrounds_tensor, bm_data.masks_tensor,
                bm_data.image_width, bm_data.image_height, bm_data.tile_size,
                bm_data.viewmats0_tensor, bm_data.viewmats1_tensor, bm_data.Ks_tensor,
                bm_data.camera_model_type, bm_data.ut_params, bm_data.rs_type,
                bm_data.radial_coeffs_tensor, bm_data.tangential_coeffs_tensor, bm_data.thin_prism_coeffs_tensor,
                bm_data.tile_offsets_tensor, bm_data.flatten_ids_tensor,
                bm_data.render_alphas_tensor, bm_data.last_ids_tensor,
                bm_data.v_render_colors_tensor, bm_data.v_render_alphas_tensor,
                bm_data.v_means3d_tensor, bm_data.v_quats_tensor, bm_data.v_scales_tensor,
                bm_data.v_colors_tensor, bm_data.v_opacities_tensor
            );
        }
    }
    MAIN_CUDA_CHECK(cudaDeviceSynchronize());

    // Timed benchmark runs
    std::vector<float> timings_ms;
    timings_ms.reserve(params.benchmark.backward_kernel_iterations);
    std::cout << "Running " << params.benchmark.backward_kernel_iterations << " timed iterations..." << std::endl;

    auto overall_start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < params.benchmark.backward_kernel_iterations; ++i) {
        MAIN_CUDA_CHECK(cudaEventRecord(start_event, stream));
        if (params.benchmark.kernel_type == "2d") {
            gsplat::launch_rasterize_to_pixels_3dgs_bwd_kernel<3>(
                bm_data.means2d_tensor, bm_data.conics_tensor, bm_data.colors_in_tensor, bm_data.opacities_in_tensor,
                bm_data.backgrounds_tensor, bm_data.masks_tensor,
                bm_data.image_width, bm_data.image_height, bm_data.tile_size,
                bm_data.tile_offsets_tensor, bm_data.flatten_ids_tensor,
                bm_data.render_alphas_tensor, bm_data.last_ids_tensor,
                bm_data.v_render_colors_tensor, bm_data.v_render_alphas_tensor,
                bm_data.v_means2d_abs_tensor, bm_data.v_means2d_tensor, bm_data.v_conics_tensor,
                bm_data.v_colors_tensor, bm_data.v_opacities_tensor
            );
        } else if (params.benchmark.kernel_type == "world") {
             gsplat::launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<3>(
                bm_data.means3d_tensor, bm_data.quats_tensor, bm_data.scales_tensor,
                bm_data.colors_in_tensor, bm_data.opacities_in_tensor,
                bm_data.backgrounds_tensor, bm_data.masks_tensor,
                bm_data.image_width, bm_data.image_height, bm_data.tile_size,
                bm_data.viewmats0_tensor, bm_data.viewmats1_tensor, bm_data.Ks_tensor,
                bm_data.camera_model_type, bm_data.ut_params, bm_data.rs_type,
                bm_data.radial_coeffs_tensor, bm_data.tangential_coeffs_tensor, bm_data.thin_prism_coeffs_tensor,
                bm_data.tile_offsets_tensor, bm_data.flatten_ids_tensor,
                bm_data.render_alphas_tensor, bm_data.last_ids_tensor,
                bm_data.v_render_colors_tensor, bm_data.v_render_alphas_tensor,
                bm_data.v_means3d_tensor, bm_data.v_quats_tensor, bm_data.v_scales_tensor,
                bm_data.v_colors_tensor, bm_data.v_opacities_tensor
            );
        }
        MAIN_CUDA_CHECK(cudaEventRecord(stop_event, stream));
        MAIN_CUDA_CHECK(cudaEventSynchronize(stop_event));
        float milliseconds = 0;
        MAIN_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        timings_ms.push_back(milliseconds);
    }
    auto overall_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> overall_duration = overall_end_time - overall_start_time;

    // Report Performance
    double total_time_ms = std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0);
    double avg_time_ms = total_time_ms / timings_ms.size();
    double min_time_ms = *std::min_element(timings_ms.begin(), timings_ms.end());
    double max_time_ms = *std::max_element(timings_ms.begin(), timings_ms.end());
    double sq_sum = std::inner_product(timings_ms.begin(), timings_ms.end(), timings_ms.begin(), 0.0);
    double stdev_ms = std::sqrt(sq_sum / timings_ms.size() - avg_time_ms * avg_time_ms);

    std::cout << "\n--- Benchmark Performance Results (" << params.benchmark.kernel_type << " kernel) ---" << std::endl;
    std::cout << "Total time for " << timings_ms.size() << " iterations: " << overall_duration.count() << " ms" << std::endl;
    std::cout << "Average kernel execution time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Min kernel execution time: " << min_time_ms << " ms" << std::endl;
    std::cout << "Max kernel execution time: " << max_time_ms << " ms" << std::endl;
    std::cout << "Stddev kernel execution time: " << stdev_ms << " ms" << std::endl;
    std::cout << "Throughput (iterations/sec): " << (timings_ms.size() / (total_time_ms / 1000.0)) << std::endl;
    std::cout << "--- END BENCHMARK ---" << std::endl;

    MAIN_CUDA_CHECK(cudaEventDestroy(start_event));
    MAIN_CUDA_CHECK(cudaEventDestroy(stop_event));
    // BenchmarkData holds at::Tensor objects which manage their own memory.
}


int main(int argc, char* argv[]) {
    try {
        //----------------------------------------------------------------------
        // 1. Parse arguments and load parameters in one step
        //----------------------------------------------------------------------
        // The gs::args::parse_args_and_params function will populate params.benchmark.enabled
        // and other benchmark-specific parameters if the corresponding CLI args are provided.
        auto params = gs::args::parse_args_and_params(argc, argv);

        //----------------------------------------------------------------------
        // Check if Benchmark Mode is Enabled
        //----------------------------------------------------------------------
        if (params.benchmark.enabled) {
            if (params.benchmark.input_snapshot_path.empty()) {
                std::cerr << "ERROR: Benchmark mode enabled, but --benchmark_input_snapshot path is missing." << std::endl;
                std::cerr << "Please provide a path to a directory containing saved input tensors for the benchmark." << std::endl;
                std::cerr << "See comments in src/main.cpp (run_benchmark function) and src/benchmark_utils.cpp for details." << std::endl;
                return -1;
            }
            std::cout << "Benchmark mode activated via command-line arguments." << std::endl;
            run_benchmark(params);
            return 0; // Exit after benchmark
        }

        //----------------------------------------------------------------------
        // 2. Save training configuration to output directory
        //----------------------------------------------------------------------
        gs::param::save_training_parameters_to_json(params, params.dataset.output_path);

        //----------------------------------------------------------------------
        // 3. Create dataset from COLMAP
        //----------------------------------------------------------------------
        auto [dataset, scene_center] = create_dataset_from_colmap(params.dataset);

        //----------------------------------------------------------------------
        // 4. Model initialisation
        //----------------------------------------------------------------------
        auto splat_data = SplatData::init_model_from_pointcloud(params, scene_center);

        //----------------------------------------------------------------------
        // 5. Create strategy
        //----------------------------------------------------------------------
        auto strategy = std::make_unique<MCMC>(std::move(splat_data));

        //----------------------------------------------------------------------
        // 6. Create trainer
        //----------------------------------------------------------------------
        auto trainer = std::make_unique<gs::Trainer>(dataset, std::move(strategy), params);

        //----------------------------------------------------------------------
        // 7. Start training based on visualization mode
        //----------------------------------------------------------------------
        if (params.optimization.enable_viz) {
            // GUI Mode: Create viewer and run it in main thread
            auto viewer = trainer->create_and_get_viewer();
            if (viewer) {
                // Start training in a separate thread
                std::thread training_thread([&trainer]() {
                    try {
                        trainer->train();
                    } catch (const std::exception& e) {
                        std::cerr << "Training thread error: " << e.what() << std::endl;
                    }
                });

                // Run GUI in main thread (blocking)
                viewer->run();

                // After viewer closes, ensure training is stopped
                if (trainer->is_running()) {
                    std::cout << "Main: Requesting training stop..." << std::endl;
                    trainer->request_stop();
                }

                // Wait for training thread to complete
                if (training_thread.joinable()) {
                    std::cout << "Main: Waiting for training thread to finish..." << std::endl;
                    training_thread.join();
                    std::cout << "Main: Training thread finished." << std::endl;
                }
            } else {
                std::cerr << "Failed to create viewer" << std::endl;
                return -1;
            }
        } else {
            // Headless Mode: Run training in main thread
            trainer->train();
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}