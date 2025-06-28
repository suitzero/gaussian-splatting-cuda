#include "core/argument_parser.hpp"
#include "core/dataset.hpp"
#include "core/mcmc.hpp"
#include "core/parameters.hpp"
#include "core/trainer.hpp"
#include "visualizer/detail.hpp"
#include <iostream>
#include <memory>
#include <thread>

int main(int argc, char* argv[]) {
    try {
        //----------------------------------------------------------------------
        // 1. Parse arguments and load parameters in one step
        //----------------------------------------------------------------------
        const auto params = gs::args::parse_args_and_params(argc, argv);

        //----------------------------------------------------------------------
        // 2. Save training configuration to output directory
        //----------------------------------------------------------------------
        gs::param::save_training_parameters_to_json(params, params.dataset.output_path);

        //----------------------------------------------------------------------
        // 3. Create dataset from COLMAP
        //----------------------------------------------------------------------
        auto [dataset, scene_center] = create_dataset_from_colmap(params.dataset); // scene_center is a CPU tensor here

        //----------------------------------------------------------------------
        // Determine target CUDA device for model initialization and training
        //----------------------------------------------------------------------
        torch::Device device = torch::kCPU; // Default to CPU if CUDA not available or not used
        if (torch::cuda::is_available()) {
            int device_id = params.optimization.cuda_device_id;
            int num_cuda_devices = torch::cuda::device_count();
            if (device_id < 0 || device_id >= num_cuda_devices) {
                std::cerr << "Warning (main.cpp): Invalid cuda_device_id " << device_id
                          << ". Available devices: 0-" << (num_cuda_devices - 1)
                          << ". Defaulting to device 0 for model initialization." << std::endl;
                device_id = 0;
            }
            device = torch::Device(torch::kCUDA, device_id);
            std::cout << "Main.cpp: Target CUDA device for model initialization: " << device_id
                      << " (" << torch::cuda::get_device_name(device_id) << ")" << std::endl;
        } else {
            // This case should ideally be caught by Trainer later, but good to be aware.
            // If no CUDA, SplatData init might fail if it expects CUDA tensors.
            // The current SplatData::init_model_from_pointcloud is now device-aware.
            std::cout << "Main.cpp: CUDA not available. Model initialization will use CPU." << std::endl;
        }
         // scene_center is moved to the target 'device' inside init_model_from_pointcloud

        //----------------------------------------------------------------------
        // 4. Model initialisation
        //----------------------------------------------------------------------
        auto splat_data = SplatData::init_model_from_pointcloud(params, scene_center, device);

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