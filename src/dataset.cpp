#include "core/dataset.hpp"
#include "core/camera.hpp"      // For Camera::load_and_get_image
#include "core/parameters.hpp"  // For gs::param::OptimizationParameters
#include <iostream>             // For std::cout, std::endl
#include <chrono>               // For timing
#include <iomanip>              // For std::fixed, std::setprecision
#include <vector>               // For std::vector
#include <torch/torch.h>        // For torch::Tensor, torch::stack, etc.

// Implementation of try_preload_images_to_cpu
// Fully qualify the class and namespace for the method definition.
void gs::CameraDataset::try_preload_images_to_cpu(const gs::param::OptimizationParameters& optim_params, const std::string& dataset_name) {
    if (!optim_params.preload_images_to_cpu) {
        return;
    }

    std::cout << "[" << dataset_name << " dataset] Starting preloading of " << this->_indices.size() << " images to CPU RAM..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<torch::Tensor> image_list_for_stacking;
    image_list_for_stacking.reserve(this->_indices.size());

    for (size_t i = 0; i < this->_indices.size(); ++i) {
        size_t original_camera_idx = this->_indices[i];
        Camera* cam = this->_cameras[original_camera_idx].get();
        torch::Tensor img_tensor = cam->load_and_get_image(this->_datasetConfig.resolution);
        img_tensor = img_tensor.to(torch::kCPU).contiguous(); // Ensure on CPU
        image_list_for_stacking.push_back(img_tensor);

        if ((i + 1) % 100 == 0 || (i + 1) == this->_indices.size()) {
            std::cout << "[" << dataset_name << " dataset] Preloaded " << (i + 1) << "/" << this->_indices.size() << " images to CPU..." << std::endl;
        }
    }

    if (!image_list_for_stacking.empty()) {
        this->preloaded_cpu_image_data_ = torch::stack(image_list_for_stacking, 0).contiguous();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        float total_size_mb = (this->preloaded_cpu_image_data_.value().nbytes() / (1024.0f * 1024.0f));

        std::cout << "[" << dataset_name << " dataset] Finished preloading " << this->_indices.size() << " images to CPU RAM." << std::endl;
        std::cout << "  Preloaded tensor shape: " << this->preloaded_cpu_image_data_.value().sizes() << std::endl;
        std::cout << "  Estimated size: " << std::fixed << std::setprecision(2) << total_size_mb << " MB" << std::endl;
        std::cout << "  Preloading time: " << duration.count() << " ms" << std::endl;
    } else {
        std::cout << "[" << dataset_name << " dataset] No images to preload." << std::endl;
    }
}
