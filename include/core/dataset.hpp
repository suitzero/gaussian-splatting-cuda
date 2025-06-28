#pragma once

#include "core/camera.hpp"
#include "core/colmap_reader.hpp"
#include "core/parameters.hpp"
#include <memory>
#include <torch/torch.h>
#include <vector>
#include <iostream> // For std::cout
#include <chrono>   // For timing
#include <iomanip>  // For std::fixed, std::setprecision

// Camera with loaded image
struct CameraWithImage {
    Camera* camera;
    torch::Tensor image;
};

using CameraExample = torch::data::Example<CameraWithImage, torch::Tensor>;

class CameraDataset : public torch::data::Dataset<CameraDataset, CameraExample> {
public:
    enum class Split {
        TRAIN,
        VAL,
        ALL
    };

    CameraDataset(std::vector<std::shared_ptr<Camera>> cameras,
                  const gs::param::DatasetConfig& params,
                  Split split = Split::ALL)
        : _cameras(std::move(cameras)),
          _datasetConfig(params),
          _split(split) {

        // Create indices based on split
        _indices.clear();
        for (size_t i = 0; i < _cameras.size(); ++i) {
            const bool is_test = (i % params.test_every) == 0;

            if (_split == Split::ALL ||
                (_split == Split::TRAIN && !is_test) ||
                (_split == Split::VAL && is_test)) {
                _indices.push_back(i);
            }
        }

        std::cout << "Dataset created with " << _indices.size()
                  << " images (split: " << static_cast<int>(_split) << ")" << std::endl;
    }
    // Default copy constructor works with shared_ptr
    CameraDataset(const CameraDataset&) = default;
    CameraDataset(CameraDataset&&) noexcept = default;
    CameraDataset& operator=(CameraDataset&&) noexcept = default;
    CameraDataset& operator=(const CameraDataset&) = default;

    CameraExample get(size_t index) override {
        if (index >= _indices.size()) {
            throw std::out_of_range("Dataset index out of range");
        }

        size_t original_camera_idx = _indices[index]; // 'index' is from sampler, map to original camera list index
        Camera* cam_ptr = _cameras[original_camera_idx].get();

        if (preloaded_cpu_image_data_.has_value()) {
            // 'index' directly maps to the preloaded tensor's first dimension
            // as preloaded_cpu_image_data_ was created based on iterating _indices
            torch::Tensor image_to_return = preloaded_cpu_image_data_.value().index({(int64_t)index});
            return {{cam_ptr, std::move(image_to_return)}, torch::empty({})};
        } else {
            // Original path: load image on demand
            torch::Tensor image = cam_ptr->load_and_get_image(_datasetConfig.resolution);
            return {{cam_ptr, std::move(image)}, torch::empty({})};
        }
    }

    torch::optional<size_t> size() const override {
        return _indices.size();
    }

    const std::vector<std::shared_ptr<Camera>>& get_cameras() const {
        return _cameras;
    }

    Split get_split() const { return _split; }

    // Method to preload images to CPU RAM
    void try_preload_images_to_cpu(const gs::param::OptimizationParameters& optim_params, const std::string& dataset_name);

private:
    std::vector<std::shared_ptr<Camera>> _cameras;
    const gs::param::DatasetConfig& _datasetConfig;
    Split _split;
    std::vector<size_t> _indices;
    torch::optional<torch::Tensor> preloaded_cpu_image_data_; // Holds all images for this dataset split if preloaded
};

// Note: Implementation of CameraDataset::try_preload_images_to_cpu is now in src/dataset.cpp

inline std::tuple<std::shared_ptr<CameraDataset>, torch::Tensor> create_dataset_from_colmap(
    const gs::param::DatasetConfig& datasetConfig) {

    if (!std::filesystem::exists(datasetConfig.data_path)) {
        throw std::runtime_error("Data path does not exist: " +
                                 datasetConfig.data_path.string());
    }

    // Read COLMAP data with specified images folder
    auto [camera_infos, scene_center] = read_colmap_cameras_and_images(
        datasetConfig.data_path, datasetConfig.images);

    std::vector<std::shared_ptr<Camera>> cameras;
    cameras.reserve(camera_infos.size());

    for (size_t i = 0; i < camera_infos.size(); ++i) {
        const auto& info = camera_infos[i];

        auto cam = std::make_shared<Camera>(
            info._R,
            info._T,
            info._fov_x,
            info._fov_y,
            info._image_name,
            info._image_path,
            info._width,
            info._height,
            static_cast<int>(i));

        cameras.push_back(std::move(cam));
    }

    // Create dataset with ALL images
    auto dataset = std::make_shared<CameraDataset>(
        std::move(cameras), datasetConfig, CameraDataset::Split::ALL);

    return {dataset, scene_center};
}

inline auto create_dataloader_from_dataset(
    std::shared_ptr<CameraDataset> dataset,
    int batch_size,
    int num_workers = 4) {

    const size_t dataset_size = dataset->size().value();

    auto loader_options = torch::data::DataLoaderOptions()
                              .batch_size(batch_size)
                              .workers(num_workers)
                              .enforce_ordering(false);

    // Removed pin_memory(true) call due to API incompatibility with torch 2.7.0 as reported.
    // DataLoader will use its default behavior regarding pinned memory.
    // Efficient non-blocking transfers still rely on the source tensor being in pinned memory,
    // which is not explicitly done for preloaded_cpu_image_data_ yet, but DataLoader
    // workers might use pinned memory internally if possible.

    return torch::data::make_data_loader(
        *dataset,
        torch::data::samplers::RandomSampler(dataset_size),
        loader_options);
}