#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [C, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [C, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [C, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [C, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [C, CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t
        *__restrict__ render_alphas,      // [C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [C, image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [C, image_height,
                                                  // image_width, CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,  // [C, N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,      // [C, N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,       // [C, N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,   // [C, N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities // [C, N] or [nnz]
) {
    auto block = cg::this_thread_block();
    uint32_t camera_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    const uint32_t kernel_block_size = block.size();
    uint32_t tr = block.thread_rank();

    tile_offsets += camera_id * tile_height * tile_width;
    render_alphas += camera_id * image_height * image_width;
    last_ids += camera_id * image_height * image_width;
    v_render_colors += camera_id * image_height * image_width * CDIM;
    v_render_alphas += camera_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += camera_id * CDIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    bool inside = (i < image_height && j < image_width);

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t num_batches =
        (range_end - range_start + kernel_block_size - 1) / kernel_block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[kernel_block_size]);
    vec3 *conic_batch = reinterpret_cast<vec3 *>(&xy_opacity_batch[kernel_block_size]);
    float *rgbs_batch = (float *)&conic_batch[kernel_block_size];
    float *s_reduction_buffer = (float *)&rgbs_batch[kernel_block_size * CDIM];

    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    float buffer[CDIM] = {0.f};
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    cg::thread_block_tile<min(32u, (unsigned int)kernel_block_size)> warp = cg::tiled_partition<min(32u, (unsigned int)kernel_block_size)>(block);
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        const int32_t batch_end_idx = range_end - 1 - kernel_block_size * b;
        const int32_t current_batch_size = min(kernel_block_size, batch_end_idx + 1 - range_start);
        const int32_t load_idx = batch_end_idx - tr;

        if (load_idx >= range_start) {
            int32_t g_load = flatten_ids[load_idx];
            id_batch[tr] = g_load;
            const vec2 xy = means2d[g_load];
            const float opac = opacities[g_load];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g_load];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g_load * CDIM + k];
            }
        }
        block.sync();

        for (uint32_t t_loop_idx = max(0, batch_end_idx - warp_bin_final); t_loop_idx < current_batch_size; ++t_loop_idx) {
            int32_t g_shared_idx = t_loop_idx;
            int32_t g_global_idx = id_batch[g_shared_idx];

            bool valid_pixel_for_gaussian = inside;
            if (batch_end_idx - g_shared_idx > bin_final) {
                valid_pixel_for_gaussian = false;
            }

            float alpha = 0.f, opac_sh = 0.f, vis = 0.f;
            vec2 delta = {0.f, 0.f};
            vec3 conic_sh = {0.f, 0.f, 0.f};

            if (valid_pixel_for_gaussian) {
                conic_sh = conic_batch[g_shared_idx];
                vec3 xy_opac_sh = xy_opacity_batch[g_shared_idx];
                opac_sh = xy_opac_sh.z;
                delta = {xy_opac_sh.x - px, xy_opac_sh.y - py};
                float sigma = 0.5f * (conic_sh.x * delta.x * delta.x +
                                      conic_sh.z * delta.y * delta.y) +
                              conic_sh.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac_sh * vis);
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    valid_pixel_for_gaussian = false;
                }
            }

            float local_v_rgb[CDIM];
            for(unsigned int k_init=0; k_init<CDIM; ++k_init) local_v_rgb[k_init] = 0.f;
            vec3 local_v_conic = {0.f, 0.f, 0.f};
            vec2 local_v_xy = {0.f, 0.f};
            vec2 local_v_xy_abs = {0.f, 0.f};
            float local_v_opacity = 0.f;

            if (valid_pixel_for_gaussian) {
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    local_v_rgb[k] = fac * v_render_c[k];
                }
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[g_shared_idx * CDIM + k] * T - buffer[k] * ra) * v_render_c[k];
                }
                v_alpha += T_final * ra * v_render_a;
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac_sh * vis <= 0.999f) {
                    const float v_sigma = -opac_sh * vis * v_alpha;
                    local_v_conic = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    local_v_xy = {
                        v_sigma * (conic_sh.x * delta.x + conic_sh.y * delta.y),
                        v_sigma * (conic_sh.y * delta.x + conic_sh.z * delta.y)
                    };
                    if (v_means2d_abs != nullptr) {
                        local_v_xy_abs = {abs(local_v_xy.x), abs(local_v_xy.y)};
                    }
                    local_v_opacity = vis * v_alpha;
                }
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[g_shared_idx * CDIM + k] * fac;
                }
            }

            #define BLOCK_REDUCE_SUM_COMPONENT_RASTERIZE(component_value) \
                s_reduction_buffer[tr] = (valid_pixel_for_gaussian) ? (component_value) : 0.0f; \
                block.sync(); \
                for (unsigned int s_offset = kernel_block_size / 2; s_offset > 0; s_offset >>= 1) { \
                    if (tr < s_offset) { \
                        s_reduction_buffer[tr] += s_reduction_buffer[tr + s_offset]; \
                    } \
                    block.sync(); \
                }

            float sum_v_opacity_val;
            vec2 sum_v_xy_val;
            vec2 sum_v_xy_abs_val;
            vec3 sum_v_conic_val;
            float sum_v_rgb_val[CDIM];

            BLOCK_REDUCE_SUM_COMPONENT_RASTERIZE(local_v_opacity);
            if (tr == 0) sum_v_opacity_val = s_reduction_buffer[0];

            BLOCK_REDUCE_SUM_COMPONENT_RASTERIZE(local_v_xy.x);
            if (tr == 0) sum_v_xy_val.x = s_reduction_buffer[0];
            BLOCK_REDUCE_SUM_COMPONENT_RASTERIZE(local_v_xy.y);
            if (tr == 0) sum_v_xy_val.y = s_reduction_buffer[0];

            if (v_means2d_abs != nullptr) {
                BLOCK_REDUCE_SUM_COMPONENT_RASTERIZE(local_v_xy_abs.x);
                if (tr == 0) sum_v_xy_abs_val.x = s_reduction_buffer[0];
                BLOCK_REDUCE_SUM_COMPONENT_RASTERIZE(local_v_xy_abs.y);
                if (tr == 0) sum_v_xy_abs_val.y = s_reduction_buffer[0];
            }

            BLOCK_REDUCE_SUM_COMPONENT_RASTERIZE(local_v_conic.x);
            if (tr == 0) sum_v_conic_val.x = s_reduction_buffer[0];
            BLOCK_REDUCE_SUM_COMPONENT_RASTERIZE(local_v_conic.y);
            if (tr == 0) sum_v_conic_val.y = s_reduction_buffer[0];
            BLOCK_REDUCE_SUM_COMPONENT_RASTERIZE(local_v_conic.z);
            if (tr == 0) sum_v_conic_val.z = s_reduction_buffer[0];

            for (uint32_t k_reduce = 0; k_reduce < CDIM; ++k_reduce) {
                BLOCK_REDUCE_SUM_COMPONENT_RASTERIZE(local_v_rgb[k_reduce]);
                if (tr == 0) sum_v_rgb_val[k_reduce] = s_reduction_buffer[0];
            }

            if (tr == 0) {
                if (sum_v_opacity_val != 0.0f) gpuAtomicAdd(v_opacities + g_global_idx, sum_v_opacity_val);

                if (sum_v_xy_val.x != 0.0f) gpuAtomicAdd((scalar_t *)v_means2d + g_global_idx * 2 + 0, sum_v_xy_val.x);
                if (sum_v_xy_val.y != 0.0f) gpuAtomicAdd((scalar_t *)v_means2d + g_global_idx * 2 + 1, sum_v_xy_val.y);

                if (v_means2d_abs != nullptr) {
                    if (sum_v_xy_abs_val.x != 0.0f) gpuAtomicAdd((scalar_t *)v_means2d_abs + g_global_idx * 2 + 0, sum_v_xy_abs_val.x);
                    if (sum_v_xy_abs_val.y != 0.0f) gpuAtomicAdd((scalar_t *)v_means2d_abs + g_global_idx * 2 + 1, sum_v_xy_abs_val.y);
                }

                if (sum_v_conic_val.x != 0.0f) gpuAtomicAdd((scalar_t *)v_conics + g_global_idx * 3 + 0, sum_v_conic_val.x);
                if (sum_v_conic_val.y != 0.0f) gpuAtomicAdd((scalar_t *)v_conics + g_global_idx * 3 + 1, sum_v_conic_val.y);
                if (sum_v_conic_val.z != 0.0f) gpuAtomicAdd((scalar_t *)v_conics + g_global_idx * 3 + 2, sum_v_conic_val.z);

                for (uint32_t k_atomic = 0; k_atomic < CDIM; ++k_atomic) {
                    if (sum_v_rgb_val[k_atomic] != 0.0f) gpuAtomicAdd((scalar_t *)v_colors + g_global_idx * CDIM + k_atomic, sum_v_rgb_val[k_atomic]);
                }
            }
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [C, N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [C, N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [C, 3]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [C, image_height, image_width, 1]
    const at::Tensor last_ids,      // [C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [C, image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [C, image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [C, N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [C, N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [C, N] or [nnz]
) {
    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    int64_t kernel_block_size_host = static_cast<int64_t>(tile_size) * tile_size;
    int64_t shmem_size =
        kernel_block_size_host * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM) +
        kernel_block_size_host * sizeof(float); // Added space for s_reduction_buffer


    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<vec2 *>(
                      v_means2d_abs.value().data_ptr<float>()
                  )
                : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>()
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dgs_bwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor conics,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const at::Tensor render_alphas,                                        \
        const at::Tensor last_ids,                                             \
        const at::Tensor v_render_colors,                                      \
        const at::Tensor v_render_alphas,                                      \
        at::optional<at::Tensor> v_means2d_abs,                                \
        at::Tensor v_means2d,                                                  \
        at::Tensor v_conics,                                                   \
        at::Tensor v_colors,                                                   \
        at::Tensor v_opacities                                                 \
    );

__INS__(1)
__INS__(2)
__INS__(3)
__INS__(4)
__INS__(5)
__INS__(8)
__INS__(9)
__INS__(16)
__INS__(17)
__INS__(32)
__INS__(33)
__INS__(64)
__INS__(65)
__INS__(128)
__INS__(129)
__INS__(256)
__INS__(257)
__INS__(512)
__INS__(513)
#undef __INS__

} // namespace gsplat
