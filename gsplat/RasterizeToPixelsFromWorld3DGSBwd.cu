#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"
#include "Cameras.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_from_world_3dgs_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec3 *__restrict__ means,       // [N, 3]
    const vec4 *__restrict__ quats,       // [N, 4]
    const vec3 *__restrict__ scales,      // [N, 3]
    const scalar_t *__restrict__ colors,      // [C, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [C, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [C, CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // camera model
    const scalar_t *__restrict__ viewmats0, // [C, 4, 4]
    const scalar_t *__restrict__ viewmats1, // [C, 4, 4] optional for rolling shutter
    const scalar_t *__restrict__ Ks,        // [C, 3, 3]
    const CameraModelType camera_model_type,
    // uncented transform
    const UnscentedTransformParameters ut_params,    
    const ShutterType rs_type,
    const scalar_t *__restrict__ radial_coeffs, // [C, 6] or [C, 4] optional
    const scalar_t *__restrict__ tangential_coeffs, // [C, 2] optional
    const scalar_t *__restrict__ thin_prism_coeffs, // [C, 2] optional
    // intersections
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
    vec3 *__restrict__ v_means,      // [N, 3]
    vec4 *__restrict__ v_quats,       // [N, 4]
    vec3 *__restrict__ v_scales,      // [N, 3]
    scalar_t *__restrict__ v_colors,   // [C, N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities // [C, N] or [nnz]
) {
    auto block = cg::this_thread_block();
    uint32_t cid = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    const uint32_t kernel_block_size = block.size();
    uint32_t tr = block.thread_rank();

    tile_offsets += cid * tile_height * tile_width;
    render_alphas += cid * image_height * image_width;
    last_ids += cid * image_height * image_width;
    v_render_colors += cid * image_height * image_width * CDIM;
    v_render_alphas += cid * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += cid * CDIM;
    }
    if (masks != nullptr) {
        masks += cid * tile_height * tile_width;
    }

    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    auto rs_params = RollingShutterParameters(
        viewmats0 + cid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + cid * 16
    );
    const vec2 focal_length = {Ks[cid * 9 + 0], Ks[cid * 9 + 4]};
    const vec2 principal_point = {Ks[cid * 9 + 2], Ks[cid * 9 + 5]};
    
    WorldRay ray;
    if (camera_model_type == CameraModelType::PINHOLE) {
        if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
            PerfectPinholeCameraModel::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.focal_length = { focal_length.x, focal_length.y };
            PerfectPinholeCameraModel camera_model(cm_params);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        } else {
            OpenCVPinholeCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.focal_length = { focal_length.x, focal_length.y };
            if (radial_coeffs != nullptr) {
                cm_params.radial_coeffs = make_array<float, 6>(radial_coeffs + cid * 6);
            }
            if (tangential_coeffs != nullptr) {
                cm_params.tangential_coeffs = make_array<float, 2>(tangential_coeffs + cid * 2);
            }
            if (thin_prism_coeffs != nullptr) {
                cm_params.thin_prism_coeffs = make_array<float, 4>(thin_prism_coeffs + cid * 4);
            }
            OpenCVPinholeCameraModel camera_model(cm_params);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        }
    } else if (camera_model_type == CameraModelType::FISHEYE) {
        OpenCVFisheyeCameraModel<>::Parameters cm_params = {};
        cm_params.resolution = {image_width, image_height};
        cm_params.shutter_type = rs_type;
        cm_params.principal_point = { principal_point.x, principal_point.y };
        cm_params.focal_length = { focal_length.x, focal_length.y };
        if (radial_coeffs != nullptr) {
            cm_params.radial_coeffs = make_array<float, 4>(radial_coeffs + cid * 4);
        }
        OpenCVFisheyeCameraModel camera_model(cm_params);
        ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
    } else {
        assert(false);
        return;
    }
    const vec3 ray_d = ray.ray_dir;
    const vec3 ray_o = ray.ray_org;

    bool valid_pixel_for_gaussian_initial = (i < image_height && j < image_width) && ray.valid_flag; // Renamed 'done'

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t num_batches =
        (range_end - range_start + kernel_block_size - 1) / kernel_block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    vec4 *xyz_opacity_batch = reinterpret_cast<vec4 *>(&id_batch[kernel_block_size]);
    vec3 *scale_batch = reinterpret_cast<vec3 *>(&xyz_opacity_batch[kernel_block_size]);
    vec4 *quat_batch = reinterpret_cast<vec4 *>(&scale_batch[kernel_block_size]);
    float *rgbs_batch = (float *)&quat_batch[kernel_block_size];
    float *s_reduction_buffer = (float *)&rgbs_batch[kernel_block_size * CDIM];

    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    float buffer[CDIM] = {0.f};
    const int32_t bin_final = valid_pixel_for_gaussian_initial ? last_ids[pix_id] : 0;

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
            const vec3 xyz_load = means[g_load]; // Renamed xyz to xyz_load to avoid conflict
            const float opac_load = opacities[g_load]; // Renamed opac to opac_load
            xyz_opacity_batch[tr] = {xyz_load.x, xyz_load.y, xyz_load.z, opac_load};
            scale_batch[tr] = scales[g_load];
            quat_batch[tr] = quats[g_load];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g_load * CDIM + k];
            }
        }
        block.sync();

        for (uint32_t t_loop_idx = max(0, batch_end_idx - warp_bin_final); t_loop_idx < current_batch_size; ++t_loop_idx) {
            int32_t g_shared_idx = t_loop_idx;
            int32_t g_global_idx = id_batch[g_shared_idx];

            bool valid_pixel_for_gaussian = valid_pixel_for_gaussian_initial; // Start with initial validity
            if (batch_end_idx - g_shared_idx > bin_final) {
                valid_pixel_for_gaussian = false;
            }

            float alpha_calc = 0.f, opac_sh = 0.f, vis_calc = 0.f; // Renamed alpha, opac, vis
            mat3 R_calc, S_calc, Mt_calc; // Renamed R, S, Mt
            vec3 xyz_sh, scale_sh; // Renamed xyz, scale
            vec4 quat_sh; // Renamed quat
            vec3 o_minus_mu_calc, gro_calc, grd_calc, grd_n_calc, gcrod_calc; // Renamed variables
            float grayDist_calc = 0.f, power_calc = 0.f; // Renamed grayDist, power


            if (valid_pixel_for_gaussian) {
                const vec4 xyz_opac_sh = xyz_opacity_batch[g_shared_idx];
                opac_sh = xyz_opac_sh[3];
                xyz_sh = {xyz_opac_sh[0], xyz_opac_sh[1], xyz_opac_sh[2]};
                scale_sh = scale_batch[g_shared_idx];
                quat_sh = quat_batch[g_shared_idx];
                
                R_calc = quat_to_rotmat(quat_sh);
                S_calc = mat3(
                    1.0f / scale_sh[0], 0.f, 0.f,
                    0.f, 1.0f / scale_sh[1], 0.f,
                    0.f, 0.f, 1.0f / scale_sh[2]
                );
                Mt_calc = glm::transpose(R_calc * S_calc);
                o_minus_mu_calc = ray_o - xyz_sh;
                gro_calc = Mt_calc * o_minus_mu_calc;
                grd_calc = Mt_calc * ray_d;
                grd_n_calc = safe_normalize(grd_calc);
                gcrod_calc = glm::cross(grd_n_calc, gro_calc);
                grayDist_calc = glm::dot(gcrod_calc, gcrod_calc);
                power_calc = -0.5f * grayDist_calc;

                vis_calc = __expf(power_calc);
                alpha_calc = min(0.999f, opac_sh * vis_calc);
                if (power_calc > 0.f || alpha_calc < 1.f / 255.f) {
                    valid_pixel_for_gaussian = false;
                }
            }

            float local_v_rgb[CDIM]; for(unsigned int k_init=0; k_init<CDIM; ++k_init) local_v_rgb[k_init] = 0.f;
            vec3 local_v_mean = {0.f, 0.f, 0.f};
            vec3 local_v_scale = {0.f, 0.f, 0.f};
            vec4 local_v_quat = {0.f, 0.f, 0.f, 0.f};
            float local_v_opacity = 0.f;

            if (valid_pixel_for_gaussian) {
                float ra = 1.0f / (1.0f - alpha_calc);
                T *= ra;
                const float fac = alpha_calc * T;
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

                if (opac_sh * vis_calc <= 0.999f) {
                    const float v_vis_val = opac_sh * v_alpha; // Renamed v_vis
                    float v_gradDist_val = -0.5f * vis_calc * v_vis_val; // Renamed v_gradDist
                    vec3 v_gcrod_val = 2.0f * v_gradDist_val * gcrod_calc; // Renamed v_gcrod
                    vec3 v_grd_n_val = - glm::cross(v_gcrod_val, gro_calc); // Renamed v_grd_n
                    vec3 v_gro_val = glm::cross(v_gcrod_val, grd_n_calc); // Renamed v_gro
                    vec3 v_grd_val = safe_normalize_bw(grd_calc, v_grd_n_val); // Renamed v_grd
                    mat3 v_Mt_val = glm::outerProduct(v_grd_val, ray_d) +
                        glm::outerProduct(v_gro_val, o_minus_mu_calc); // Renamed v_Mt
                    vec3 v_o_minus_mu_val = glm::transpose(Mt_calc) * v_gro_val; // Renamed v_o_minus_mu

                    local_v_mean += -v_o_minus_mu_val;
                    quat_scale_to_preci_half_vjp(
                        quat_sh, scale_sh, R_calc, glm::transpose(v_Mt_val), local_v_quat, local_v_scale
                    );
                    local_v_opacity = vis_calc * v_alpha;
                }
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[g_shared_idx * CDIM + k] * fac;
                }
            }

            #define BLOCK_REDUCE_SUM_COMPONENT_WORLD(component_value) \
                s_reduction_buffer[tr] = (valid_pixel_for_gaussian) ? (component_value) : 0.0f; \
                block.sync(); \
                for (unsigned int s_offset = kernel_block_size / 2; s_offset > 0; s_offset >>= 1) { \
                    if (tr < s_offset) { \
                        s_reduction_buffer[tr] += s_reduction_buffer[tr + s_offset]; \
                    } \
                    block.sync(); \
                }

            float sum_v_opacity_val;
            vec3 sum_v_mean_val;
            vec3 sum_v_scale_val;
            vec4 sum_v_quat_val;
            float sum_v_rgb_val[CDIM];

            BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_opacity);
            if (tr == 0) sum_v_opacity_val = s_reduction_buffer[0];

            BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_mean.x); if (tr == 0) sum_v_mean_val.x = s_reduction_buffer[0];
            BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_mean.y); if (tr == 0) sum_v_mean_val.y = s_reduction_buffer[0];
            BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_mean.z); if (tr == 0) sum_v_mean_val.z = s_reduction_buffer[0];

            BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_scale.x); if (tr == 0) sum_v_scale_val.x = s_reduction_buffer[0];
            BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_scale.y); if (tr == 0) sum_v_scale_val.y = s_reduction_buffer[0];
            BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_scale.z); if (tr == 0) sum_v_scale_val.z = s_reduction_buffer[0];

            BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_quat.x); if (tr == 0) sum_v_quat_val.x = s_reduction_buffer[0];
            BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_quat.y); if (tr == 0) sum_v_quat_val.y = s_reduction_buffer[0];
            BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_quat.z); if (tr == 0) sum_v_quat_val.z = s_reduction_buffer[0];
            BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_quat.w); if (tr == 0) sum_v_quat_val.w = s_reduction_buffer[0];

            for (uint32_t k_reduce = 0; k_reduce < CDIM; ++k_reduce) {
                BLOCK_REDUCE_SUM_COMPONENT_WORLD(local_v_rgb[k_reduce]);
                if (tr == 0) sum_v_rgb_val[k_reduce] = s_reduction_buffer[0];
            }

            if (tr == 0) {
                if (sum_v_opacity_val != 0.0f) gpuAtomicAdd(v_opacities + g_global_idx, sum_v_opacity_val);

                if (sum_v_mean_val.x != 0.0f) gpuAtomicAdd((scalar_t*)v_means + g_global_idx * 3 + 0, sum_v_mean_val.x);
                if (sum_v_mean_val.y != 0.0f) gpuAtomicAdd((scalar_t*)v_means + g_global_idx * 3 + 1, sum_v_mean_val.y);
                if (sum_v_mean_val.z != 0.0f) gpuAtomicAdd((scalar_t*)v_means + g_global_idx * 3 + 2, sum_v_mean_val.z);

                if (sum_v_scale_val.x != 0.0f) gpuAtomicAdd((scalar_t*)v_scales + g_global_idx * 3 + 0, sum_v_scale_val.x);
                if (sum_v_scale_val.y != 0.0f) gpuAtomicAdd((scalar_t*)v_scales + g_global_idx * 3 + 1, sum_v_scale_val.y);
                if (sum_v_scale_val.z != 0.0f) gpuAtomicAdd((scalar_t*)v_scales + g_global_idx * 3 + 2, sum_v_scale_val.z);

                if (sum_v_quat_val.x != 0.0f) gpuAtomicAdd((scalar_t*)v_quats + g_global_idx * 4 + 0, sum_v_quat_val.x);
                if (sum_v_quat_val.y != 0.0f) gpuAtomicAdd((scalar_t*)v_quats + g_global_idx * 4 + 1, sum_v_quat_val.y);
                if (sum_v_quat_val.z != 0.0f) gpuAtomicAdd((scalar_t*)v_quats + g_global_idx * 4 + 2, sum_v_quat_val.z);
                if (sum_v_quat_val.w != 0.0f) gpuAtomicAdd((scalar_t*)v_quats + g_global_idx * 4 + 3, sum_v_quat_val.w);

                for (uint32_t k_atomic = 0; k_atomic < CDIM; ++k_atomic) {
                    if (sum_v_rgb_val[k_atomic] != 0.0f) gpuAtomicAdd((scalar_t *)v_colors + g_global_idx * CDIM + k_atomic, sum_v_rgb_val[k_atomic]);
                }
            }
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means, // [N, 3]
    const at::Tensor quats, // [N, 4]
    const at::Tensor scales, // [N, 3]
    const at::Tensor colors,                    // [C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [C, 3]
    const at::optional<at::Tensor> masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,             // [C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                   // [C, 3, 3]
    const CameraModelType camera_model,
    // uncented transform
    const UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs, // [C, 6] or [C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [C, 2] optional
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
    at::Tensor v_means,      // [N, 3]
    at::Tensor v_quats,      // [N, 4]
    at::Tensor v_scales,     // [N, 3]
    at::Tensor v_colors,                    // [C, N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [C, N] or [nnz]
) {
    bool packed = opacities.dim() == 1;
    assert (packed == false); // only support non-packed for now

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means.size(0); // number of gaussians
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {C, tile_height, tile_width};

    int64_t kernel_block_size_host = static_cast<int64_t>(tile_size) * tile_size;
    int64_t shmem_size =
        kernel_block_size_host * (sizeof(int32_t) + sizeof(vec4) + sizeof(vec3) + sizeof(vec4) + sizeof(float) * CDIM) +
        kernel_block_size_host * sizeof(float); // Added space for s_reduction_buffer


    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec3 *>(means.data_ptr<float>()),
            reinterpret_cast<vec4 *>(quats.data_ptr<float>()),
            reinterpret_cast<vec3 *>(scales.data_ptr<float>()),
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
            // camera model
            viewmats0.data_ptr<float>(),
            viewmats1.has_value() ? viewmats1.value().data_ptr<float>()
                                : nullptr,
            Ks.data_ptr<float>(),
            camera_model,
            // uncented transform
            ut_params,
            rs_type,
            radial_coeffs.has_value() ? radial_coeffs.value().data_ptr<float>()
                                    : nullptr,
            tangential_coeffs.has_value()
                ? tangential_coeffs.value().data_ptr<float>()
                : nullptr,
            thin_prism_coeffs.has_value()
                ? thin_prism_coeffs.value().data_ptr<float>()
                : nullptr,
            // intersections
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            // outputs
            reinterpret_cast<vec3 *>(v_means.data_ptr<float>()),
            reinterpret_cast<vec4 *>(v_quats.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_scales.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>()
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM>( \
        const at::Tensor means,                                                \
        const at::Tensor quats,                                                \
        const at::Tensor scales,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        const uint32_t image_width,                                            \
        const uint32_t image_height,                                           \
        const uint32_t tile_size,                                              \
        const at::Tensor viewmats0,                                            \
        const at::optional<at::Tensor> viewmats1,                              \
        const at::Tensor Ks,                                                   \
        const CameraModelType camera_model,                                    \
        const UnscentedTransformParameters ut_params,                         \
        const ShutterType rs_type,                                             \
        const at::optional<at::Tensor> radial_coeffs,                         \
        const at::optional<at::Tensor> tangential_coeffs,                     \
        const at::optional<at::Tensor> thin_prism_coeffs,                     \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const at::Tensor render_alphas,                                        \
        const at::Tensor last_ids,                                             \
        const at::Tensor v_render_colors,                                      \
        const at::Tensor v_render_alphas,                                      \
        at::Tensor v_means,                                                    \
        at::Tensor v_quats,                                                    \
        at::Tensor v_scales,                                                   \
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
