// kernels/newton_kernels.cu
#include "newton_kernels.cuh" // Now in the same directory
#include "kernels/ssim.cuh"   // For fusedssim, fusedssim_backward C++ functions
#include <cuda_runtime.h> // Includes vector_types.h for ::float3, ::float2
#include <device_launch_parameters.h>
#include <torch/torch.h> // For AT_ASSERTM
#include <cmath> // For fabsf, sqrtf, etc.

// Basic CUDA utilities (normally in a separate header)
#define CUDA_CHECK(status) AT_ASSERTM(status == cudaSuccess, cudaGetErrorString(status))

constexpr int CUDA_NUM_THREADS = 256; // Default number of threads per block
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


// --- CUDA Math Helper Functions ---
// Using standard CUDA vector types (e.g., ::float3, ::float2 from vector_types.h)
// and basic operations.
namespace CudaMath {

// Vector operations using ::float3, ::float2
__device__ __forceinline__ ::float3 add_float3(const ::float3& a, const ::float3& b) {
    return ::make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ ::float3 sub_float3(const ::float3& a, const ::float3& b) {
    return ::make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ ::float3 mul_float3_scalar(const ::float3& v, float s) {
    return ::make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ __forceinline__ ::float3 div_float3_scalar(const ::float3& v, float s) {
    float inv_s = 1.0f / (s + 1e-8f); // Add epsilon for stability
    return ::make_float3(v.x * inv_s, v.y * inv_s, v.z * inv_s);
}

__device__ __forceinline__ float dot_product(const ::float3& a, const ::float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ ::float3 cross_product(const ::float3& a, const ::float3& b) {
    return ::make_float3(a.y * b.z - a.z * b.y,
                         a.z * b.x - a.x * b.z,
                         a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float length_sq_float3(const ::float3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ __forceinline__ float length_float3(const ::float3& v) {
    return sqrtf(length_sq_float3(v));
}

__device__ __forceinline__ ::float3 normalize_vec3(const ::float3& v) {
    float l = length_float3(v);
    return div_float3_scalar(v, l);
}

// Matrix operations (assuming row-major for M)
__device__ __forceinline__ ::float3 mul_mat3_vec3(const float* M, const ::float3& v) {
    ::float3 r;
    r.x = M[0] * v.x + M[1] * v.y + M[2] * v.z;
    r.y = M[3] * v.x + M[4] * v.y + M[5] * v.z;
    r.z = M[6] * v.x + M[7] * v.y + M[8] * v.z;
    return r;
}

__device__ __forceinline__ void mat3_transpose_inplace(float* M) {
    float temp;
    temp = M[1]; M[1] = M[3]; M[3] = temp;
    temp = M[2]; M[2] = M[6]; M[6] = temp;
    temp = M[5]; M[5] = M[7]; M[7] = temp;
}

__device__ __forceinline__ void outer_product_3x3(const ::float3& a, const ::float3& b, float* out_M) {
    out_M[0] = a.x * b.x; out_M[1] = a.x * b.y; out_M[2] = a.x * b.z;
    out_M[3] = a.y * b.x; out_M[4] = a.y * b.y; out_M[5] = a.y * b.z;
    out_M[6] = a.z * b.x; out_M[7] = a.z * b.y; out_M[8] = a.z * b.z;
}

__device__ __forceinline__ void mul_mat4_vec4(const float* PW, const float* p_k_h, float* result) {
    for (int i = 0; i < 4; ++i) {
        result[i] = 0;
        for (int j = 0; j < 4; ++j) {
            result[i] += PW[i * 4 + j] * p_k_h[j];
        }
    }
}

__device__ __forceinline__ void mat_mul_vec(const float* M, const float* v, float* out) {
    for (int i = 0; i < 3; ++i) {
        out[i] = 0;
        for (int j = 0; j < 3; ++j) {
            out[i] += M[i * 3 + j] * v[j];
        }
    }
}

__device__ __forceinline__ void mat_mul_mat(const float* A, const float* B, float* C,
                                         int A_rows, int A_cols_B_rows, int B_cols) {
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols_B_rows; ++k) {
                sum += A[i * A_cols_B_rows + k] * B[k * B_cols + j];
            }
            C[i * B_cols + j] = sum;
        }
    }
}

} // namespace CudaMath


// --- Projection Derivative Helper Functions ---
namespace ProjectionDerivs {

__device__ __forceinline__ void compute_h_vec(const ::float3& p_k, const float* PW, float* h_vec4) { // Using ::float3
    float p_k_h[4] = {p_k.x, p_k.y, p_k.z, 1.0f};
    CudaMath::mul_mat4_vec4(PW, p_k_h, h_vec4);
}

__device__ __forceinline__ void compute_projection_jacobian(
    const float* PW, float W_I_t, float H_I_t,
    const float* h_vec4, float* jacobian_out_2x3
) {
    float hx = h_vec4[0];
    float hy = h_vec4[1];
    float hw = h_vec4[3];
    float inv_hw = 1.0f / (hw + 1e-8f);

    float term_x_coeff = W_I_t / 2.0f;
    float term_y_coeff = H_I_t / 2.0f;

    for (int j = 0; j < 3; ++j) {
        jacobian_out_2x3[0 * 3 + j] = term_x_coeff * (inv_hw * PW[0 * 4 + j] - (hx * inv_hw * inv_hw) * PW[3 * 4 + j]);
    }
    for (int j = 0; j < 3; ++j) {
        jacobian_out_2x3[1 * 3 + j] = term_y_coeff * (inv_hw * PW[1 * 4 + j] - (hy * inv_hw * inv_hw) * PW[3 * 4 + j]);
    }
}

__device__ __forceinline__ void compute_projection_hessian(
    const float* PW, float W_I_t, float H_I_t,
    const float* h_vec4,
    float* hessian_out_pi_x_3x3, float* hessian_out_pi_y_3x3
) {
    float hx = h_vec4[0];
    float hy = h_vec4[1];
    float hw = h_vec4[3];
    float inv_hw_sq = 1.0f / (hw * hw + 1e-8f);

    ::float3 PW0_vec = ::make_float3(PW[0], PW[1], PW[2]);
    ::float3 PW1_vec = ::make_float3(PW[4], PW[5], PW[6]);
    ::float3 PW3_vec = ::make_float3(PW[12], PW[13], PW[14]);

    float PW3_outer_PW3[9];
    CudaMath::outer_product_3x3(PW3_vec, PW3_vec, PW3_outer_PW3);
    float PW3_outer_PW0[9];
    CudaMath::outer_product_3x3(PW3_vec, PW0_vec, PW3_outer_PW0);
    float PW3_outer_PW1[9];
    CudaMath::outer_product_3x3(PW3_vec, PW1_vec, PW3_outer_PW1);

    float factor_x1 = W_I_t * (2.0f * hx / (hw*hw*hw + 1e-9f));
    float factor_x2 = W_I_t * (-1.0f * inv_hw_sq);
    for (int i = 0; i < 9; ++i) {
        float term1_x = factor_x1 * PW3_outer_PW3[i];
        int row = i / 3;
        int col = i % 3;
        float term2_x = factor_x2 * (PW3_outer_PW0[i] + PW3_outer_PW0[col * 3 + row]);
        hessian_out_pi_x_3x3[i] = term1_x + term2_x;
    }

    float factor_y1 = H_I_t * (2.0f * hy / (hw*hw*hw + 1e-9f));
    float factor_y2 = H_I_t * (-1.0f * inv_hw_sq);
    for (int i = 0; i < 9; ++i) {
        float term1_y = factor_y1 * PW3_outer_PW3[i];
        int row = i / 3;
        int col = i % 3;
        float term2_y = factor_y2 * (PW3_outer_PW1[i] + PW3_outer_PW1[col * 3 + row]);
        hessian_out_pi_y_3x3[i] = term1_y + term2_y;
    }
}
} // namespace ProjectionDerivs

// --- SH Basis and Color Derivative Helper Functions ---
namespace SHDerivs {

__device__ __forceinline__ void eval_sh_basis_up_to_degree3(
    int degree, const ::float3& r_k_normalized, float* basis_out // Using ::float3
) {
    float x = r_k_normalized.x;
    float y = r_k_normalized.y;
    float z = r_k_normalized.z;
    basis_out[0] = 0.2820947917738781f;
    if (degree == 0) return;
    basis_out[1] = -0.48860251190292f * y;
    basis_out[2] =  0.48860251190292f * z;
    basis_out[3] = -0.48860251190292f * x;
    if (degree == 1) return;
    float x2 = x*x; float y2 = y*y; float z2 = z*z;
    basis_out[4] =  0.5462742152960395f * (2.f * x * y);
    basis_out[5] = -1.092548430592079f * y * z;
    basis_out[6] =  0.3153915652525201f * (3.f * z2 - 1.f);
    basis_out[7] = -1.092548430592079f * x * z;
    basis_out[8] =  0.5462742152960395f * (x2 - y2);
    if (degree == 2) return;
    float fC1 = x2 - y2; float fS1 = 2.f * x * y;
    float fC2 = x * fC1 - y * fS1; float fS2 = y * fC1 + x * fS1;
    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    basis_out[9]  = -0.5900435899266435f * fS2;
    basis_out[10] = fTmp1B * fS1;
    basis_out[11] = fTmp0C * y;
    basis_out[12] = z * (1.865881662950577f * z2 - 1.119528997770346f);
    basis_out[13] = fTmp0C * x;
    basis_out[14] = fTmp1B * fC1;
    basis_out[15] = -0.5900435899266435f * fC2;
}

__device__ __forceinline__ void compute_drk_dpk(
    const ::float3& r_k_normalized, float r_k_norm, float* drk_dpk_out_3x3 // Using ::float3
) {
    float inv_r_k_norm = 1.0f / (r_k_norm + 1e-8f);
    drk_dpk_out_3x3[0] = 1.0f; drk_dpk_out_3x3[1] = 0.0f; drk_dpk_out_3x3[2] = 0.0f;
    drk_dpk_out_3x3[3] = 0.0f; drk_dpk_out_3x3[4] = 1.0f; drk_dpk_out_3x3[5] = 0.0f;
    drk_dpk_out_3x3[6] = 0.0f; drk_dpk_out_3x3[7] = 0.0f; drk_dpk_out_3x3[8] = 1.0f;
    drk_dpk_out_3x3[0] -= r_k_normalized.x * r_k_normalized.x;
    drk_dpk_out_3x3[1] -= r_k_normalized.x * r_k_normalized.y;
    drk_dpk_out_3x3[2] -= r_k_normalized.x * r_k_normalized.z;
    drk_dpk_out_3x3[3] -= r_k_normalized.y * r_k_normalized.x;
    drk_dpk_out_3x3[4] -= r_k_normalized.y * r_k_normalized.y;
    drk_dpk_out_3x3[5] -= r_k_normalized.y * r_k_normalized.z;
    drk_dpk_out_3x3[6] -= r_k_normalized.z * r_k_normalized.x;
    drk_dpk_out_3x3[7] -= r_k_normalized.z * r_k_normalized.y;
    drk_dpk_out_3x3[8] -= r_k_normalized.z * r_k_normalized.z;
    for (int i = 0; i < 9; ++i) {
        drk_dpk_out_3x3[i] *= inv_r_k_norm;
    }
}

__device__ __forceinline__ void compute_dphi_drk_up_to_degree3(
    int degree, const ::float3& r_k_normalized, float* dPhi_drk_out // Using ::float3
) {
    float x = r_k_normalized.x; float y = r_k_normalized.y; float z = r_k_normalized.z;
    float x2 = x*x; float y2 = y*y; float z2 = z*z;
    dPhi_drk_out[0*3 + 0] = 0.0f; dPhi_drk_out[0*3 + 1] = 0.0f; dPhi_drk_out[0*3 + 2] = 0.0f;
    if (degree == 0) return;
    dPhi_drk_out[1*3 + 0] = 0.0f; dPhi_drk_out[1*3 + 1] = -0.48860251190292f; dPhi_drk_out[1*3 + 2] = 0.0f;
    dPhi_drk_out[2*3 + 0] = 0.0f; dPhi_drk_out[2*3 + 1] = 0.0f; dPhi_drk_out[2*3 + 2] = 0.48860251190292f;
    dPhi_drk_out[3*3 + 0] = -0.48860251190292f; dPhi_drk_out[3*3 + 1] = 0.0f; dPhi_drk_out[3*3 + 2] = 0.0f;
    if (degree == 1) return;
    const float C2_0_val = 1.092548430592079f;
    const float C2_1_val = -1.092548430592079f;
    const float C2_2_val_scaled = 0.9461746957575601f;
    dPhi_drk_out[4*3 + 0] = (C2_0_val/2.f) * y; dPhi_drk_out[4*3 + 1] = (C2_0_val/2.f) * x; dPhi_drk_out[4*3 + 2] = 0.0f;
    dPhi_drk_out[5*3 + 0] = 0.0f; dPhi_drk_out[5*3 + 1] = C2_1_val * z; dPhi_drk_out[5*3 + 2] = C2_1_val * y;
    dPhi_drk_out[6*3 + 0] = 0.0f; dPhi_drk_out[6*3 + 1] = 0.0f; dPhi_drk_out[6*3 + 2] = (C2_2_val_scaled/3.f) * (6.f * z);
    dPhi_drk_out[7*3 + 0] = C2_1_val * z; dPhi_drk_out[7*3 + 1] = 0.0f; dPhi_drk_out[7*3 + 2] = C2_1_val * x;
    dPhi_drk_out[8*3 + 0] = (C2_0_val/2.f) * (2.f * x); dPhi_drk_out[8*3 + 1] = (C2_0_val/2.f) * (-2.f * y); dPhi_drk_out[8*3 + 2] = 0.0f;
    if (degree == 2) return;
    const float K9_val = -0.5900435899266435f;
    const float K10_z_coeff_val = 1.445305721320277f;
    const float K11_a_coeff_val = -2.285228997322329f;
    const float K11_b_coeff_val = 0.4570457994644658f;
    const float K12_a_coeff_val = 1.865881662950577f;
    const float K12_b_coeff_val = -1.119528997770346f;
    const float K15_val = -0.5900435899266435f;
    dPhi_drk_out[9*3 + 0] = K9_val * (6.f*x*y);
    dPhi_drk_out[9*3 + 1] = K9_val * (3.f*x2 - 3.f*y2);
    dPhi_drk_out[9*3 + 2] = 0.0f;
    dPhi_drk_out[10*3 + 0] = K10_z_coeff_val * z * (2.f*y);
    dPhi_drk_out[10*3 + 1] = K10_z_coeff_val * z * (2.f*x);
    dPhi_drk_out[10*3 + 2] = K10_z_coeff_val * (2.f*x*y);
    dPhi_drk_out[11*3 + 0] = 0.0f;
    dPhi_drk_out[11*3 + 1] = K11_a_coeff_val * z2 + K11_b_coeff_val;
    dPhi_drk_out[11*3 + 2] = K11_a_coeff_val * y * (2.f*z);
    dPhi_drk_out[12*3 + 0] = 0.0f;
    dPhi_drk_out[12*3 + 1] = 0.0f;
    dPhi_drk_out[12*3 + 2] = K12_a_coeff_val * 3.f*z2 + K12_b_coeff_val;
    dPhi_drk_out[13*3 + 0] = K11_a_coeff_val * z2 + K11_b_coeff_val;
    dPhi_drk_out[13*3 + 1] = 0.0f;
    dPhi_drk_out[13*3 + 2] = K11_a_coeff_val * x * (2.f*z);
    dPhi_drk_out[14*3 + 0] = K10_z_coeff_val * z * (2.f*x);
    dPhi_drk_out[14*3 + 1] = K10_z_coeff_val * z * (-2.f*y);
    dPhi_drk_out[14*3 + 2] = K10_z_coeff_val * (x2 - y2);
    dPhi_drk_out[15*3 + 0] = K15_val * (3.f*x2 - 3.f*y2);
    dPhi_drk_out[15*3 + 1] = K15_val * (-6.f*x*y);
    dPhi_drk_out[15*3 + 2] = 0.0f;
}

__device__ __forceinline__ void compute_sh_color_jacobian_single_channel(
    const float* sh_coeffs_single_channel, const float* sh_basis_values,
    const float* dPhi_drk, const float* drk_dpk,
    int num_basis_coeffs, float* jac_out_3
) {
    float M_prod[16*3];
    CudaMath::mat_mul_mat(dPhi_drk, drk_dpk, M_prod, num_basis_coeffs, 3, 3);
    jac_out_3[0] = 0.0f; jac_out_3[1] = 0.0f; jac_out_3[2] = 0.0f;
    for (int i = 0; i < num_basis_coeffs; ++i) {
        float v_i = sh_basis_values[i] * sh_coeffs_single_channel[i];
        jac_out_3[0] += v_i * M_prod[i * 3 + 0];
        jac_out_3[1] += v_i * M_prod[i * 3 + 1];
        jac_out_3[2] += v_i * M_prod[i * 3 + 2];
    }
}
} // namespace SHDerivs

// --- KERNEL DEFINITIONS ---

__device__ __forceinline__ void get_projected_cov2d_and_derivs_placeholder(
    const ::float3& p_k_world, // Using ::float3
    const float* scales_k, const float* rotations_k,
    const float* view_matrix, const float* proj_matrix,
    const float* jacobian_d_pi_d_pk,
    float img_W, float img_H,
    float* cov2d_sym, float* inv_cov2d_sym, float* det_cov2d,
    float* d_Gk_d_pik, float* d2_Gk_d_pik2
) {
    cov2d_sym[0] = 1.0f; cov2d_sym[1] = 0.0f; cov2d_sym[2] = 1.0f;
    inv_cov2d_sym[0] = 1.0f; inv_cov2d_sym[1] = 0.0f; inv_cov2d_sym[2] = 1.0f;
    *det_cov2d = 1.0f;
    if (d_Gk_d_pik) {
        d_Gk_d_pik[0] = 0.0f; d_Gk_d_pik[1] = 0.0f;
    }
    if (d2_Gk_d_pik2) {
        d2_Gk_d_pik2[0] = -1.0f * inv_cov2d_sym[0];
        d2_Gk_d_pik2[1] = -1.0f * inv_cov2d_sym[1];
        d2_Gk_d_pik2[2] = -1.0f * inv_cov2d_sym[2];
    }
}

__global__ void compute_l1l2_loss_derivatives_kernel(
    const float* rendered_image, const float* gt_image, bool use_l2_loss_term,
    float inv_N_pixels, float* out_dL_dc_l1l2, float* out_d2L_dc2_diag_l1l2,
    int H, int W, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = H * W * C;
    if (idx >= total_elements) return;
    float diff = rendered_image[idx] - gt_image[idx];
    if (use_l2_loss_term) {
        out_dL_dc_l1l2[idx] = inv_N_pixels * 2.f * diff;
        out_d2L_dc2_diag_l1l2[idx] = inv_N_pixels * 2.f;
    } else {
        out_dL_dc_l1l2[idx] = inv_N_pixels * ((diff > 1e-6f) ? 1.f : ((diff < -1e-6f) ? -1.f : 0.f));
        out_d2L_dc2_diag_l1l2[idx] = 0.f;
    }
}

__global__ void compute_position_hessian_components_kernel(
    int H_img, int W_img, int C_img,
    int P_total,
    const float* means_3d_all,
    const float* scales_all,
    const float* rotations_all,
    const float* opacities_all,
    const float* shs_all,
    int sh_degree,
    int sh_coeffs_per_color_channel,
    const float* view_matrix,
    const float* projection_matrix_for_jacobian,
    const float* cam_pos_world,
    const bool* visibility_mask_for_model,
    const float* dL_dc_pixelwise,
    const float* d2L_dc2_diag_pixelwise,
    int num_output_gaussians,
    float* H_p_output_packed,
    float* grad_p_output,
    const int* output_index_map,
    bool debug_prints_enabled
) {
    int p_idx_total = blockIdx.x * blockDim.x + threadIdx.x;

    if (p_idx_total >= P_total) return;
    if (!visibility_mask_for_model[p_idx_total]) return;

    int output_idx = output_index_map[p_idx_total];
    if (output_idx < 0 || output_idx >= num_output_gaussians) return;

    ::float3 pk_vec3 = ::make_float3(
        means_3d_all[p_idx_total * 3 + 0],
        means_3d_all[p_idx_total * 3 + 1],
        means_3d_all[p_idx_total * 3 + 2]);

    const float* scales_k = scales_all + p_idx_total * 3;
    const float* rotations_k = rotations_all + p_idx_total * 4;
    float opacity_k = opacities_all[p_idx_total];
    const float* sh_coeffs_k_all_channels = shs_all + p_idx_total * sh_coeffs_per_color_channel * 3;

    ::float3 cam_pos_world_vec3 = ::make_float3(cam_pos_world[0], cam_pos_world[1], cam_pos_world[2]);

    ::float3 view_dir_to_pk_unnormalized = CudaMath::sub_float3(pk_vec3, cam_pos_world_vec3);
    float r_k_norm = CudaMath::length_float3(view_dir_to_pk_unnormalized);
    ::float3 r_k_normalized = CudaMath::div_float3_scalar(view_dir_to_pk_unnormalized, r_k_norm);

    float proj_view_matrix[16];
    CudaMath::mat_mul_mat(projection_matrix_for_jacobian, view_matrix, proj_view_matrix, 4, 4, 4);

    float h_vec4_data[4];
    ProjectionDerivs::compute_h_vec(pk_vec3, proj_view_matrix, h_vec4_data);

    float d_pi_d_pk_data[2*3];
    ProjectionDerivs::compute_projection_jacobian(proj_view_matrix, (float)W_img, (float)H_img, h_vec4_data, d_pi_d_pk_data);

    float d2_pi_d_pk2_x_data[3*3];
    float d2_pi_d_pk2_y_data[3*3];
    ProjectionDerivs::compute_projection_hessian(proj_view_matrix, (float)W_img, (float)H_img, h_vec4_data, d2_pi_d_pk2_x_data, d2_pi_d_pk2_y_data);

    float sh_basis_eval_data[16];
    SHDerivs::eval_sh_basis_up_to_degree3(sh_degree, r_k_normalized, sh_basis_eval_data);

    float d_rk_d_pk_data[3*3];
    SHDerivs::compute_drk_dpk(r_k_normalized, r_k_norm, d_rk_d_pk_data);

    float d_phi_d_rk_data[16*3];
    SHDerivs::compute_dphi_drk_up_to_degree3(sh_degree, r_k_normalized, d_phi_d_rk_data);

    ::float3 d_c_bar_R_d_pk_val, d_c_bar_G_d_pk_val, d_c_bar_B_d_pk_val;
    float sh_coeffs_k_R[16], sh_coeffs_k_G[16], sh_coeffs_k_B[16];
    for(int i=0; i<sh_coeffs_per_color_channel; ++i) {
        sh_coeffs_k_R[i] = sh_coeffs_k_all_channels[i*3 + 0];
        sh_coeffs_k_G[i] = sh_coeffs_k_all_channels[i*3 + 1];
        sh_coeffs_k_B[i] = sh_coeffs_k_all_channels[i*3 + 2];
    }

    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_R, sh_basis_eval_data, d_phi_d_rk_data, d_rk_d_pk_data, sh_coeffs_per_color_channel, &d_c_bar_R_d_pk_val.x);
    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_G, sh_basis_eval_data, d_phi_d_rk_data, d_rk_d_pk_data, sh_coeffs_per_color_channel, &d_c_bar_G_d_pk_val.x);
    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_B, sh_basis_eval_data, d_phi_d_rk_data, d_rk_d_pk_data, sh_coeffs_per_color_channel, &d_c_bar_B_d_pk_val.x);

    ::float3 g_p_k_accum_val = ::make_float3(0.f, 0.f, 0.f);
    float H_p_k_accum_symm[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int r = 0; r < H_img; ++r) {
        for (int c = 0; c < W_img; ++c) {
            float pixel_ndc_x = (2.0f * (c + 0.5f) / W_img - 1.0f);
            float pixel_ndc_y = (2.0f * (r + 0.5f) / H_img - 1.0f);

            float pi_k_ndc_x_unscaled = h_vec4_data[0] / (h_vec4_data[3] + 1e-7f);
            float pi_k_ndc_y_unscaled = h_vec4_data[1] / (h_vec4_data[3] + 1e-7f);

            ::float2 diff_ndc_val = ::make_float2(pi_k_ndc_x_unscaled - pixel_ndc_x, pi_k_ndc_y_unscaled - pixel_ndc_y);

            float cov2d_sym_data[3], inv_cov2d_sym_data[3], det_cov2d_data;
            float d_Gk_d_pik_data[2];
            float d2_Gk_d_pik2_data[3];

            get_projected_cov2d_and_derivs_placeholder(pk_vec3, scales_k, rotations_k,
                                                       view_matrix, projection_matrix_for_jacobian,
                                                       d_pi_d_pk_data, (float)W_img, (float)H_img,
                                                       cov2d_sym_data, inv_cov2d_sym_data, &det_cov2d_data,
                                                       nullptr, nullptr);

            float G_k_pixel = expf(-0.5f * (diff_ndc_val.x*diff_ndc_val.x*inv_cov2d_sym_data[0] +
                                            2*diff_ndc_val.x*diff_ndc_val.y*inv_cov2d_sym_data[1] +
                                            diff_ndc_val.y*diff_ndc_val.y*inv_cov2d_sym_data[2]));
            if (det_cov2d_data <= 1e-7f) G_k_pixel = 0.f;

            if (G_k_pixel < 1e-4f) continue;

            ::float2 sigma_inv_diff_val;
            sigma_inv_diff_val.x = inv_cov2d_sym_data[0]*diff_ndc_val.x + inv_cov2d_sym_data[1]*diff_ndc_val.y;
            sigma_inv_diff_val.y = inv_cov2d_sym_data[1]*diff_ndc_val.x + inv_cov2d_sym_data[2]*diff_ndc_val.y;
            d_Gk_d_pik_data[0] = -G_k_pixel * sigma_inv_diff_val.x;
            d_Gk_d_pik_data[1] = -G_k_pixel * sigma_inv_diff_val.y;

            d2_Gk_d_pik2_data[0] = G_k_pixel * (sigma_inv_diff_val.x * sigma_inv_diff_val.x - inv_cov2d_sym_data[0]);
            d2_Gk_d_pik2_data[1] = G_k_pixel * (sigma_inv_diff_val.x * sigma_inv_diff_val.y - inv_cov2d_sym_data[1]);
            d2_Gk_d_pik2_data[2] = G_k_pixel * (sigma_inv_diff_val.y * sigma_inv_diff_val.y - inv_cov2d_sym_data[2]);


            float alpha_k_pixel = opacity_k * G_k_pixel;

            ::float3 c_bar_k_rgb_val;
            c_bar_k_rgb_val.x =0; for(int i=0; i<sh_coeffs_per_color_channel; ++i) c_bar_k_rgb_val.x += sh_coeffs_k_R[i] * sh_basis_eval_data[i];
            c_bar_k_rgb_val.y =0; for(int i=0; i<sh_coeffs_per_color_channel; ++i) c_bar_k_rgb_val.y += sh_coeffs_k_G[i] * sh_basis_eval_data[i];
            c_bar_k_rgb_val.z =0; for(int i=0; i<sh_coeffs_per_color_channel; ++i) c_bar_k_rgb_val.z += sh_coeffs_k_B[i] * sh_basis_eval_data[i];


            ::float3 d_c_final_d_Gk_val = CudaMath::mul_float3_scalar(c_bar_k_rgb_val, opacity_k);

            ::float3 d_Gk_d_pk_chain_val;
            d_Gk_d_pk_chain_val.x = d_Gk_d_pik_data[0] * d_pi_d_pk_data[0*3+0] + d_Gk_d_pik_data[1] * d_pi_d_pk_data[1*3+0];
            d_Gk_d_pk_chain_val.y = d_Gk_d_pik_data[0] * d_pi_d_pk_data[0*3+1] + d_Gk_d_pik_data[1] * d_pi_d_pk_data[1*3+1];
            d_Gk_d_pk_chain_val.z = d_Gk_d_pik_data[0] * d_pi_d_pk_data[0*3+2] + d_Gk_d_pik_data[1] * d_pi_d_pk_data[1*3+2];

            ::float3 J_c_pk_R_val, J_c_pk_G_val, J_c_pk_B_val;
            J_c_pk_R_val = CudaMath::add_float3(CudaMath::mul_float3_scalar(d_c_bar_R_d_pk_val, alpha_k_pixel), CudaMath::mul_float3_scalar(d_Gk_d_pk_chain_val, d_c_final_d_Gk_val.x));
            J_c_pk_G_val = CudaMath::add_float3(CudaMath::mul_float3_scalar(d_c_bar_G_d_pk_val, alpha_k_pixel), CudaMath::mul_float3_scalar(d_Gk_d_pk_chain_val, d_c_final_d_Gk_val.y));
            J_c_pk_B_val = CudaMath::add_float3(CudaMath::mul_float3_scalar(d_c_bar_B_d_pk_val, alpha_k_pixel), CudaMath::mul_float3_scalar(d_Gk_d_pk_chain_val, d_c_final_d_Gk_val.z));

            int pixel_idx_flat = (r * W_img + c) * C_img;
            ::float3 dL_dc_val_pixel = ::make_float3(dL_dc_pixelwise[pixel_idx_flat+0], dL_dc_pixelwise[pixel_idx_flat+1], dL_dc_pixelwise[pixel_idx_flat+2]);
            ::float3 d2L_dc2_diag_val_pixel = ::make_float3(d2L_dc2_diag_pixelwise[pixel_idx_flat+0], d2L_dc2_diag_pixelwise[pixel_idx_flat+1], d2L_dc2_diag_pixelwise[pixel_idx_flat+2]);

            g_p_k_accum_val.x += J_c_pk_R_val.x * dL_dc_val_pixel.x + J_c_pk_G_val.x * dL_dc_val_pixel.y + J_c_pk_B_val.x * dL_dc_val_pixel.z;
            g_p_k_accum_val.y += J_c_pk_R_val.y * dL_dc_val_pixel.x + J_c_pk_G_val.y * dL_dc_val_pixel.y + J_c_pk_B_val.y * dL_dc_val_pixel.z;
            g_p_k_accum_val.z += J_c_pk_R_val.z * dL_dc_val_pixel.x + J_c_pk_G_val.z * dL_dc_val_pixel.y + J_c_pk_B_val.z * dL_dc_val_pixel.z;

            H_p_k_accum_symm[0] += J_c_pk_R_val.x * d2L_dc2_diag_val_pixel.x * J_c_pk_R_val.x + J_c_pk_G_val.x * d2L_dc2_diag_val_pixel.y * J_c_pk_G_val.x + J_c_pk_B_val.x * d2L_dc2_diag_val_pixel.z * J_c_pk_B_val.x;
            H_p_k_accum_symm[1] += J_c_pk_R_val.x * d2L_dc2_diag_val_pixel.x * J_c_pk_R_val.y + J_c_pk_G_val.x * d2L_dc2_diag_val_pixel.y * J_c_pk_G_val.y + J_c_pk_B_val.x * d2L_dc2_diag_val_pixel.z * J_c_pk_B_val.y;
            H_p_k_accum_symm[2] += J_c_pk_R_val.x * d2L_dc2_diag_val_pixel.x * J_c_pk_R_val.z + J_c_pk_G_val.x * d2L_dc2_diag_val_pixel.y * J_c_pk_G_val.z + J_c_pk_B_val.x * d2L_dc2_diag_val_pixel.z * J_c_pk_B_val.z;
            H_p_k_accum_symm[3] += J_c_pk_R_val.y * d2L_dc2_diag_val_pixel.x * J_c_pk_R_val.y + J_c_pk_G_val.y * d2L_dc2_diag_val_pixel.y * J_c_pk_G_val.y + J_c_pk_B_val.y * d2L_dc2_diag_val_pixel.z * J_c_pk_B_val.y;
            H_p_k_accum_symm[4] += J_c_pk_R_val.y * d2L_dc2_diag_val_pixel.x * J_c_pk_R_val.z + J_c_pk_G_val.y * d2L_dc2_diag_val_pixel.y * J_c_pk_G_val.z + J_c_pk_B_val.y * d2L_dc2_diag_val_pixel.z * J_c_pk_B_val.z;
            H_p_k_accum_symm[5] += J_c_pk_R_val.z * d2L_dc2_diag_val_pixel.x * J_c_pk_R_val.z + J_c_pk_G_val.z * d2L_dc2_diag_val_pixel.y * J_c_pk_G_val.z + J_c_pk_B_val.z * d2L_dc2_diag_val_pixel.z * J_c_pk_B_val.z;
        }
    }

    grad_p_output[output_idx * 3 + 0] = g_p_k_accum_val.x;
    grad_p_output[output_idx * 3 + 1] = g_p_k_accum_val.y;
    grad_p_output[output_idx * 3 + 2] = g_p_k_accum_val.z;

    for(int i=0; i<6; ++i) {
        H_p_output_packed[output_idx * 6 + i] = H_p_k_accum_symm[i];
    }
}

// Kernel for projecting Hessian and Gradient
__global__ void project_position_hessian_gradient_kernel(
    int num_visible_gaussians,
    const float* H_p_packed_input, // [N_vis, 6] (Hpxx, Hpxy, Hpxz, Hpyy, Hpyz, Hpzz)
    const float* grad_p_input,     // [N_vis, 3]
    const float* means_3d_visible, // [N_vis, 3]
    const float* view_matrix,      // [16] (col-major or row-major assumed by caller)
    const float* cam_pos_world,    // [3]
    float* out_H_v_packed,         // [N_vis, 3] (Hvxx, Hvxy, Hvyy)
    float* out_grad_v) {           // [N_vis, 2]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_visible_gaussians) return;

    float ux[3] = {view_matrix[0], view_matrix[4], view_matrix[8]};
    float uy[3] = {view_matrix[1], view_matrix[5], view_matrix[9]};

    out_grad_v[idx*2 + 0] = ux[0]*grad_p_input[idx*3+0] + ux[1]*grad_p_input[idx*3+1] + ux[2]*grad_p_input[idx*3+2];
    out_grad_v[idx*2 + 1] = uy[0]*grad_p_input[idx*3+0] + uy[1]*grad_p_input[idx*3+1] + uy[2]*grad_p_input[idx*3+2];

    const float* Hp = &H_p_packed_input[idx*6];
    float Hpu_x[3];
    Hpu_x[0] = Hp[0]*ux[0] + Hp[1]*ux[1] + Hp[2]*ux[2];
    Hpu_x[1] = Hp[1]*ux[0] + Hp[3]*ux[1] + Hp[4]*ux[2];
    Hpu_x[2] = Hp[2]*ux[0] + Hp[4]*ux[1] + Hp[5]*ux[2];

    float Hpu_y[3];
    Hpu_y[0] = Hp[0]*uy[0] + Hp[1]*uy[1] + Hp[2]*uy[2];
    Hpu_y[1] = Hp[1]*uy[0] + Hp[3]*uy[1] + Hp[4]*uy[2];
    Hpu_y[2] = Hp[2]*uy[0] + Hp[4]*uy[1] + Hp[5]*uy[2];

    out_H_v_packed[idx*3 + 0] = ux[0]*Hpu_x[0] + ux[1]*Hpu_x[1] + ux[2]*Hpu_x[2];
    out_H_v_packed[idx*3 + 1] = ux[0]*Hpu_y[0] + ux[1]*Hpu_y[1] + ux[2]*Hpu_y[2];
    out_H_v_packed[idx*3 + 2] = uy[0]*Hpu_y[0] + uy[1]*Hpu_y[1] + uy[2]*Hpu_y[2];
}

// Kernel for batch 2x2 solve
__global__ void batch_solve_2x2_system_kernel(
    int num_systems,
    const float* H_v_packed,
    const float* g_v,
    float damping,
    float step_scale,
    float* out_delta_v) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_systems) return;

    float H00 = H_v_packed[idx*3 + 0];
    float H01 = H_v_packed[idx*3 + 1];
    float H11 = H_v_packed[idx*3 + 2];

    float g0 = g_v[idx*2 + 0];
    float g1 = g_v[idx*2 + 1];

    H00 += damping;
    H11 += damping;

    float det = H00 * H11 - H01 * H01;

    if (abs(det) < 1e-8f) {
        out_delta_v[idx*2 + 0] = -step_scale * g0 / (H00 + 1e-6f);
        out_delta_v[idx*2 + 1] = -step_scale * g1 / (H11 + 1e-6f);
        return;
    }
    float inv_det = 1.f / det;
    out_delta_v[idx*2 + 0] = -step_scale * inv_det * (H11 * g0 - H01 * g1);
    out_delta_v[idx*2 + 1] = -step_scale * inv_det * (-H01 * g0 + H00 * g1);
}

// Kernel for re-projecting delta_v to delta_p
__global__ void project_update_to_3d_kernel(
    int num_updates,
    const float* delta_v,
    const float* means_3d_visible,
    const float* view_matrix,
    const float* cam_pos_world,
    float* out_delta_p) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_updates) return;

    float ux[3] = {view_matrix[0], view_matrix[4], view_matrix[8]};
    float uy[3] = {view_matrix[1], view_matrix[5], view_matrix[9]};

    float dvx = delta_v[idx*2 + 0];
    float dvy = delta_v[idx*2 + 1];

    out_delta_p[idx*3 + 0] = ux[0] * dvx + uy[0] * dvy;
    out_delta_p[idx*3 + 1] = ux[1] * dvx + uy[1] * dvy;
    out_delta_p[idx*3 + 2] = ux[2] * dvx + uy[2] * dvy;
}

// --- Spherical Harmonics Basis Evaluation Kernel ---
__global__ void eval_sh_basis_kernel(
    const int num_points, const int degree,
    const float* dirs, float* sh_basis_output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    float x = dirs[idx * 3 + 0]; float y = dirs[idx * 3 + 1]; float z = dirs[idx * 3 + 2];
    int num_sh_coeffs = (degree + 1) * (degree + 1);
    float* current_sh_output = sh_basis_output + idx * num_sh_coeffs;
    current_sh_output[0] = 0.2820947917738781f;
    if (degree == 0) return;
    current_sh_output[1] = -0.48860251190292f * y;
    current_sh_output[2] = 0.48860251190292f * z;
    current_sh_output[3] = -0.48860251190292f * x;
    if (degree == 1) return;
    float z2 = z * z; float fTmp0B = -1.092548430592079f * z;
    float fC1 = x * x - y * y; float fS1 = 2.f * x * y;
    current_sh_output[4] = 0.5462742152960395f * fS1;
    current_sh_output[5] = fTmp0B * y;
    current_sh_output[6] = (0.9461746957575601f * z2 - 0.3153915652525201f);
    current_sh_output[7] = fTmp0B * x;
    current_sh_output[8] = 0.5462742152960395f * fC1;
    if (degree == 2) return;
    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    float fC2 = x * fC1 - y * fS1; float fS2 = x * fS1 + y * fC1;
    current_sh_output[9]  = -0.5900435899266435f * fS2;
    current_sh_output[10] = fTmp1B * fS1;
    current_sh_output[11] = fTmp0C * y;
    current_sh_output[12] = z * (1.865881662950577f * z2 - 1.119528997770346f);
    current_sh_output[13] = fTmp0C * x;
    current_sh_output[14] = fTmp1B * fC1;
    current_sh_output[15] = -0.5900435899266435f * fC2;
    if (degree == 3) return;
    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fC3 = x * fC2 - y * fS2; float fS3 = x * fS2 + y * fC2;
    float pSH6_val = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH12_val = z * (1.865881662950577f * z2 - 1.119528997770346f);
    current_sh_output[16] = 0.6258357354491763f * fS3;
    current_sh_output[17] = fTmp2B * fS2;
    current_sh_output[18] = fTmp1C * fS1;
    current_sh_output[19] = fTmp0D * y;
    current_sh_output[20] = (1.984313483298443f * z * pSH12_val - 1.006230589874905f * pSH6_val);
    current_sh_output[21] = fTmp0D * x;
    current_sh_output[22] = fTmp1C * fC1;
    current_sh_output[23] = fTmp2B * fC2;
    current_sh_output[24] = 0.6258357354491763f * fC3;
}

// --- Kernel for batch 3x3 solve ---
__global__ void batch_solve_3x3_symmetric_system_kernel(
    int num_systems,
    const float* H_packed, const float* g,
    float damping, float* out_x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_systems) return;
    const float* Hp = &H_packed[idx * 6];
    const float* gp = &g[idx * 3];
    float* xp = &out_x[idx * 3];
    float a00 = Hp[0] + damping; float a01 = Hp[1];         float a02 = Hp[2];
    float a10 = Hp[1];         float a11 = Hp[3] + damping; float a12 = Hp[4];
    float a20 = Hp[2];         float a21 = Hp[4];         float a22 = Hp[5] + damping;
    float detA = a00 * (a11 * a22 - a12 * a21) -
                 a01 * (a10 * a22 - a12 * a20) +
                 a02 * (a10 * a21 - a11 * a20);
    if (abs(detA) < 1e-9f) {
        xp[0] = -gp[0] / (a00 + 1e-6f);
        xp[1] = -gp[1] / (a11 + 1e-6f);
        xp[2] = -gp[2] / (a22 + 1e-6f);
        return;
    }
    float invDetA = 1.0f / detA;
    xp[0] = invDetA * ((a11*a22 - a12*a21)*(-gp[0]) + (a02*a21 - a01*a22)*(-gp[1]) + (a01*a12 - a02*a11)*(-gp[2]));
    xp[1] = invDetA * ((a12*a20 - a10*a22)*(-gp[0]) + (a00*a22 - a02*a20)*(-gp[1]) + (a02*a10 - a00*a12)*(-gp[2]));
    xp[2] = invDetA * ((a10*a21 - a11*a20)*(-gp[0]) + (a01*a20 - a00*a21)*(-gp[1]) + (a00*a11 - a01*a10)*(-gp[2]));
}

// --- Kernel for batch 1x1 solve ---
__global__ void batch_solve_1x1_system_kernel(
    int num_systems,
    const float* H_scalar, const float* g_scalar,
    float damping, float* out_x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_systems) return;
    float h_val = H_scalar[idx];
    float g_val = g_scalar[idx];
    float h_damped = h_val + damping;
    if (abs(h_damped) < 1e-9f) {
        out_x[idx] = 0.0f;
    } else {
        out_x[idx] = -g_val / h_damped;
    }
}


// --- LAUNCHER FUNCTIONS ---

void NewtonKernels::compute_loss_derivatives_kernel_launcher(
    const torch::Tensor& rendered_image_tensor,
    const torch::Tensor& gt_image_tensor,
    float lambda_dssim,
    bool use_l2_loss_term,
    torch::Tensor& out_dL_dc_tensor,
    torch::Tensor& out_d2L_dc2_diag_tensor) {

    int H = rendered_image_tensor.size(0);
    int W = rendered_image_tensor.size(1);
    int C = rendered_image_tensor.size(2);
    int total_elements = H * W * C;

    const float* rendered_image_ptr = gs::torch_utils::get_const_data_ptr<float>(rendered_image_tensor);
    const float* gt_image_ptr = gs::torch_utils::get_const_data_ptr<float>(gt_image_tensor);
    float* out_dL_dc_ptr = gs::torch_utils::get_data_ptr<float>(out_dL_dc_tensor);
    float* out_d2L_dc2_diag_ptr = gs::torch_utils::get_data_ptr<float>(out_d2L_dc2_diag_tensor);

    auto tensor_opts = rendered_image_tensor.options();
    torch::Tensor dL_dc_l1l2 = torch::empty_like(rendered_image_tensor, tensor_opts);
    torch::Tensor d2L_dc2_diag_l1l2 = torch::empty_like(rendered_image_tensor, tensor_opts);

    const float N_pixels = static_cast<float>(H * W);
    const float inv_N_pixels = (N_pixels > 0) ? (1.0f / N_pixels) : 1.0f;

    compute_l1l2_loss_derivatives_kernel<<<GET_BLOCKS(total_elements), CUDA_NUM_THREADS>>>(
        rendered_image_ptr, gt_image_ptr, use_l2_loss_term, inv_N_pixels,
        gs::torch_utils::get_data_ptr<float>(dL_dc_l1l2),
        gs::torch_utils::get_data_ptr<float>(d2L_dc2_diag_l1l2),
        H, W, C
    );
    CUDA_CHECK(cudaGetLastError());

    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;

    torch::Tensor img1_bchw = rendered_image_tensor.unsqueeze(0).permute({0, 3, 1, 2}).contiguous();
    torch::Tensor img2_bchw = gt_image_tensor.unsqueeze(0).permute({0, 3, 1, 2}).contiguous();

    auto ssim_outputs = fusedssim(C1, C2, img1_bchw, img2_bchw, true );
    torch::Tensor ssim_map_bchw = std::get<0>(ssim_outputs);
    torch::Tensor dm_dmu1 = std::get<1>(ssim_outputs);
    torch::Tensor dm_dsigma1_sq = std::get<2>(ssim_outputs);
    torch::Tensor dm_dsigma12 = std::get<3>(ssim_outputs);

    torch::Tensor dL_dmap_tensor = torch::full_like(ssim_map_bchw, -0.5f);

    torch::Tensor dL_dc_ssim_bchw = fusedssim_backward(
        C1, C2, img1_bchw, img2_bchw, dL_dmap_tensor, dm_dmu1, dm_dsigma1_sq, dm_dsigma12
    );

    torch::Tensor dL_dc_ssim_hwc_unnormalized = dL_dc_ssim_bchw.permute({0, 2, 3, 1}).squeeze(0).contiguous();
    torch::Tensor dL_dc_ssim_hwc_normalized = dL_dc_ssim_hwc_unnormalized * inv_N_pixels;

    out_dL_dc_tensor.copy_(dL_dc_l1l2 + lambda_dssim * dL_dc_ssim_hwc_normalized);
    out_d2L_dc2_diag_tensor.copy_(d2L_dc2_diag_l1l2);
    CUDA_CHECK(cudaGetLastError());
}


void NewtonKernels::compute_position_hessian_components_kernel_launcher(
    int H_img, int W_img, int C_img,
    int P_total,
    const float* means_3d_all, const float* scales_all, const float* rotations_all,
    const float* opacities_all, const float* shs_all,
    int sh_degree,
    int sh_coeffs_per_color_channel,
    const float* view_matrix_ptr,
    const float* perspective_proj_matrix_ptr,
    const float* cam_pos_world_ptr,
    const torch::Tensor& visibility_mask_for_model_tensor,
    const float* dL_dc_pixelwise_ptr,
    const float* d2L_dc2_diag_pixelwise_ptr,
    int num_output_gaussians,
    float* H_p_output_packed_ptr,
    float* grad_p_output_ptr,
    bool debug_prints_enabled
) {
    TORCH_CHECK(visibility_mask_for_model_tensor.defined(), "visibility_mask_for_model_tensor is not defined in launcher.");
    TORCH_CHECK(visibility_mask_for_model_tensor.scalar_type() == torch::kBool, "visibility_mask_for_model_tensor must be Bool type.");
    TORCH_CHECK(static_cast<int>(visibility_mask_for_model_tensor.size(0)) == P_total, "visibility_mask_for_model_tensor size mismatch with P_total.");

    torch::Tensor visibility_mask_cpu = visibility_mask_for_model_tensor.to(torch::kCPU).contiguous();
    const bool* cpu_visibility_ptr = visibility_mask_cpu.data_ptr<bool>();

    std::vector<int> output_index_map_cpu(P_total);
    int current_out_idx = 0;
    for(int i=0; i<P_total; ++i) {
        if(cpu_visibility_ptr[i]) {
            output_index_map_cpu[i] = current_out_idx++;
        } else {
            output_index_map_cpu[i] = -1;
        }
    }

    torch::Tensor output_index_map_tensor = torch::tensor(output_index_map_cpu,
        torch::TensorOptions().dtype(torch::kInt).device(visibility_mask_for_model_tensor.device()));
    const int* output_index_map_gpu = gs::torch_utils::get_const_data_ptr<int>(output_index_map_tensor, "output_index_map_tensor_in_launcher");
    const bool* visibility_mask_gpu_ptr = gs::torch_utils::get_const_data_ptr<bool>(visibility_mask_for_model_tensor, "visibility_mask_for_model_tensor_for_kernel");

    if (debug_prints_enabled && P_total > 0) {
         printf("[CUDA LAUNCHER] compute_position_hessian_components_kernel. P_total: %d, num_output_gaussians: %d, H/W/C: %d/%d/%d\n",
                P_total, num_output_gaussians, H_img, W_img, C_img);
    }

    compute_position_hessian_components_kernel<<<GET_BLOCKS(P_total), CUDA_NUM_THREADS>>>(
        H_img, W_img, C_img,
        P_total,
        means_3d_all,
        scales_all,
        rotations_all,
        opacities_all,
        shs_all,
        sh_degree,
        sh_coeffs_per_color_channel,
        view_matrix_ptr,
        perspective_proj_matrix_ptr,
        cam_pos_world_ptr,
        visibility_mask_gpu_ptr,
        dL_dc_pixelwise_ptr,
        d2L_dc2_diag_pixelwise_ptr,
        num_output_gaussians,
        H_p_output_packed_ptr,
        grad_p_output_ptr,
        output_index_map_gpu,
        debug_prints_enabled
    );
    CUDA_CHECK(cudaGetLastError());
}

void NewtonKernels::project_position_hessian_gradient_kernel_launcher(
    int num_visible_gaussians,
    const float* H_p_packed_input, const float* grad_p_input,
    const float* means_3d_visible, const float* view_matrix,
    const float* cam_pos_world,
    float* out_H_v_packed, float* out_grad_v ) {

    project_position_hessian_gradient_kernel<<<GET_BLOCKS(num_visible_gaussians), CUDA_NUM_THREADS>>>(
        num_visible_gaussians, H_p_packed_input, grad_p_input, means_3d_visible,
        view_matrix, cam_pos_world, out_H_v_packed, out_grad_v
    );
    CUDA_CHECK(cudaGetLastError());
}

void NewtonKernels::batch_solve_2x2_system_kernel_launcher(
    int num_systems,
    const float* H_v_packed, const float* g_v, float damping, float step_scale,
    float* out_delta_v ) {

    batch_solve_2x2_system_kernel<<<GET_BLOCKS(num_systems), CUDA_NUM_THREADS>>>(
        num_systems, H_v_packed, g_v, damping, step_scale, out_delta_v
    );
    CUDA_CHECK(cudaGetLastError());
}

void NewtonKernels::project_update_to_3d_kernel_launcher(
    int num_updates,
    const float* delta_v, const float* means_3d_visible,
    const float* view_matrix, const float* cam_pos_world,
    float* out_delta_p ) {

    project_update_to_3d_kernel<<<GET_BLOCKS(num_updates), CUDA_NUM_THREADS>>>(
        num_updates, delta_v, means_3d_visible, view_matrix, cam_pos_world, out_delta_p
    );
    CUDA_CHECK(cudaGetLastError());
}

// --- Definitions for Scale Optimization Launchers (Stubs) ---

void NewtonKernels::compute_scale_hessian_gradient_components_kernel_launcher(
    int H_img, int W_img, int C_img,
    int P_total,
    const torch::Tensor& means_all,
    const torch::Tensor& scales_all,
    const torch::Tensor& rotations_all,
    const torch::Tensor& opacities_all,
    const torch::Tensor& shs_all,
    int sh_degree,
    const torch::Tensor& view_matrix,
    const torch::Tensor& K_matrix,
    const torch::Tensor& cam_pos_world,
    const gs::RenderOutput& render_output,
    const torch::Tensor& visible_indices,
    const torch::Tensor& dL_dc_pixelwise,
    const torch::Tensor& d2L_dc2_diag_pixelwise,
    torch::Tensor& out_H_s_packed,
    torch::Tensor& out_g_s
) {
    if (out_H_s_packed.defined()) out_H_s_packed.zero_(); // Stub behavior
    if (out_g_s.defined()) out_g_s.zero_();             // Stub behavior
}

void NewtonKernels::batch_solve_3x3_system_kernel_launcher(
    int num_systems,
    const torch::Tensor& H_s_packed,
    const torch::Tensor& g_s,
    float damping,
    torch::Tensor& out_delta_s
) {
    TORCH_CHECK(H_s_packed.defined() && H_s_packed.dim() == 2 && H_s_packed.size(1) == 6, "H_s_packed shape must be [N, 6]");
    TORCH_CHECK(g_s.defined() && g_s.dim() == 2 && g_s.size(1) == 3, "g_s shape must be [N, 3]");
    TORCH_CHECK(out_delta_s.defined() && out_delta_s.dim() == 2 && out_delta_s.size(1) == 3, "out_delta_s shape must be [N, 3]");
    TORCH_CHECK(H_s_packed.size(0) == num_systems && g_s.size(0) == num_systems && out_delta_s.size(0) == num_systems, "Batch size mismatch");
    TORCH_CHECK(H_s_packed.is_cuda() && g_s.is_cuda() && out_delta_s.is_cuda(), "All tensors must be CUDA tensors");
    TORCH_CHECK(H_s_packed.is_contiguous() && g_s.is_contiguous() && out_delta_s.is_contiguous(), "All tensors must be contiguous");

    if (num_systems == 0) return;

    const float* H_ptr = gs::torch_utils::get_const_data_ptr<float>(H_s_packed, "H_s_packed");
    const float* g_ptr = gs::torch_utils::get_const_data_ptr<float>(g_s, "g_s");
    float* delta_s_ptr = gs::torch_utils::get_data_ptr<float>(out_delta_s, "out_delta_s");

    batch_solve_3x3_symmetric_system_kernel<<<GET_BLOCKS(num_systems), CUDA_NUM_THREADS>>>(
        num_systems, H_ptr, g_ptr, damping, delta_s_ptr
    );
    CUDA_CHECK(cudaGetLastError());
}

// --- Definitions for Rotation Optimization Launchers (Stubs) ---

void NewtonKernels::compute_rotation_hessian_gradient_components_kernel_launcher(
    int H_img, int W_img, int C_img,
    int P_total,
    const torch::Tensor& means_all,
    const torch::Tensor& scales_all,
    const torch::Tensor& rotations_all,
    const torch::Tensor& opacities_all,
    const torch::Tensor& shs_all,
    int sh_degree,
    const torch::Tensor& view_matrix,
    const torch::Tensor& K_matrix,
    const torch::Tensor& cam_pos_world,
    const torch::Tensor& r_k_vecs,
    const gs::RenderOutput& render_output,
    const torch::Tensor& visible_indices,
    const torch::Tensor& dL_dc_pixelwise,
    const torch::Tensor& d2L_dc2_diag_pixelwise,
    torch::Tensor& out_H_theta,
    torch::Tensor& out_g_theta
) {
    if (out_H_theta.defined()) out_H_theta.zero_(); // Stub behavior
    if (out_g_theta.defined()) out_g_theta.zero_(); // Stub behavior
}

void NewtonKernels::batch_solve_1x1_system_kernel_launcher(
    int num_systems,
    const torch::Tensor& H_theta,
    const torch::Tensor& g_theta,
    float damping,
    torch::Tensor& out_delta_theta
) {
    TORCH_CHECK(H_theta.defined() && H_theta.size(0) == num_systems, "H_theta size mismatch");
    TORCH_CHECK(g_theta.defined() && g_theta.size(0) == num_systems, "g_theta size mismatch");
    TORCH_CHECK(out_delta_theta.defined() && out_delta_theta.size(0) == num_systems, "out_delta_theta size mismatch");
    TORCH_CHECK(H_theta.is_cuda() && g_theta.is_cuda() && out_delta_theta.is_cuda(), "All tensors must be CUDA tensors");

    if (num_systems == 0) return;

    const float* H_ptr = gs::torch_utils::get_const_data_ptr<float>(H_theta.contiguous(), "H_theta");
    const float* g_ptr = gs::torch_utils::get_const_data_ptr<float>(g_theta.contiguous(), "g_theta");
    float* delta_theta_ptr = gs::torch_utils::get_data_ptr<float>(out_delta_theta.contiguous(), "out_delta_theta");

    batch_solve_1x1_system_kernel<<<GET_BLOCKS(num_systems), CUDA_NUM_THREADS>>>(
        num_systems, H_ptr, g_ptr, damping, delta_theta_ptr
    );
    CUDA_CHECK(cudaGetLastError());
}

// --- Definitions for Opacity Optimization Launchers (Stubs) ---

void NewtonKernels::compute_opacity_hessian_gradient_components_kernel_launcher(
    int H_img, int W_img, int C_img,
    int P_total,
    const torch::Tensor& means_all,
    const torch::Tensor& scales_all,
    const torch::Tensor& rotations_all,
    const torch::Tensor& opacities_all,
    const torch::Tensor& shs_all,
    int sh_degree,
    const torch::Tensor& view_matrix,
    const torch::Tensor& K_matrix,
    const torch::Tensor& cam_pos_world,
    const gs::RenderOutput& render_output,
    const torch::Tensor& visible_indices,
    const torch::Tensor& dL_dc_pixelwise,
    const torch::Tensor& d2L_dc2_diag_pixelwise,
    torch::Tensor& out_H_sigma_base,
    torch::Tensor& out_g_sigma_base
) {
    if (out_H_sigma_base.defined()) out_H_sigma_base.zero_();
    if (out_g_sigma_base.defined()) out_g_sigma_base.zero_();
}

// --- Definitions for SH (Color) Optimization Launchers (Stubs) ---

torch::Tensor NewtonKernels::compute_sh_bases_kernel_launcher(
    int sh_degree,
    const torch::Tensor& normalized_view_vectors
) {
    TORCH_CHECK(normalized_view_vectors.defined(), "normalized_view_vectors must be defined.");
    TORCH_CHECK(normalized_view_vectors.dim() == 2 && normalized_view_vectors.size(1) == 3,
                "normalized_view_vectors must have shape [N, 3]. Got ", normalized_view_vectors.sizes());
    TORCH_CHECK(normalized_view_vectors.is_cuda(), "normalized_view_vectors must be a CUDA tensor.");
    TORCH_CHECK(normalized_view_vectors.is_contiguous(), "normalized_view_vectors must be contiguous.");
    TORCH_CHECK(sh_degree >= 0 && sh_degree <= 4, "sh_degree must be between 0 and 4. Got ", sh_degree);

    const int num_points = normalized_view_vectors.size(0);
    if (num_points == 0) {
        return torch::empty({0, (sh_degree + 1) * (sh_degree + 1)}, normalized_view_vectors.options());
    }

    const int num_sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
    torch::Tensor sh_bases_tensor = torch::empty({num_points, num_sh_coeffs}, normalized_view_vectors.options());

    const float* dirs_ptr = gs::torch_utils::get_const_data_ptr<float>(normalized_view_vectors, "normalized_view_vectors");
    float* sh_basis_output_ptr = gs::torch_utils::get_data_ptr<float>(sh_bases_tensor, "sh_bases_tensor");

    eval_sh_basis_kernel<<<GET_BLOCKS(num_points), CUDA_NUM_THREADS>>>(
        num_points, sh_degree, dirs_ptr, sh_basis_output_ptr
    );
    CUDA_CHECK(cudaGetLastError());
    return sh_bases_tensor;
}

void NewtonKernels::compute_sh_hessian_gradient_components_kernel_launcher(
    int H_img, int W_img, int C_img,
    int P_total,
    const torch::Tensor& means_all,
    const torch::Tensor& scales_all,
    const torch::Tensor& rotations_all,
    const torch::Tensor& opacities_all,
    const torch::Tensor& shs_all,
    int sh_degree,
    const torch::Tensor& sh_bases_values,
    const torch::Tensor& view_matrix,
    const torch::Tensor& proj_param_for_sh_hess, // Renamed from K_matrix
    const gs::RenderOutput& render_output,
    const torch::Tensor& visible_indices,
    const torch::Tensor& dL_dc_pixelwise,
    const torch::Tensor& d2L_dc2_diag_pixelwise,
    torch::Tensor& out_H_ck_diag,
    torch::Tensor& out_g_ck
) {
    // Stub - body intentionally empty for debugging
}
} // namespace NewtonKernels
