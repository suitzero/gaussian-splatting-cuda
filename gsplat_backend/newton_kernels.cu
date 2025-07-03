// kernels/newton_kernels.cu
#include "newton_kernels.cuh" // Now in the same directory
#include "kernels/ssim.cuh"   // For fusedssim, fusedssim_backward C++ functions
#include <cuda_runtime.h> // Includes vector_types.h for ::float3, ::float2
#include <device_launch_parameters.h>
#include <torch/torch.h> // For AT_ASSERTM
#include <cmath> // For fabsf, sqrtf, etc.

// GLM includes
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp> // For glm::value_ptr if needed
#include <glm/gtx/norm.hpp>     // For glm::length2 (squared length)


// Basic CUDA utilities (normally in a separate header)
#define CUDA_CHECK(status) AT_ASSERTM(status == cudaSuccess, cudaGetErrorString(status))

constexpr int CUDA_NUM_THREADS = 256; // Default number of threads per block
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// --- CudaMath namespace: To be phased out or significantly reduced ---
// Vector functions are replaced by GLM operators and functions.
// Matrix functions operating on float* might be kept temporarily if direct GLM matrix types
// are not immediately propagated everywhere.
namespace CudaMath {

// The following vector functions are now OBSOLETE and should be replaced by GLM equivalents:
// add_float3 -> a + b
// sub_float3 -> a - b
// mul_float3_scalar -> v * s
// div_float3_scalar -> v / s (with epsilon check or ensure s != 0)
// dot_product -> glm::dot(a,b)
// cross_product -> glm::cross(a,b)
// length_sq_float3 -> glm::length2(v) or glm::dot(v,v)
// length_float3 -> glm::length(v)
// normalize_vec3 -> glm::normalize(v)
// mul_mat3_vec3 (if M becomes glm::mat3) -> M * v

// Keep matrix operations on raw pointers if still needed during transition
__device__ __forceinline__ void mat3_transpose_inplace(float* M) {
    float temp;
    temp = M[1]; M[1] = M[3]; M[3] = temp;
    temp = M[2]; M[2] = M[6]; M[6] = temp;
    temp = M[5]; M[5] = M[7]; M[7] = temp;
}

// This can be replaced by glm::outerProduct(a,b) if result is glm::mat3
__device__ __forceinline__ void outer_product_3x3_ptr(const glm::vec3& a, const glm::vec3& b, float* out_M_ptr) {
    glm::mat3 result_mat = glm::outerProduct(a,b);
    // GLM matrices are column-major. If out_M_ptr expects row-major:
    const float* p = glm::value_ptr(result_mat);
    out_M_ptr[0] = p[0]; out_M_ptr[1] = p[3]; out_M_ptr[2] = p[6]; // col 0
    out_M_ptr[3] = p[1]; out_M_ptr[4] = p[4]; out_M_ptr[5] = p[7]; // col 1
    out_M_ptr[6] = p[2]; out_M_ptr[7] = p[5]; out_M_ptr[8] = p[8]; // col 2
    // If out_M_ptr expects column-major, then simple memcpy or loop.
    // For now, assuming row-major output for compatibility with original code.
}


__device__ __forceinline__ void mul_mat4_vec4_ptr(const float* PW_ptr, const float* p_k_h_ptr, float* result_ptr) {
    // This function assumes PW_ptr is row-major.
    // If to be replaced by GLM: glm::mat4 PW_glm = glm::transpose(glm::make_mat4(PW_ptr)); // if PW_ptr is row-major
    // glm::vec4 pk_h_glm = glm::make_vec4(p_k_h_ptr);
    // glm::vec4 result_glm = PW_glm * pk_h_glm;
    // then store result_glm back to result_ptr.
    for (int i = 0; i < 4; ++i) {
        result_ptr[i] = 0;
        for (int j = 0; j < 4; ++j) {
            result_ptr[i] += PW_ptr[i * 4 + j] * p_k_h_ptr[j];
        }
    }
}

// A (A_rows x A_cols_B_rows), B (A_cols_B_rows x B_cols), C (A_rows x B_cols)
// All row-major, raw pointers
__device__ __forceinline__ void mat_mul_mat_ptr(const float* A_ptr, const float* B_ptr, float* C_ptr,
                                             int A_rows, int A_cols_B_rows, int B_cols) {
    // Example: C_glm = glm::make_mat_col_major(A_ptr) * glm::make_mat_col_major(B_ptr) (if A,B are col-major)
    // Or if A,B row-major: C_glm = glm::transpose(glm::make_mat4(A_ptr)) * glm::transpose(glm::make_mat4(B_ptr))
    // then store C_glm (transposed if C_ptr needs row-major).
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols_B_rows; ++k) {
                sum += A_ptr[i * A_cols_B_rows + k] * B_ptr[k * B_cols + j];
            }
            C_ptr[i * B_cols + j] = sum;
        }
    }
}

} // namespace CudaMath


// --- Projection Derivative Helper Functions ---
namespace ProjectionDerivs {

__device__ __forceinline__ void compute_h_vec(const glm::vec3& p_k, const float* PW_ptr, float* h_vec4_ptr) { // Changed to glm::vec3, PW is float*
    float p_k_h[4] = {p_k.x, p_k.y, p_k.z, 1.0f};
    CudaMath::mul_mat4_vec4_ptr(PW_ptr, p_k_h, h_vec4_ptr);
}

__device__ __forceinline__ void compute_projection_jacobian(
    const float* PW_ptr, float W_I_t, float H_I_t, // PW_ptr is float*
    const float* h_vec4, float* jacobian_out_2x3
) {


// --- Projection Derivative Helper Functions ---
namespace ProjectionDerivs {

__device__ __forceinline__ void compute_h_vec(const glm::vec3& p_k, const glm::mat4& PW, glm::vec4& h_vec_out) {
    h_vec_out = PW * glm::vec4(p_k, 1.0f);
}

// Output jacobian_out_2x3_row_major is float[6] representing a 2x3 row-major matrix
__device__ __forceinline__ void compute_projection_jacobian(
    const glm::mat4& PW_col_major, float W_I_t, float H_I_t,
    const glm::vec4& h_vec, float* jacobian_out_2x3_row_major
) {
    float hx = h_vec.x;
    float hy = h_vec.y;
    float hw = h_vec.w;
    float inv_hw = 1.0f / (hw + 1e-8f);
    float inv_hw_sq = inv_hw * inv_hw;

    float term_x_coeff = W_I_t / 2.0f;
    float term_y_coeff = H_I_t / 2.0f;

    // GLM stores matrices in column-major order. PW_col_major[col][row]
    // The formula (PW)_i refers to the i-th row of the conceptual matrix PW.
    // So, (PW)_0 = first row of PW = (PW_col_major[0][0], PW_col_major[1][0], PW_col_major[2][0])
    // (PW)_3 = fourth row of PW = (PW_col_major[0][3], PW_col_major[1][3], PW_col_major[2][3])

    // Row 0 of Jacobian (∂π_x / ∂p_k) (pk_x, pk_y, pk_z)
    jacobian_out_2x3_row_major[0] = term_x_coeff * (inv_hw * PW_col_major[0][0] - hx * inv_hw_sq * PW_col_major[0][3]); // d(pi_x)/d(pk_x)
    jacobian_out_2x3_row_major[1] = term_x_coeff * (inv_hw * PW_col_major[1][0] - hx * inv_hw_sq * PW_col_major[1][3]); // d(pi_x)/d(pk_y)
    jacobian_out_2x3_row_major[2] = term_x_coeff * (inv_hw * PW_col_major[2][0] - hx * inv_hw_sq * PW_col_major[2][3]); // d(pi_x)/d(pk_z)

    // Row 1 of Jacobian (∂π_y / ∂p_k) (pk_x, pk_y, pk_z)
    jacobian_out_2x3_row_major[3] = term_y_coeff * (inv_hw * PW_col_major[0][1] - hy * inv_hw_sq * PW_col_major[0][3]); // d(pi_y)/d(pk_x)
    jacobian_out_2x3_row_major[4] = term_y_coeff * (inv_hw * PW_col_major[1][1] - hy * inv_hw_sq * PW_col_major[1][3]); // d(pi_y)/d(pk_y)
    jacobian_out_2x3_row_major[5] = term_y_coeff * (inv_hw * PW_col_major[2][1] - hy * inv_hw_sq * PW_col_major[2][3]); // d(pi_y)/d(pk_z)
}

__device__ __forceinline__ void compute_projection_hessian(
    const glm::mat4& PW_col_major, float W_I_t, float H_I_t,
    const glm::vec4& h_vec,
    float* hessian_out_pi_x_3x3_row_major,
    float* hessian_out_pi_y_3x3_row_major
) {
    float hx = h_vec.x;
    float hy = h_vec.y;
    float hw = h_vec.w;
    float inv_hw_sq = 1.0f / (hw * hw + 1e-8f);
    float hw_cubed = hw * hw * hw + 1e-9f; // Avoid division by zero

    // (PW)_i notation from paper means i-th row of the matrix PW.
    // Since PW_col_major is column-major (GLM default), PW_col_major[c][r]:
    // (PW)_0 (row 0) = glm::vec3(PW_col_major[0][0], PW_col_major[1][0], PW_col_major[2][0])
    // (PW)_1 (row 1) = glm::vec3(PW_col_major[0][1], PW_col_major[1][1], PW_col_major[2][1])
    // (PW)_3 (row 3, xyz part) = glm::vec3(PW_col_major[0][3], PW_col_major[1][3], PW_col_major[2][3])
    glm::vec3 PWr0 = glm::vec3(PW_col_major[0][0], PW_col_major[1][0], PW_col_major[2][0]);
    glm::vec3 PWr1 = glm::vec3(PW_col_major[0][1], PW_col_major[1][1], PW_col_major[2][1]);
    glm::vec3 PWr3 = glm::vec3(PW_col_major[0][3], PW_col_major[1][3], PW_col_major[2][3]);

    glm::mat3 PW3_outer_PW3 = glm::outerProduct(PWr3, PWr3);
    glm::mat3 PW3_outer_PW0 = glm::outerProduct(PWr3, PWr0);
    glm::mat3 PW0_outer_PW3 = glm::transpose(PW3_outer_PW0); // ( (PW_3)^T (PW_0) )^T  is (PW_0)^T (PW_3)
    glm::mat3 PW3_outer_PW1 = glm::outerProduct(PWr3, PWr1);
    glm::mat3 PW1_outer_PW3 = glm::transpose(PW3_outer_PW1);


    float factor_x1 = W_I_t * (2.0f * hx / hw_cubed);
    float factor_x2 = W_I_t * (-1.0f * inv_hw_sq);

    glm::mat3 H_pi_x_col_major = factor_x1 * PW3_outer_PW3 + factor_x2 * (PW3_outer_PW0 + PW0_outer_PW3);

    // Store GLM column-major mat3 to row-major float array
    const float* p_x = glm::value_ptr(H_pi_x_col_major);
    hessian_out_pi_x_3x3_row_major[0] = p_x[0]; hessian_out_pi_x_3x3_row_major[1] = p_x[3]; hessian_out_pi_x_3x3_row_major[2] = p_x[6];
    hessian_out_pi_x_3x3_row_major[3] = p_x[1]; hessian_out_pi_x_3x3_row_major[4] = p_x[4]; hessian_out_pi_x_3x3_row_major[5] = p_x[7];
    hessian_out_pi_x_3x3_row_major[6] = p_x[2]; hessian_out_pi_x_3x3_row_major[7] = p_x[5]; hessian_out_pi_x_3x3_row_major[8] = p_x[8];

    float factor_y1 = H_I_t * (2.0f * hy / hw_cubed);
    float factor_y2 = H_I_t * (-1.0f * inv_hw_sq);

    glm::mat3 H_pi_y_col_major = factor_y1 * PW3_outer_PW3 + factor_y2 * (PW3_outer_PW1 + PW1_outer_PW3);
    const float* p_y = glm::value_ptr(H_pi_y_col_major);
    hessian_out_pi_y_3x3_row_major[0] = p_y[0]; hessian_out_pi_y_3x3_row_major[1] = p_y[3]; hessian_out_pi_y_3x3_row_major[2] = p_y[6];
    hessian_out_pi_y_3x3_row_major[3] = p_y[1]; hessian_out_pi_y_3x3_row_major[4] = p_y[4]; hessian_out_pi_y_3x3_row_major[5] = p_y[7];
    hessian_out_pi_y_3x3_row_major[6] = p_y[2]; hessian_out_pi_y_3x3_row_major[7] = p_y[5]; hessian_out_pi_y_3x3_row_major[8] = p_y[8];
}
} // namespace ProjectionDerivs

// --- SH Basis and Color Derivative Helper Functions ---
namespace SHDerivs {

__device__ __forceinline__ void eval_sh_basis_up_to_degree3(
    int degree, const glm::vec3& r_k_normalized, float* basis_out // Using glm::vec3
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

// drk_dpk_out_3x3 is still float* for now, but represents a 3x3 matrix.
// It could be changed to output glm::mat3, and the caller would use glm::value_ptr if needed.
__device__ __forceinline__ void compute_drk_dpk(
    const glm::vec3& r_k_normalized, float r_k_norm,
    float* drk_dpk_out_3x3_row_major // Output as row-major float[9]
) {
    float inv_r_k_norm = 1.0f / (r_k_norm + 1e-8f);
    glm::mat3 I_minus_rktrk = glm::mat3(1.0f) - glm::outerProduct(r_k_normalized, r_k_normalized);
    glm::mat3 result_col_major = inv_r_k_norm * I_minus_rktrk;

    // Store GLM column-major mat3 to row-major float array
    const float* p = glm::value_ptr(result_col_major);
    drk_dpk_out_3x3_row_major[0] = p[0]; drk_dpk_out_3x3_row_major[1] = p[3]; drk_dpk_out_3x3_row_major[2] = p[6];
    drk_dpk_out_3x3_row_major[3] = p[1]; drk_dpk_out_3x3_row_major[4] = p[4]; drk_dpk_out_3x3_row_major[5] = p[7];
    drk_dpk_out_3x3_row_major[6] = p[2]; drk_dpk_out_3x3_row_major[7] = p[5]; drk_dpk_out_3x3_row_major[8] = p[8];
}

// dPhi_drk_out is still float* (num_coeffs * 3), row-major (dPhi0/drx, dPhi0/dry, dPhi0/drz, dPhi1/drx, ...)
__device__ __forceinline__ void compute_dphi_drk_up_to_degree3(
    int degree, const glm::vec3& r_k_normalized, float* dPhi_drk_out // Using glm::vec3
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

// Output jac_out_3 is float* but will store components of a glm::vec3
__device__ __forceinline__ void compute_sh_color_jacobian_single_channel(
    const float* sh_coeffs_single_channel, const float* sh_basis_values,
    const float* dPhi_drk_ptr, // (num_coeffs x 3) matrix
    const float* drk_dpk_ptr,  // (3x3) matrix
    int num_basis_coeffs,
    float* jac_out_3_ptr
) {
    // Convert drk_dpk_ptr (row-major 3x3) to glm::mat3 (column-major)
    glm::mat3 drk_dpk_mat = glm::transpose(glm::make_mat3(drk_dpk_ptr));

    // M_prod = dPhi_drk * drk_dpk_mat ( (num_coeffs x 3) * (3x3) = (num_coeffs x 3) )
    // dPhi_drk is row-major: [dPhi0/drx, dPhi0/dry, dPhi0/drz, dPhi1/drx, ...]
    // M_prod_ij = sum_l (dPhi_drk_il * drk_dpk_mat_lj)
    float M_prod_row_major[16*3]; // Max ( (3+1)^2 * 3 )

    for(int i=0; i < num_basis_coeffs; ++i) { // M_prod row index
        glm::vec3 dPhi_drk_row_i = glm::vec3(dPhi_drk_ptr[i*3+0], dPhi_drk_ptr[i*3+1], dPhi_drk_ptr[i*3+2]);
        glm::vec3 result_row = dPhi_drk_row_i * drk_dpk_mat; // Vector * matrix (row vector semantic)
        M_prod_row_major[i*3+0] = result_row.x;
        M_prod_row_major[i*3+1] = result_row.y;
        M_prod_row_major[i*3+2] = result_row.z;
    }

    jac_out_3_ptr[0] = 0.0f; jac_out_3_ptr[1] = 0.0f; jac_out_3_ptr[2] = 0.0f;
    for (int i = 0; i < num_basis_coeffs; ++i) {
        float v_i = sh_basis_values[i] * sh_coeffs_single_channel[i];
        jac_out_3_ptr[0] += v_i * M_prod_row_major[i * 3 + 0];
        jac_out_3_ptr[1] += v_i * M_prod_row_major[i * 3 + 1];
        jac_out_3_ptr[2] += v_i * M_prod_row_major[i * 3 + 2];
    }
}
} // namespace SHDerivs

// --- KERNEL DEFINITIONS ---

__device__ __forceinline__ void get_projected_cov2d_and_derivs_placeholder(
    const glm::vec3& p_k_world, // Using glm::vec3
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
    const float* view_matrix_ptr,        // Changed to _ptr to indicate it's a raw pointer
    const float* perspective_proj_matrix_ptr, // Changed from projection_matrix_for_jacobian
    const float* cam_pos_world_ptr,      // Changed to _ptr
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

    glm::vec3 pk_vec3 = glm::vec3(
        means_3d_all[p_idx_total * 3 + 0],
        means_3d_all[p_idx_total * 3 + 1],
        means_3d_all[p_idx_total * 3 + 2]);

    const float* scales_k = scales_all + p_idx_total * 3;
    const float* rotations_k = rotations_all + p_idx_total * 4;
    float opacity_k = opacities_all[p_idx_total];
    const float* sh_coeffs_k_all_channels = shs_all + p_idx_total * sh_coeffs_per_color_channel * 3;

    glm::vec3 cam_pos_world_vec3 = glm::vec3(cam_pos_world_ptr[0], cam_pos_world_ptr[1], cam_pos_world_ptr[2]);

    glm::vec3 view_dir_to_pk_unnormalized = pk_vec3 - cam_pos_world_vec3;
    float r_k_norm = glm::length(view_dir_to_pk_unnormalized);
    glm::vec3 r_k_normalized = glm::normalize(view_dir_to_pk_unnormalized);

    // Load raw pointers into GLM matrices. Assume row-major storage for input pointers.
    // GLM matrices are column-major. glm::make_mat4 assumes column-major input from pointer.
    // So, if input is row-major, we need to transpose after loading or load manually.
    glm::mat4 V_col_major = glm::transpose(glm::make_mat4(view_matrix_ptr));
    glm::mat4 P_col_major = glm::transpose(glm::make_mat4(perspective_proj_matrix_ptr));
    glm::mat4 PW_col_major = P_col_major * V_col_major;

    glm::vec4 h_vec4_data; // Changed from float h_vec4_data[4]
    ProjectionDerivs::compute_h_vec(pk_vec3, PW_col_major, h_vec4_data); // Pass glm::mat4

    float d_pi_d_pk_data_row_major[6];
    ProjectionDerivs::compute_projection_jacobian(PW_col_major, (float)W_img, (float)H_img, h_vec4_data, d_pi_d_pk_data_row_major);

    float d2_pi_d_pk2_x_data_row_major[9];
    float d2_pi_d_pk2_y_data_row_major[9];
    ProjectionDerivs::compute_projection_hessian(PW_col_major, (float)W_img, (float)H_img, h_vec4_data,
                                                 d2_pi_d_pk2_x_data_row_major, d2_pi_d_pk2_y_data_row_major);

    float sh_basis_eval_data[16];
    SHDerivs::eval_sh_basis_up_to_degree3(sh_degree, r_k_normalized, sh_basis_eval_data);

    float d_rk_d_pk_data_row_major[9];
    SHDerivs::compute_drk_dpk(r_k_normalized, r_k_norm, d_rk_d_pk_data_row_major);

    float d_phi_d_rk_data_row_major[16*3];
    SHDerivs::compute_dphi_drk_up_to_degree3(sh_degree, r_k_normalized, d_phi_d_rk_data_row_major);

    glm::vec3 d_c_bar_R_d_pk_val, d_c_bar_G_d_pk_val, d_c_bar_B_d_pk_val;
    float sh_coeffs_k_R[16], sh_coeffs_k_G[16], sh_coeffs_k_B[16];
    for(int i=0; i<sh_coeffs_per_color_channel; ++i) {
        sh_coeffs_k_R[i] = sh_coeffs_k_all_channels[i*3 + 0];
        sh_coeffs_k_G[i] = sh_coeffs_k_all_channels[i*3 + 1];
        sh_coeffs_k_B[i] = sh_coeffs_k_all_channels[i*3 + 2];
    }

    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_R, sh_basis_eval_data, d_phi_d_rk_data_row_major, d_rk_d_pk_data_row_major, sh_coeffs_per_color_channel, glm::value_ptr(d_c_bar_R_d_pk_val));
    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_G, sh_basis_eval_data, d_phi_d_rk_data_row_major, d_rk_d_pk_data_row_major, sh_coeffs_per_color_channel, glm::value_ptr(d_c_bar_G_d_pk_val));
    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_B, sh_basis_eval_data, d_phi_d_rk_data_row_major, d_rk_d_pk_data_row_major, sh_coeffs_per_color_channel, glm::value_ptr(d_c_bar_B_d_pk_val));

    glm::vec3 g_p_k_accum_val = glm::vec3(0.f, 0.f, 0.f);
    float H_p_k_accum_symm[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int r = 0; r < H_img; ++r) {
        for (int c = 0; c < W_img; ++c) {
            float pixel_ndc_x = (2.0f * (c + 0.5f) / W_img - 1.0f);
            float pixel_ndc_y = (2.0f * (r + 0.5f) / H_img - 1.0f);

            // pi_k is the projected center of the Gaussian pk_vec3
            // h_vec4_data = P_col_major * V_col_major * pk_world_homo
            // pi_k_ndc_x_unscaled = h_vec4_data.x / h_vec4_data.w (already in NDC if P_col_major is to NDC)
            float pi_k_ndc_x_unscaled = h_vec4_data.x / (h_vec4_data.w + 1e-7f);
            float pi_k_ndc_y_unscaled = h_vec4_data.y / (h_vec4_data.w + 1e-7f);

            glm::vec2 diff_ndc_val = glm::vec2(pi_k_ndc_x_unscaled - pixel_ndc_x, pi_k_ndc_y_unscaled - pixel_ndc_y);

            float cov2d_sym_data[3], inv_cov2d_sym_data[3], det_cov2d_data;
            float d_Gk_d_pik_data[2];
            float d2_Gk_d_pik2_data[3];

            // get_projected_cov2d_and_derivs_placeholder expects float* for view and proj matrices
            // but we have glm::mat4 V_col_major and P_col_major (and PW_col_major)
            // For now, pass the original pointers, assuming the placeholder can handle them or is adapted.
            // The d_pi_d_pk_data_row_major is float[6]
            get_projected_cov2d_and_derivs_placeholder(pk_vec3, scales_k, rotations_k,
                                                       view_matrix_ptr, perspective_proj_matrix_ptr,
                                                       d_pi_d_pk_data_row_major, (float)W_img, (float)H_img,
                                                       cov2d_sym_data, inv_cov2d_sym_data, &det_cov2d_data,
                                                       nullptr, nullptr);

            float G_k_pixel = expf(-0.5f * (diff_ndc_val.x*diff_ndc_val.x*inv_cov2d_sym_data[0] +
                                            2*diff_ndc_val.x*diff_ndc_val.y*inv_cov2d_sym_data[1] +
                                            diff_ndc_val.y*diff_ndc_val.y*inv_cov2d_sym_data[2]));
            if (det_cov2d_data <= 1e-7f) G_k_pixel = 0.f;

            if (G_k_pixel < 1e-4f) continue;

            glm::vec2 sigma_inv_diff_val;
            sigma_inv_diff_val.x = inv_cov2d_sym_data[0]*diff_ndc_val.x + inv_cov2d_sym_data[1]*diff_ndc_val.y;
            sigma_inv_diff_val.y = inv_cov2d_sym_data[1]*diff_ndc_val.x + inv_cov2d_sym_data[2]*diff_ndc_val.y;
            d_Gk_d_pik_data[0] = -G_k_pixel * sigma_inv_diff_val.x;
            d_Gk_d_pik_data[1] = -G_k_pixel * sigma_inv_diff_val.y;

            d2_Gk_d_pik2_data[0] = G_k_pixel * (sigma_inv_diff_val.x * sigma_inv_diff_val.x - inv_cov2d_sym_data[0]);
            d2_Gk_d_pik2_data[1] = G_k_pixel * (sigma_inv_diff_val.x * sigma_inv_diff_val.y - inv_cov2d_sym_data[1]);
            d2_Gk_d_pik2_data[2] = G_k_pixel * (sigma_inv_diff_val.y * sigma_inv_diff_val.y - inv_cov2d_sym_data[2]);


            float alpha_k_pixel = opacity_k * G_k_pixel;

            glm::vec3 c_bar_k_rgb_val;
            c_bar_k_rgb_val.x =0; for(int i=0; i<sh_coeffs_per_color_channel; ++i) c_bar_k_rgb_val.x += sh_coeffs_k_R[i] * sh_basis_eval_data[i];
            c_bar_k_rgb_val.y =0; for(int i=0; i<sh_coeffs_per_color_channel; ++i) c_bar_k_rgb_val.y += sh_coeffs_k_G[i] * sh_basis_eval_data[i];
            c_bar_k_rgb_val.z =0; for(int i=0; i<sh_coeffs_per_color_channel; ++i) c_bar_k_rgb_val.z += sh_coeffs_k_B[i] * sh_basis_eval_data[i];


            glm::vec3 d_c_final_d_Gk_val = c_bar_k_rgb_val * opacity_k;

            glm::vec3 d_Gk_d_pk_chain_val;
            // d_pi_d_pk_data_row_major is 2x3 row-major: [J11 J12 J13 J21 J22 J23]
            // d_Gk_d_pik_data is [dG/dπx, dG/dπy] (1x2)
            // d_Gk_d_pk_chain = d_Gk_d_pik * d_pi_d_pk
            d_Gk_d_pk_chain_val.x = d_Gk_d_pik_data[0] * d_pi_d_pk_data_row_major[0] + d_Gk_d_pik_data[1] * d_pi_d_pk_data_row_major[3]; // col 0 of J_c_pk
            d_Gk_d_pk_chain_val.y = d_Gk_d_pik_data[0] * d_pi_d_pk_data_row_major[1] + d_Gk_d_pik_data[1] * d_pi_d_pk_data_row_major[4]; // col 1 of J_c_pk
            d_Gk_d_pk_chain_val.z = d_Gk_d_pik_data[0] * d_pi_d_pk_data_row_major[2] + d_Gk_d_pik_data[1] * d_pi_d_pk_data_row_major[5]; // col 2 of J_c_pk


            glm::vec3 J_c_pk_R_val, J_c_pk_G_val, J_c_pk_B_val;
            J_c_pk_R_val = d_c_bar_R_d_pk_val * alpha_k_pixel + d_Gk_d_pk_chain_val * d_c_final_d_Gk_val.x;
            J_c_pk_G_val = d_c_bar_G_d_pk_val * alpha_k_pixel + d_Gk_d_pk_chain_val * d_c_final_d_Gk_val.y;
            J_c_pk_B_val = d_c_bar_B_d_pk_val * alpha_k_pixel + d_Gk_d_pk_chain_val * d_c_final_d_Gk_val.z;

            int pixel_idx_flat = (r * W_img + c) * C_img;
            glm::vec3 dL_dc_val_pixel = glm::vec3(dL_dc_pixelwise[pixel_idx_flat+0], dL_dc_pixelwise[pixel_idx_flat+1], dL_dc_pixelwise[pixel_idx_flat+2]);
            glm::vec3 d2L_dc2_diag_val_pixel = glm::vec3(d2L_dc2_diag_pixelwise[pixel_idx_flat+0], d2L_dc2_diag_pixelwise[pixel_idx_flat+1], d2L_dc2_diag_pixelwise[pixel_idx_flat+2]);

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
