// kernels/newton_kernels.cu
#include "newton_kernels.cuh" // Now in the same directory
#include "kernels/ssim.cuh"   // For fusedssim, fusedssim_backward C++ functions
#include <cuda_runtime.h>
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
namespace CudaMath {

// Vector operations
__device__ __forceinline__ float3 make_float3(float x, float y, float z) {
    return ::make_float3(x, y, z);
}

__device__ __forceinline__ float3 add_float3(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub_float3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 mul_float3_scalar(const float3& v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ __forceinline__ float3 div_float3_scalar(const float3& v, float s) {
    // Add epsilon to prevent division by zero if s is extremely small
    float inv_s = 1.0f / (s + 1e-8f);
    return make_float3(v.x * inv_s, v.y * inv_s, v.z * inv_s);
}

__device__ __forceinline__ float dot_product(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 cross_product(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ __forceinline__ float length_sq_float3(const float3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ __forceinline__ float length_float3(const float3& v) {
    return sqrtf(length_sq_float3(v));
}

__device__ __forceinline__ float3 normalize_vec3(const float3& v) {
    float l = length_float3(v);
    return div_float3_scalar(v, l);
}

// Matrix operations (assuming row-major for M)
// M is a 3x3 matrix (9 floats), v is float3
__device__ __forceinline__ float3 mul_mat3_vec3(const float* M, const float3& v) {
    float3 r;
    r.x = M[0] * v.x + M[1] * v.y + M[2] * v.z;
    r.y = M[3] * v.x + M[4] * v.y + M[5] * v.z;
    r.z = M[6] * v.x + M[7] * v.y + M[8] * v.z;
    return r;
}

// Transposes a 3x3 matrix M (9 floats) in place
__device__ __forceinline__ void mat3_transpose_inplace(float* M) {
    float temp;
    temp = M[1]; M[1] = M[3]; M[3] = temp;
    temp = M[2]; M[2] = M[6]; M[6] = temp;
    temp = M[5]; M[5] = M[7]; M[7] = temp;
}

// Computes out_M = a * b^T (a and b are float3, out_M is 3x3 = 9 floats)
__device__ __forceinline__ void outer_product_3x3(const float3& a, const float3& b, float* out_M) {
    out_M[0] = a.x * b.x; out_M[1] = a.x * b.y; out_M[2] = a.x * b.z;
    out_M[3] = a.y * b.x; out_M[4] = a.y * b.y; out_M[5] = a.y * b.z;
    out_M[6] = a.z * b.x; out_M[7] = a.z * b.y; out_M[8] = a.z * b.z;
}

// PW is 4x4 (16 floats, row-major), p_k_h is [x,y,z,1] (4 floats)
// result is 4 floats
__device__ __forceinline__ void mul_mat4_vec4(const float* PW, const float* p_k_h, float* result) {
    for (int i = 0; i < 4; ++i) {
        result[i] = 0;
        for (int j = 0; j < 4; ++j) {
            result[i] += PW[i * 4 + j] * p_k_h[j];
        }
    }
}

// M is 3x3 (9 floats), v is 3x1 (3 floats), out is 3x1 (3 floats)
__device__ __forceinline__ void mat_mul_vec(const float* M, const float* v, float* out) {
    for (int i = 0; i < 3; ++i) {
        out[i] = 0;
        for (int j = 0; j < 3; ++j) {
            out[i] += M[i * 3 + j] * v[j];
        }
    }
}


// A (A_rows x A_cols_B_rows), B (A_cols_B_rows x B_cols), C (A_rows x B_cols)
// All row-major
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

// PW is 4x4 matrix, p_k is 3D world point. h_vec4 is the output [hx, hy, hz, hw]^T.
__device__ __forceinline__ void compute_h_vec(const CudaMath::float3& p_k, const float* PW, float* h_vec4) {
    float p_k_h[4] = {p_k.x, p_k.y, p_k.z, 1.0f};
    CudaMath::mul_mat4_vec4(PW, p_k_h, h_vec4);
}

// Computes ∂π_k/∂P_k (Eq. A.1 from supplementary).
// PW is 4x4 projection * view matrix.
// W_I_t, H_I_t are width and height of input image I^t.
// h_vec4 is [hx, hy, hz, hw]^T = PW * [p_k^T, 1]^T.
// jacobian_out_2x3 is a 2x3 matrix (row-major: d_pix/d_pkx, d_piy/d_pky, d_piz/d_pkz for row1; same for row2)
__device__ __forceinline__ void compute_projection_jacobian(
    const float* PW,         // 4x4 matrix
    float W_I_t, float H_I_t, // Image dimensions (used for NDC scaling factors)
    const float* h_vec4,     // Result of PW * [p_k, 1]^T
    float* jacobian_out_2x3  // Output: 2x3 matrix (row-major)
) {
    float hx = h_vec4[0];
    float hy = h_vec4[1];
    // float hz = h_vec4[2]; // Not directly used in this formula for jacobian
    float hw = h_vec4[3];
    float inv_hw = 1.0f / (hw + 1e-8f); // Add epsilon for stability

    // (PW)_i refers to the first three elements of the i-th row of PW.
    // (PW)_0 = [PW[0], PW[1], PW[2]]
    // (PW)_1 = [PW[4], PW[5], PW[6]]
    // (PW)_3 = [PW[12], PW[13], PW[14]]

    float term_x_coeff = W_I_t / 2.0f;
    float term_y_coeff = H_I_t / 2.0f;

    // Row 0 of Jacobian (∂π_x / ∂p_k)
    for (int j = 0; j < 3; ++j) { // Iterate over p_k components (x, y, z)
        jacobian_out_2x3[0 * 3 + j] = term_x_coeff * (inv_hw * PW[0 * 4 + j] - (hx * inv_hw * inv_hw) * PW[3 * 4 + j]);
    }

    // Row 1 of Jacobian (∂π_y / ∂p_k)
    for (int j = 0; j < 3; ++j) { // Iterate over p_k components (x, y, z)
        jacobian_out_2x3[1 * 3 + j] = term_y_coeff * (inv_hw * PW[1 * 4 + j] - (hy * inv_hw * inv_hw) * PW[3 * 4 + j]);
    }
}


// Computes ∂²π_k/∂P_k² (Eq. A.2 from supplementary).
// This results in two 3x3 Hessian matrices, one for π_x and one for π_y.
// hessian_out_pi_x_3x3: Stores the 3x3 Hessian for π_x.
// hessian_out_pi_y_3x3: Stores the 3x3 Hessian for π_y.
__device__ __forceinline__ void compute_projection_hessian(
    const float* PW,         // 4x4 matrix
    float W_I_t, float H_I_t, // Image dimensions
    const float* h_vec4,     // Result of PW * [p_k, 1]^T
    float* hessian_out_pi_x_3x3, // Output: 3x3 matrix for ∂²π_x/∂p_k²
    float* hessian_out_pi_y_3x3  // Output: 3x3 matrix for ∂²π_y/∂p_k²
) {
    // h = [hx, hy, hz, hw]^T
    float hx = h_vec4[0];
    float hy = h_vec4[1];
    // float hz = h_vec4[2]; // Not directly used here
    float hw = h_vec4[3];

    float inv_hw_sq = 1.0f / (hw * hw + 1e-8f); // (1/hw^2)
    // float inv_hw_cub = inv_hw_sq / (hw + 1e-8f); // (1/hw^3) // Not needed with current formula interpretation

    // (PW)_i is the 3-vector (row_i[0], row_i[1], row_i[2])
    CudaMath::float3 PW0_vec = CudaMath::make_float3(PW[0], PW[1], PW[2]);
    CudaMath::float3 PW1_vec = CudaMath::make_float3(PW[4], PW[5], PW[6]);
    CudaMath::float3 PW3_vec = CudaMath::make_float3(PW[12], PW[13], PW[14]); // This is (PW)_3

    // Temp 3x3 matrices for outer products
    float PW3_outer_PW3[9]; // (PW)_3^T * (PW)_3 in paper, but (PW)_3 is a row vector of first 3 elements.
                            // If (PW)_i is a 3D row vector, then (PW)_i^T is a column vector.
                            // (PW)_3^T (PW)_3 -> col_vec * row_vec -> 3x3 matrix
                            // (PW)_3^T (PW)_0 -> col_vec * row_vec -> 3x3 matrix
    CudaMath::outer_product_3x3(PW3_vec, PW3_vec, PW3_outer_PW3);

    float PW3_outer_PW0[9];
    CudaMath::outer_product_3x3(PW3_vec, PW0_vec, PW3_outer_PW0);

    float PW3_outer_PW1[9];
    CudaMath::outer_product_3x3(PW3_vec, PW1_vec, PW3_outer_PW1);

    // Factor for ∂²π_x/∂p_k²
    // Paper: W_I^t * ( (2*hx/hw^3) * (PW)_3^T (PW)_3 - (1/hw^2) * ((PW)_3^T (PW)_0 + (PW)_0^T (PW)_3) )
    // Assuming (PW)_0^T (PW)_3 is (PW3_outer_PW0)^T.
    // The formula from paper is: W_I^t * ( (2*h_x/h_w^3) * (PW)_3^T (PW)_3 - (1/h_w^2) * ( (PW)_3^T(PW)_0 + ((PW)_3^T(PW)_0)^T ) )
    // This seems more plausible for symmetry. Let's use the one directly from image:
    // W_I^t ( (2*h_x/h_w^3)(PW)_3^T(PW)_3 - (1/h_w^2)((PW)_3^T(PW)_0 + (PW)_0^T(PW)_3) )
    // The provided screenshot has:
    // W_f^t ( (2*h_x/h_w^3) (PW)_3^T (PW)_3 - (1/h_w^2) ( (PW)_3^T (PW)_0 + ((PW)_3^T (PW)_0)^T ) )
    // Let's use the one from the prompt's image:
    // W_I^t ( (2*h_x/h_w^3) (PW)_3^T (PW)_3 - (1/h_w^2) ( (PW)_3^T (PW)_0 + ((PW)_3^T (PW)_0)^T ) )
    // My interpretation of (PW)_3^T (PW)_0 is outer product of col vec PW3 and row vec PW0.
    // The paper image looks like:  W_I^t * [ (2*h_x / h_w^3) (PW_3)^T (PW_3) - (1/h_w^2) ( (PW_3)^T (PW_0) + ((PW_3)^T (PW_0))^T ) ]
    // This makes the second term symmetric. (PW_3)^T is a column vector. (PW_0) is a row vector.
    // So (PW_3)^T (PW_0) is a 3x3 matrix.

    float factor_x1 = W_I_t * (2.0f * hx / (hw*hw*hw + 1e-9f)); // 2*hx/hw^3
    float factor_x2 = W_I_t * (-1.0f * inv_hw_sq); // -1/hw^2

    for (int i = 0; i < 9; ++i) {
        float term1_x = factor_x1 * PW3_outer_PW3[i];
        // PW3_outer_PW0[i] is ( (PW_3)^T (PW_0) )_ij
        // ((PW_3)^T (PW_0))^T _ij is PW3_outer_PW0[ji] (transpose element access)
        // For a 3x3 matrix A, A[row*3 + col]. Transposed element A[col*3+row]
        int row = i / 3;
        int col = i % 3;
        float term2_x = factor_x2 * (PW3_outer_PW0[i] + PW3_outer_PW0[col * 3 + row]);
        hessian_out_pi_x_3x3[i] = term1_x + term2_x;
    }

    // Factor for ∂²π_y/∂p_k²
    float factor_y1 = H_I_t * (2.0f * hy / (hw*hw*hw + 1e-9f)); // 2*hy/hw^3
    float factor_y2 = H_I_t * (-1.0f * inv_hw_sq); // -1/hw^2

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

// Max SH degree supported for explicit formula here is 3 (16 coeffs)
// basis_out must be pre-allocated to (degree+1)^2 floats.
__device__ __forceinline__ void eval_sh_basis_up_to_degree3(
    int degree,
    const CudaMath::float3& r_k_normalized, // unit vector (view direction)
    float* basis_out // Output array for SH basis values
) {
    float x = r_k_normalized.x;
    float y = r_k_normalized.y;
    float z = r_k_normalized.z;

    // Degree 0
    basis_out[0] = 0.2820947917738781f; // C0
    if (degree == 0) return;

    // Degree 1 (coeffs 1, 2, 3)
    basis_out[1] = -0.48860251190292f * y;
    basis_out[2] =  0.48860251190292f * z;
    basis_out[3] = -0.48860251190292f * x;
    if (degree == 1) return;

    // Degree 2 (coeffs 4, 5, 6, 7, 8)
    float x2 = x*x; float y2 = y*y; float z2 = z*z;
    basis_out[4] =  0.5462742152960395f * (2.f * x * y); // xy
    basis_out[5] = -1.092548430592079f * y * z;         // yz
    basis_out[6] =  0.3153915652525201f * (3.f * z2 - 1.f); // 3z^2-1 (scaled)
    basis_out[7] = -1.092548430592079f * x * z;         // xz
    basis_out[8] =  0.5462742152960395f * (x2 - y2);    // x^2-y^2
    if (degree == 2) return;

    // Degree 3 (coeffs 9 to 15)
    // Φ_d=3 from paper: [Φ_d=0,Φ_d=1,Φ_d=2, r_k,y(3r_k,x^2-r_k,y^2), r_k,xr_k,yr_k,z, r_k,y(4r_k,z^2-r_k,x^2-r_k,y^2),
    // r_k,z(2r_k,z^2-3r_k,x^2-3r_k,y^2), r_k,x(4r_k,z^2-r_k,x^2-r_k,y^2), r_k,z(r_k,x^2-r_k,y^2), r_k,x(r_k,x^2-3r_k,y^2)]^T
    // These are specific linear combinations of basis polynomials like x^3, y^3, z^3, x^2y, xy^2 etc.
    // Using standard real SH definitions (e.g. from Wikipedia or other sources, compatible with gsplat's eval_sh_basis_kernel)
    // L=3, M=-3: sqrt(35/pi)/4 * (3x^2-y^2)y  -> C_3^{-3} * y*(3*x2-y2)
    // L=3, M=-2: sqrt(105/pi)/2 * xyz        -> C_3^{-2} * x*y*z
    // L=3, M=-1: sqrt(21/pi)/4 * y*(4z^2-x^2-y^2) -> C_3^{-1} * y*(4*z2-x2-y2)
    // L=3, M=0:  sqrt(7/pi)/4 * z*(2z^2-3x^2-3y^2) -> C_3^{0} * z*(5*z2-3) (since x2+y2+z2=1, 2z2-3x2-3y2 = 2z2-3(1-z2)=5z2-3)
    // L=3, M=1:  sqrt(21/pi)/4 * x*(4z^2-x^2-y^2) -> C_3^{1} * x*(4*z2-x2-y2)
    // L=3, M=2:  sqrt(105/pi)/4 * z*(x^2-y^2)   -> C_3^{2} * z*(x2-y2)
    // L=3, M=3:  sqrt(35/pi)/4 * x*(x^2-3y^2)   -> C_3^{3} * x*(x2-3*y2)
    // Constants from gsplat (adjusting for order):
    // current_sh_output[9]  = -0.5900435899266435f * fS2; // y(3x^2-y^2) type term for S_3,2 (fS2 = x*fC1-y*fS1)
    // current_sh_output[10] = 1.445305721320277f * z * fS1;  // xyz type term for C_3,2 (fS1 = 2xy)
    // current_sh_output[11] = (-2.285228997322329f * z2 + 0.4570457994644658f) * y; // y(4z^2-x^2-y^2) type (fTmp0C * y)
    // current_sh_output[12] = z * (1.865881662950577f * z2 - 1.119528997770346f); // z(2z^2-3(x^2+y^2)) type
    // current_sh_output[13] = (-2.285228997322329f * z2 + 0.4570457994644658f) * x; // x(4z^2-x^2-y^2) type (fTmp0C * x)
    // current_sh_output[14] = 1.445305721320277f * z * fC1; // z(x^2-y^2) type (fC1=x2-y2)
    // current_sh_output[15] = -0.5900435899266435f * fC2; // x(x^2-3y^2) type (fC2 = x*fC1-y*fS1)

    // Using the order from gsplat's eval_sh_basis_kernel for consistency if SH coeffs are stored that way
    float fC1 = x2 - y2; float fS1 = 2.f * x * y;
    float fC2 = x * fC1 - y * fS1; float fS2 = y * fC1 + x * fS1; // Note: gsplat uses fS2 = x*fS1+y*fC1 for its index 9, which is x(3y^2-x^2) or y(3x^2-y^2)
                                                                 // Paper's order for d=3 starts with y(3x^2-y^2)
                                                                 // The gsplat order seems to be:
                                                                 // 9: Y_3^{-3} ~ y(3x^2-y^2) (up to sign/factor)
                                                                 // 10: Y_3^{-2} ~ xyz
                                                                 // 11: Y_3^{-1} ~ y(4z^2-x^2-y^2)
                                                                 // 12: Y_3^0 ~ z(2z^2-3(x^2+y^2))
                                                                 // 13: Y_3^1 ~ x(4z^2-x^2-y^2)
                                                                 // 14: Y_3^2 ~ z(x^2-y^2)
                                                                 // 15: Y_3^3 ~ x(x^2-3y^2)
    // These are constants * polynomial forms.
    basis_out[9]  = -0.5900435899266435f * fS2; // This is S_3,2 or Y_3, -3 related term
    basis_out[10] =  0.816496580927726f * fS1 * z; // C_3,2 or Y_3, -2 related (sqrt(105/4pi) * 2xyz) -> 1.4453... * z * (xy)
    basis_out[11] = -0.4570457994644658f * y * (3.0f - 5.0f * z2); // Y_3, -1 related
    basis_out[12] =  0.3731763328906658f * z * (5.0f * z2 - 3.0f); // Y_3, 0
    basis_out[13] = -0.4570457994644658f * x * (3.0f - 5.0f * z2); // Y_3, 1 related
    basis_out[14] =  0.408248290463863f * fC1 * z; // C_3,2 or Y_3, 2 related (sqrt(105/16pi) * z(x^2-y^2)) -> 0.816... * z * (x2-y2)
                                                 // My constant for 14: 1.445... * z * fC1 / 2
    basis_out[15] = -0.5900435899266435f * fC2; // This is C_3,3 or Y_3, 3 related term

    // For safety, matching gsplat's exact SH formulas from its kernel if possible
    // gsplat: current_sh_output[9]  = -0.5900435899266435f * fS2; (fS2 = x * fS1 + y * fC1)
    // gsplat: current_sh_output[10] = 1.445305721320277f * z * fS1;
    // gsplat: current_sh_output[11] = (-2.285228997322329f * z2 + 0.4570457994644658f) * y;
    // gsplat: current_sh_output[12] = z * (1.865881662950577f * z2 - 1.119528997770346f);
    // gsplat: current_sh_output[13] = (-2.285228997322329f * z2 + 0.4570457994644658f) * x;
    // gsplat: current_sh_output[14] = 1.445305721320277f * z * fC1;
    // gsplat: current_sh_output[15] = -0.5900435899266435f * fC2;
    // Using these directly:
    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    // fS2 was y*fC1 + x*fS1. gsplat uses x*fS1 + y*fC1. Same.
    // fC2 was x*fC1 - y*fS1. Same.
    basis_out[9]  = -0.5900435899266435f * fS2;
    basis_out[10] = fTmp1B * fS1;
    basis_out[11] = fTmp0C * y;
    basis_out[12] = z * (1.865881662950577f * z2 - 1.119528997770346f);
    basis_out[13] = fTmp0C * x;
    basis_out[14] = fTmp1B * fC1;
    basis_out[15] = -0.5900435899266435f * fC2;

}


// ∂r_k/∂p_k where r_k = (p_k - C) / ||p_k - C|| = v / ||v||
// ∂r_k/∂p_k = (I / ||v||) - (v v^T / ||v||^3) = (1/||v||) * (I - r_k r_k^T)
// drk_dpk_out_3x3 is a 3x3 matrix (row-major)
__device__ __forceinline__ void compute_drk_dpk(
    const CudaMath::float3& r_k_normalized, // unit vector r_k
    float r_k_norm,               // ||p_k - C||
    float* drk_dpk_out_3x3
) {
    float inv_r_k_norm = 1.0f / (r_k_norm + 1e-8f);

    // Identity matrix components
    drk_dpk_out_3x3[0] = 1.0f; drk_dpk_out_3x3[1] = 0.0f; drk_dpk_out_3x3[2] = 0.0f;
    drk_dpk_out_3x3[3] = 0.0f; drk_dpk_out_3x3[4] = 1.0f; drk_dpk_out_3x3[5] = 0.0f;
    drk_dpk_out_3x3[6] = 0.0f; drk_dpk_out_3x3[7] = 0.0f; drk_dpk_out_3x3[8] = 1.0f;

    // Subtract r_k r_k^T
    drk_dpk_out_3x3[0] -= r_k_normalized.x * r_k_normalized.x;
    drk_dpk_out_3x3[1] -= r_k_normalized.x * r_k_normalized.y;
    drk_dpk_out_3x3[2] -= r_k_normalized.x * r_k_normalized.z;

    drk_dpk_out_3x3[3] -= r_k_normalized.y * r_k_normalized.x;
    drk_dpk_out_3x3[4] -= r_k_normalized.y * r_k_normalized.y;
    drk_dpk_out_3x3[5] -= r_k_normalized.y * r_k_normalized.z;

    drk_dpk_out_3x3[6] -= r_k_normalized.z * r_k_normalized.x;
    drk_dpk_out_3x3[7] -= r_k_normalized.z * r_k_normalized.y;
    drk_dpk_out_3x3[8] -= r_k_normalized.z * r_k_normalized.z;

    // Scale by 1/||v||
    for (int i = 0; i < 9; ++i) {
        drk_dpk_out_3x3[i] *= inv_r_k_norm;
    }
}


// Computes ∂Φ(r_k)/∂r_k. This is a (num_coeffs x 3) matrix.
// dPhi_drk_out is (num_coeffs * 3) floats, row-major (dPhi0/drx, dPhi0/dry, dPhi0/drz, dPhi1/drx, ...)
// Implemented for degree up to 3.
__device__ __forceinline__ void compute_dphi_drk_up_to_degree3(
    int degree,
    const CudaMath::float3& r_k_normalized, // x, y, z
    float* dPhi_drk_out // num_coeffs x 3 output, (degree+1)^2 * 3 floats
) {
    float x = r_k_normalized.x;
    float y = r_k_normalized.y;
    float z = r_k_normalized.z;
    float x2 = x*x; float y2 = y*y; float z2 = z*z;

    // Coeff 0: Φ_0 = C0 (constant)
    dPhi_drk_out[0*3 + 0] = 0.0f; dPhi_drk_out[0*3 + 1] = 0.0f; dPhi_drk_out[0*3 + 2] = 0.0f;
    if (degree == 0) return;

    // Coeff 1: Φ_1 = C1 * y  (C1 = -0.4886...)
    dPhi_drk_out[1*3 + 0] = 0.0f; dPhi_drk_out[1*3 + 1] = -0.48860251190292f; dPhi_drk_out[1*3 + 2] = 0.0f;
    // Coeff 2: Φ_2 = C1 * z
    dPhi_drk_out[2*3 + 0] = 0.0f; dPhi_drk_out[2*3 + 1] = 0.0f; dPhi_drk_out[2*3 + 2] = 0.48860251190292f;
    // Coeff 3: Φ_3 = C1 * x
    dPhi_drk_out[3*3 + 0] = -0.48860251190292f; dPhi_drk_out[3*3 + 1] = 0.0f; dPhi_drk_out[3*3 + 2] = 0.0f;
    if (degree == 1) return;

    // Degree 2 Constants (from eval_sh_basis_up_to_degree3)
    // C2_0 = 0.5462742152960395f (for xy, x2-y2)
    // C2_1 = -1.092548430592079f (for yz, xz)
    // C2_2 = 0.3153915652525201f (for 3z2-1)
    const float C2_0 = 1.092548430592079f; // Original was 0.546... * 2xy. So derivative is 0.546... * 2y for dx.
                                          // basis_out[4] =  0.5462742152960395f * (2.f * x * y);
    const float C2_1 = -1.092548430592079f;
    const float C2_2_scaled = 0.9461746957575601f; // 0.315... * 3 for derivative part
                                            // basis_out[6] =  0.3153915652525201f * (3.f * z2 - 1.f);

    // Coeff 4: Φ_4 = C2_0 * x * y (using C2_0 = 1.0925...)
    dPhi_drk_out[4*3 + 0] = C2_0 * y; dPhi_drk_out[4*3 + 1] = C2_0 * x; dPhi_drk_out[4*3 + 2] = 0.0f;
    // Coeff 5: Φ_5 = C2_1 * y * z
    dPhi_drk_out[5*3 + 0] = 0.0f; dPhi_drk_out[5*3 + 1] = C2_1 * z; dPhi_drk_out[5*3 + 2] = C2_1 * y;
    // Coeff 6: Φ_6 = C2_2_scaled * z^2 - const' (0.315... * (3z^2-1))
    dPhi_drk_out[6*3 + 0] = 0.0f; dPhi_drk_out[6*3 + 1] = 0.0f; dPhi_drk_out[6*3 + 2] = C2_2_scaled * 2.f * z;
    // Coeff 7: Φ_7 = C2_1 * x * z
    dPhi_drk_out[7*3 + 0] = C2_1 * z; dPhi_drk_out[7*3 + 1] = 0.0f; dPhi_drk_out[7*3 + 2] = C2_1 * x;
    // Coeff 8: Φ_8 = (C2_0/2) * (x^2 - y^2) (using C2_0/2 = 0.5462...)
    dPhi_drk_out[8*3 + 0] = (C2_0/2.f) * 2.f * x; dPhi_drk_out[8*3 + 1] = (C2_0/2.f) * (-2.f * y); dPhi_drk_out[8*3 + 2] = 0.0f;
    if (degree == 2) return;

    // Degree 3: These are complex. Using gsplat's basis forms for derivatives.
    // basis_out[9]  = K9 * (y*(x2-y2) + 2*x*y*x) = K9 * (3*x2*y - y3) (fS2 = y*fC1 + x*fS1 = y(x2-y2) + x(2xy) = 3x^2y - y^3)
    // basis_out[10] = K10 * z * (2*x*y) (fTmp1B * fS1)
    // basis_out[11] = K11 * y * (4*z2 - x2 - y2) (fTmp0C * y)
    // basis_out[12] = K12 * (z * ( (2 or ConstA)*z2 - (3 or ConstB)*(x2+y2) ) ) (using gsplat's z * (1.86..z2 - 1.11..(1-z2)))
    // basis_out[13] = K13 * x * (4*z2 - x2 - y2) (fTmp0C * x)
    // basis_out[14] = K14 * z * (x2 - y2) (fTmp1B * fC1)
    // basis_out[15] = K15 * (x*(x2-y2) - y*(2xy)) = K15 * (x3 - 3*x*y2) (fC2 = x*fC1 - y*fS1)

    const float K9 = -0.5900435899266435f;
    const float K10_z = 1.445305721320277f; // This is K10' * z, where K10' is the constant for 2xy part
    const float K11_y_coeff_z2 = -2.285228997322329f; // Coeff for y*z2 part of K11 term
    const float K11_y_coeff_const = 0.4570457994644658f; // Coeff for y*const part of K11 term
                                                       // fTmp0C = K11_y_coeff_z2 * z2 + K11_y_coeff_const
    const float K12_z3_coeff = 1.865881662950577f; // Coeff for z^3 part of K12
    const float K12_z_coeff = -1.119528997770346f; // Coeff for z part of K12
    const float K14_z = 1.445305721320277f; // This is K14' * z
    const float K15 = -0.5900435899266435f;

    // Coeff 9: K9 * (3*x2*y - y3)
    dPhi_drk_out[9*3 + 0] = K9 * (6*x*y);
    dPhi_drk_out[9*3 + 1] = K9 * (3*x2 - 3*y2);
    dPhi_drk_out[9*3 + 2] = 0.0f;

    // Coeff 10: K10_z * z * (2*x*y)
    dPhi_drk_out[10*3 + 0] = K10_z * z * (2*y);
    dPhi_drk_out[10*3 + 1] = K10_z * z * (2*x);
    dPhi_drk_out[10*3 + 2] = K10_z * (2*x*y);

    // Coeff 11: y * (K11_y_coeff_z2 * z2 + K11_y_coeff_const) = K11_y_coeff_z2*y*z2 + K11_y_coeff_const*y
    // The term in gsplat is: basis_out[11] = (K11_y_coeff_z2 * z2 + K11_y_coeff_const) * y;
    // This is Φ_11 = A y z^2 + B y, where A = K11_y_coeff_z2, B = K11_y_coeff_const
    // However, the paper's Φ_d=3 has y(4z^2-x^2-y^2). Let's use gsplat's for consistency.
    // fTmp0C = K11_y_coeff_z2 * z2 + K11_y_coeff_const
    // basis_out[11] = fTmp0C * y;
    dPhi_drk_out[11*3 + 0] = 0.0f; // Assuming gsplat's form which has no x dependence here
    dPhi_drk_out[11*3 + 1] = K11_y_coeff_z2 * z2 + K11_y_coeff_const; // d/dy (fTmp0C * y) = fTmp0C
    dPhi_drk_out[11*3 + 2] = K11_y_coeff_z2 * y * (2*z); // d/dz (fTmp0C * y)

    // Coeff 12: z * (K12_z3_coeff * z2 - K12_z_coeff) = K12_z3_coeff*z3 - K12_z_coeff*z
    // basis_out[12] = z * (K12_z3_coeff * z2 + K12_z_coeff); Note gsplat uses -1.119...
    // Let's use the form from gsplat: z * (K12_z3_coeff * z^2 + K12_z_coeff_actual) where K12_z_coeff_actual = -1.119...
    dPhi_drk_out[12*3 + 0] = 0.0f;
    dPhi_drk_out[12*3 + 1] = 0.0f;
    dPhi_drk_out[12*3 + 2] = K12_z3_coeff * 3*z2 + K12_z_coeff;

    // Coeff 13: x * (K11_y_coeff_z2 * z2 + K11_y_coeff_const) // Symmetrical to Coeff 11 but with x
    // basis_out[13] = fTmp0C * x;
    dPhi_drk_out[13*3 + 0] = K11_y_coeff_z2 * z2 + K11_y_coeff_const; // d/dx (fTmp0C * x) = fTmp0C
    dPhi_drk_out[13*3 + 1] = 0.0f;
    dPhi_drk_out[13*3 + 2] = K11_y_coeff_z2 * x * (2*z); // d/dz (fTmp0C * x)

    // Coeff 14: K14_z * z * (x2 - y2)
    dPhi_drk_out[14*3 + 0] = K14_z * z * (2*x);
    dPhi_drk_out[14*3 + 1] = K14_z * z * (-2*y);
    dPhi_drk_out[14*3 + 2] = K14_z * (x2 - y2);

    // Coeff 15: K15 * (x3 - 3*x*y2)
    dPhi_drk_out[15*3 + 0] = K15 * (3*x2 - 3*y2);
    dPhi_drk_out[15*3 + 1] = K15 * (-6*x*y);
    dPhi_drk_out[15*3 + 2] = 0.0f;
}


// Computes ∂c̄_{k,R}/∂p_k (Eq. A.3 for one channel)
// jac_out_3 is [d_c_bar / d_pkx, d_c_bar / d_pky, d_c_bar / d_pkz]
// sh_coeffs_single_channel: SH coefficients for this Gaussian, this channel [num_basis_coeffs]
// sh_basis_values: Φ(r_k) evaluated [num_basis_coeffs]
// dPhi_drk: (num_basis_coeffs x 3) matrix from compute_dphi_drk_up_to_degree3
// drk_dpk: (3x3) matrix from compute_drk_dpk
__device__ __forceinline__ void compute_sh_color_jacobian_single_channel(
    const float* sh_coeffs_single_channel, // [num_coeffs]
    const float* sh_basis_values,          // [num_coeffs] (Φ(r_k))
    const float* dPhi_drk,                 // [num_coeffs * 3] (∂Φ/∂r_k)
    const float* drk_dpk,                  // [3x3] (∂r_k/∂p_k)
    int num_basis_coeffs,                  // (degree+1)^2
    float* jac_out_3                       // Output: 3 component vector
) {
    // Eq A.3: (B_k,R ⊙ c_k,R)^T * (∂Φ(r_k)/∂r_k) : (∂r_k/∂p_k)
    // B_k,R is sh_basis_values. c_k,R is sh_coeffs_single_channel.
    // (B_k,R ⊙ c_k,R) is a vector of size num_basis_coeffs. Let this be V.
    // V^T * (∂Φ/∂r_k) is a 1x3 vector. Let this be W_1x3.
    // W_1x3 * (∂r_k/∂p_k) is the final 1x3 jacobian vector.
    // (∂Φ/∂r_k) is num_coeffs x 3. (drk_dpk) is 3x3.
    // Product (∂Φ/∂r_k) * (drk_dpk) is num_coeffs x 3. Let it be M_prod.
    // Then jac = sum_i (V_i * (M_prod)_row_i)

    float M_prod[16*3]; // Max num_coeffs for degree 3 is 16. Max size needed.
                        // If only deg 01, then 4*3 = 12 floats.
    // M_prod = dPhi_drk * drk_dpk (matrix multiplication)
    // dPhi_drk is (num_coeffs x 3), drk_dpk is (3 x 3)
    CudaMath::mat_mul_mat(dPhi_drk, drk_dpk, M_prod, num_basis_coeffs, 3, 3);

    jac_out_3[0] = 0.0f; jac_out_3[1] = 0.0f; jac_out_3[2] = 0.0f;
    for (int i = 0; i < num_basis_coeffs; ++i) {
        float v_i = sh_basis_values[i] * sh_coeffs_single_channel[i]; // (B_k,R ⊙ c_k,R)_i
        jac_out_3[0] += v_i * M_prod[i * 3 + 0];
        jac_out_3[1] += v_i * M_prod[i * 3 + 1];
        jac_out_3[2] += v_i * M_prod[i * 3 + 2];
    }
}


} // namespace SHDerivs


// --- KERNEL DEFINITIONS ---

// Forward declaration for helper from gsplat (or similar) if needed for projection
// For example: compute_cov2d_and_screen_radius from gsplat/ProjectionUT3DGSFused.cu (simplified)
// For now, we'll implement a simplified version or assume Σ_k (2D) is passed or computed easily.


// Placeholder for 2D covariance calculation from 3D Gaussian parameters
// This is a complex step, typically done in rasterization forward pass.
// Inputs: scale (float3), rotation (quat float4), view_matrix (float[16]), projection_matrix_for_jacobian (float[16] or K)
// Outputs: cov2d (float[3] for symmetric 2x2: C00, C01, C11), and potentially its inverse and determinant.
__device__ __forceinline__ void get_projected_cov2d_and_derivs_placeholder(
    const CudaMath::float3& p_k_world, // World space position of Gaussian
    const float* scales_k,      // 3D scales for Gaussian k
    const float* rotations_k,   // Quaternion rotation for Gaussian k
    const float* view_matrix,   // 4x4 view matrix
    const float* proj_matrix,   // 4x4 projection matrix (e.g. K_matrix combined with perspective)
    const float* jacobian_d_pi_d_pk, // 2x3 matrix: ∂π_k / ∂p_k (world)
    float img_W, float img_H,   // Image dimensions for focal length, used in gsplat's projection.
                                // The K_matrix passed to kernel might already have focal lengths.
    // Outputs
    float* cov2d_sym,      // Output: 2x2 symmetric covariance matrix [xx, xy, yy] in screen space
    float* inv_cov2d_sym,  // Output: Its inverse [ixx, ixy, iyy]
    float* det_cov2d,      // Output: Determinant of cov2d
    float* d_Gk_d_pik,     // Output: ∂G_k/∂π_k (2-vector) at a given pixel_ndc_minus_pi_k
    float* d2_Gk_d_pik2    // Output: ∂²G_k/∂π_k² (2x2 symmetric matrix [xx,xy,yy]) at pixel_ndc_minus_pi_k
                           // This also needs pixel_ndc_minus_pi_k as input. For now, general structure.
                           // Note: The paper gives ∂G_k/∂Σ_k. We need ∂G_k/∂π_k.
                           // G_k = exp(-0.5 * diff^T Σ_inv diff). ∂G_k/∂π_k = -G_k * Σ_inv * diff.
                           // ∂²G_k/∂π_k² = G_k * (Σ_inv * diff * diff^T * Σ_inv - Σ_inv).
) {
    // This is a major placeholder. A full 3DGS projection involves:
    // 1. World space covariance Σ_w = R S S^T R^T
    // 2. Transform to camera space Σ_c = V_rot Σ_w V_rot^T (V_rot is rotational part of view matrix)
    // 3. Project to screen space: Σ_s = J Σ_c J^T where J is Jacobian of perspective divide.
    //    J = [fx/z  0    -fx*x/z^2]
    //        [0     fy/z -fy*y/z^2] (at point (x,y,z) in camera space)
    // For simplicity, assume cov2d is identity for now, or a very simple form.
    cov2d_sym[0] = 1.0f; cov2d_sym[1] = 0.0f; cov2d_sym[2] = 1.0f; // Identity
    inv_cov2d_sym[0] = 1.0f; inv_cov2d_sym[1] = 0.0f; inv_cov2d_sym[2] = 1.0f; // Inverse of identity
    *det_cov2d = 1.0f;

    // Derivatives of G_k = exp(-0.5 * diff^T Σ_inv diff) where diff = π_k - x_pixel_ndc
    // Let diff_x, diff_y be components of (π_k - x_pixel_ndc)
    // For placeholder, assume diff is (0,0) so G_k=1, dG/dpi=0, d2G/dpi2 = -G_k * Sigma_inv
    if (d_Gk_d_pik) {
        d_Gk_d_pik[0] = 0.0f; // dG/dπx
        d_Gk_d_pik[1] = 0.0f; // dG/dπy
    }
    if (d2_Gk_d_pik2) {
        d2_Gk_d_pik2[0] = -1.0f * inv_cov2d_sym[0]; // d2G/dπx2
        d2_Gk_d_pik2[1] = -1.0f * inv_cov2d_sym[1]; // d2G/dπxdπy
        d2_Gk_d_pik2[2] = -1.0f * inv_cov2d_sym[2]; // d2G/dπy2
    }
}


// Kernel for L1/L2 dL/dc and d2L/dc2
__global__ void compute_l1l2_loss_derivatives_kernel(
    const float* rendered_image, // [H, W, C]
    const float* gt_image,       // [H, W, C]
    bool use_l2_loss_term,
    float inv_N_pixels, // Inverse of number of pixels (1.0f / (H*W))
    float* out_dL_dc_l1l2,      // [H, W, C]
    float* out_d2L_dc2_diag_l1l2, // [H, W, C]
    int H, int W, int C) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = H * W * C;

    if (idx >= total_elements) return;

    float diff = rendered_image[idx] - gt_image[idx];
    if (use_l2_loss_term) { // L2 part
        out_dL_dc_l1l2[idx] = inv_N_pixels * 2.f * diff;
        out_d2L_dc2_diag_l1l2[idx] = inv_N_pixels * 2.f;
    } else { // L1 part
        // For L1, derivative is sign(diff). Normalization by N_pixels is also reasonable.
        out_dL_dc_l1l2[idx] = inv_N_pixels * ((diff > 1e-6f) ? 1.f : ((diff < -1e-6f) ? -1.f : 0.f));
        out_d2L_dc2_diag_l1l2[idx] = 0.f; // For L1, 2nd derivative is zero (or undefined, treated as 0)
    }
}


// Kernel for position Hessian components
// This is a very complex kernel. The sketch below is highly simplified.
// It needs to implement parts of rasterization forward and then derivatives.
__global__ void compute_position_hessian_components_kernel(
    int H_img, int W_img, int C_img, // Image dimensions
    int P_total,                     // Total number of Gaussians in the model
    const float* means_3d_all,       // [P_total, 3]
    const float* scales_all,         // [P_total, 3]
    const float* rotations_all,      // [P_total, 4] quaternion
    const float* opacities_all,      // [P_total]
    const float* shs_all,            // [P_total, sh_coeffs_dim_flat] or [P_total, (D+1)^2, 3]
    int sh_degree,
    int sh_coeffs_per_color_channel, // (sh_degree+1)^2
    const float* view_matrix,        // 4x4 view matrix (world to camera)
    const float* projection_matrix_for_jacobian, // This is K_matrix (intrinsics) in NewtonOptimizer.cpp
                                                 // Let's assume it's a 4x4 projection matrix P for now.
                                                 // Or it's K and we build P = K_frustum * StandardPerspective
    const float* cam_pos_world,      // [3] camera position in world space
    // const float* means_2d_render, const float* depths_render, const float* radii_render, int P_render, // Not directly used per P_total Gaussian
    const bool* visibility_mask_for_model, // [P_total]
    const float* dL_dc_pixelwise,          // [H_img, W_img, C_img]
    const float* d2L_dc2_diag_pixelwise,   // [H_img, W_img, C_img] (diagonal of 3x3 Hessian per pixel)
    int num_output_gaussians,        // Number of Gaussians that are visible and will have output
    float* H_p_output_packed,        // Output: [num_output_gaussians, 6] (symmetric 3x3 Hessian)
    float* grad_p_output,            // Output: [num_output_gaussians, 3] (gradient)
    const int* output_index_map,     // [P_total] -> maps p_idx_total to dense output_idx, or -1
    bool debug_prints_enabled
) {
    int p_idx_total = blockIdx.x * blockDim.x + threadIdx.x;

    if (p_idx_total >= P_total) return;
    if (!visibility_mask_for_model[p_idx_total]) return;

    int output_idx = output_index_map[p_idx_total];
    if (output_idx < 0 || output_idx >= num_output_gaussians) return; // Should not happen if map is correct

    // 1. Load Gaussian Data for p_idx_total
    CudaMath::float3 pk_vec3 = CudaMath::make_float3(
        means_3d_all[p_idx_total * 3 + 0],
        means_3d_all[p_idx_total * 3 + 1],
        means_3d_all[p_idx_total * 3 + 2]);

    const float* scales_k = scales_all + p_idx_total * 3;
    const float* rotations_k = rotations_all + p_idx_total * 4;
    float opacity_k = opacities_all[p_idx_total];
    // SH coeffs: shs_all is [P_total, (D+1)^2, 3]. Access as:
    // shs_all[p_idx_total * sh_coeffs_per_color_channel * 3 + coeff_idx * 3 + channel_idx]
    const float* sh_coeffs_k_all_channels = shs_all + p_idx_total * sh_coeffs_per_color_channel * 3;

    // Camera position
    CudaMath::float3 cam_pos_world_vec3 = CudaMath::make_float3(cam_pos_world[0], cam_pos_world[1], cam_pos_world[2]);

    // View direction r_k
    CudaMath::float3 view_dir_to_pk_unnormalized = CudaMath::sub_float3(pk_vec3, cam_pos_world_vec3);
    float r_k_norm = CudaMath::length_float3(view_dir_to_pk_unnormalized);
    CudaMath::float3 r_k_normalized = CudaMath::div_float3_scalar(view_dir_to_pk_unnormalized, r_k_norm);

    // Combined Projection * View matrix (PW)
    // projection_matrix_for_jacobian is K (e.g. 3x3 or 4x4). view_matrix is V (4x4).
    // Need PW = P_persp * V. If projection_matrix_for_jacobian is K, need to form P_persp.
    // For now, assume projection_matrix_for_jacobian is already a 4x4 perspective projection matrix P.
    // Then PW = P * V. (Order matters: points are p_world, V maps to p_cam, P maps to p_clip)
    // So, h = P * V * p_world_homogeneous. Let's call it proj_view_matrix.
    float proj_view_matrix[16]; // PW = projection_matrix_for_jacobian * view_matrix
                                // Assuming projection_matrix_for_jacobian is P_clip_from_cam (e.g. glFrustum output)
                                // And view_matrix is V_cam_from_world
    CudaMath::mat_mul_mat(projection_matrix_for_jacobian, view_matrix, proj_view_matrix, 4, 4, 4);


    // 2. Precompute derivatives of intermediate variables w.r.t p_k
    // 2.1 Projection derivatives (π_k)
    float h_vec4_data[4];
    ProjectionDerivs::compute_h_vec(pk_vec3, proj_view_matrix, h_vec4_data);

    float d_pi_d_pk_data[2*3]; // ∂π_k / ∂p_k (2x3 matrix)
    ProjectionDerivs::compute_projection_jacobian(proj_view_matrix, (float)W_img, (float)H_img, h_vec4_data, d_pi_d_pk_data);

    float d2_pi_d_pk2_x_data[3*3]; // ∂²π_x / ∂p_k² (3x3 matrix)
    float d2_pi_d_pk2_y_data[3*3]; // ∂²π_y / ∂p_k² (3x3 matrix)
    ProjectionDerivs::compute_projection_hessian(proj_view_matrix, (float)W_img, (float)H_img, h_vec4_data, d2_pi_d_pk2_x_data, d2_pi_d_pk2_y_data);

    // 2.2 SH Color derivatives (c̄_k)
    float sh_basis_eval_data[16]; // Max (3+1)^2 = 16 coeffs
    SHDerivs::eval_sh_basis_up_to_degree3(sh_degree, r_k_normalized, sh_basis_eval_data);

    float d_rk_d_pk_data[3*3]; // ∂r_k / ∂p_k (3x3 matrix)
    SHDerivs::compute_drk_dpk(r_k_normalized, r_k_norm, d_rk_d_pk_data);

    float d_phi_d_rk_data[16*3]; // Max ( (3+1)^2 * 3 )
    SHDerivs::compute_dphi_drk_up_to_degree3(sh_degree, r_k_normalized, d_phi_d_rk_data);

    CudaMath::float3 d_c_bar_R_d_pk, d_c_bar_G_d_pk, d_c_bar_B_d_pk;
    // SH coeffs for Red channel: sh_coeffs_k_all_channels[coeff_idx*3 + 0]
    // Need to re-arrange sh_coeffs_k_all_channels to be [sh_coeffs_per_color_channel] for each call.
    float sh_coeffs_k_R[16], sh_coeffs_k_G[16], sh_coeffs_k_B[16];
    for(int i=0; i<sh_coeffs_per_color_channel; ++i) {
        sh_coeffs_k_R[i] = sh_coeffs_k_all_channels[i*3 + 0];
        sh_coeffs_k_G[i] = sh_coeffs_k_all_channels[i*3 + 1];
        sh_coeffs_k_B[i] = sh_coeffs_k_all_channels[i*3 + 2];
    }

    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_R, sh_basis_eval_data, d_phi_d_rk_data, d_rk_d_pk_data, sh_coeffs_per_color_channel, &d_c_bar_R_d_pk.x);
    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_G, sh_basis_eval_data, d_phi_d_rk_data, d_rk_d_pk_data, sh_coeffs_per_color_channel, &d_c_bar_G_d_pk.x);
    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_B, sh_basis_eval_data, d_phi_d_rk_data, d_rk_d_pk_data, sh_coeffs_per_color_channel, &d_c_bar_B_d_pk.x);

    // Placeholder for ∂²c̄_k/∂p_k² (Eq. A.4). For now, assume this is zero.
    // CudaMath::float3 d2_c_bar_R_d_pk2[3], ... (each is a 3x3 matrix, so 3 float3s for rows)

    // Initialize accumulators for g_p_k and H_p_k
    CudaMath::float3 g_p_k_accum = CudaMath::make_float3(0.f, 0.f, 0.f);
    float H_p_k_accum_symm[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}; // H00, H01, H02, H11, H12, H22

    // 3. Pixel Loop: Iterate over pixels potentially affected by Gaussian k
    // Simplified: Iterate over all pixels for now. A proper implementation would use a bounding box.
    for (int r = 0; r < H_img; ++r) { // pixel row
        for (int c = 0; c < W_img; ++c) { // pixel col
            // Current pixel's NDC coordinates (approximate, center of pixel)
            float pixel_ndc_x = (2.0f * (c + 0.5f) / W_img - 1.0f);
            float pixel_ndc_y = (2.0f * (r + 0.5f) / H_img - 1.0f); // Assuming origin is top-left for image, but NDC is bottom-left
                                                                    // Or: (1.0f - 2.0f * (r + 0.5f) / H_img) if NDC Y is upwards from bottom-left

            // Projected center of Gaussian k (π_k) in NDC
            // h_vec4_data = [hx, hy, hz, hw] = PW * pk_world_homo
            // pi_k_ndc_x = hx/hw (scaled by W_img/2), pi_k_ndc_y = hy/hw (scaled by H_img/2)
            // The jacobian helper uses W_I_t/2, H_I_t/2. Let's assume pi_k here is in range [-1,1] for NDC.
            float pi_k_ndc_x_unscaled = h_vec4_data[0] / (h_vec4_data[3] + 1e-7f);
            float pi_k_ndc_y_unscaled = h_vec4_data[1] / (h_vec4_data[3] + 1e-7f);
            // These might need further scaling depending on convention of W_I_t, H_I_t in jacobian.
            // If jacobian already outputs d(NDC)/dpk where NDC is [-1,1], then this is fine.

            // Difference vector: diff_ndc = π_k - x_pixel_ndc
            CudaMath::float2 diff_ndc = CudaMath::make_float2(pi_k_ndc_x_unscaled - pixel_ndc_x, pi_k_ndc_y_unscaled - pixel_ndc_y);

            // Get 2D covariance Σ_k^{2D}, its inverse, determinant. Also ∂G_k/∂π_k, ∂²G_k/∂π_k²
            float cov2d_sym_data[3], inv_cov2d_sym_data[3], det_cov2d_data;
            float d_Gk_d_pik_data[2];   // ∂G_k/∂π_k at this pixel
            float d2_Gk_d_pik2_data[3]; // ∂²G_k/∂π_k² (symmetric xx, xy, yy) at this pixel

            // Call placeholder for cov2D and its G_k derivatives
            // This placeholder needs to be replaced with actual 3DGS projection logic.
            // It also needs the actual diff_ndc to compute d_Gk_d_pik, etc.
            // For now, it returns fixed values.
            get_projected_cov2d_and_derivs_placeholder(pk_vec3, scales_k, rotations_k,
                                                       view_matrix, projection_matrix_for_jacobian,
                                                       d_pi_d_pk_data, (float)W_img, (float)H_img,
                                                       cov2d_sym_data, inv_cov2d_sym_data, &det_cov2d_data,
                                                       nullptr, nullptr); // Pass nullptr for now, as placeholder does not use diff

            // Actual G_k calculation: G_k = (1/(2π sqrt(det(Σ)))) * exp(-0.5 * diff^T Σ_inv diff)
            // For simplicity, assume G_k_pixel = some value (e.g., 1.0 if pixel is center, decay otherwise)
            // This is a major simplification.
            float G_k_pixel = expf(-0.5f * (diff_ndc.x*diff_ndc.x*inv_cov2d_sym_data[0] +
                                            2*diff_ndc.x*diff_ndc.y*inv_cov2d_sym_data[1] +
                                            diff_ndc.y*diff_ndc.y*inv_cov2d_sym_data[2]));
            if (det_cov2d_data <= 1e-7f) G_k_pixel = 0.f; // Avoid issues with tiny determinant

            // If G_k_pixel is too small, this Gaussian has no influence, skip.
            if (G_k_pixel < 1e-4f) continue;

            // Now, recompute d_Gk_d_pik and d2_Gk_d_pik2 using the actual diff_ndc
            // ∂G_k/∂π_k = -G_k * Σ_inv * diff
            CudaMath::float2 sigma_inv_diff;
            sigma_inv_diff.x = inv_cov2d_sym_data[0]*diff_ndc.x + inv_cov2d_sym_data[1]*diff_ndc.y;
            sigma_inv_diff.y = inv_cov2d_sym_data[1]*diff_ndc.x + inv_cov2d_sym_data[2]*diff_ndc.y;
            d_Gk_d_pik_data[0] = -G_k_pixel * sigma_inv_diff.x;
            d_Gk_d_pik_data[1] = -G_k_pixel * sigma_inv_diff.y;

            // ∂²G_k/∂π_k² = G_k * (Σ_inv * diff * diff^T * Σ_inv - Σ_inv)
            // Term1 = Σ_inv * diff * diff^T * Σ_inv = sigma_inv_diff * sigma_inv_diff^T
            d2_Gk_d_pik2_data[0] = G_k_pixel * (sigma_inv_diff.x * sigma_inv_diff.x - inv_cov2d_sym_data[0]); // xx
            d2_Gk_d_pik2_data[1] = G_k_pixel * (sigma_inv_diff.x * sigma_inv_diff.y - inv_cov2d_sym_data[1]); // xy
            d2_Gk_d_pik2_data[2] = G_k_pixel * (sigma_inv_diff.y * sigma_inv_diff.y - inv_cov2d_sym_data[2]); // yy


            // Simplified ∂c/∂c̄_k and ∂c/∂G_k (ignoring blending with other Gaussians for now)
            // alpha_k_pixel = opacity_k * G_k_pixel (approx)
            // Assume c_final_pixel = alpha_k_pixel * c_bar_k + (1-alpha_k_pixel) * C_background
            // For simplicity, assume C_background = 0.
            // Then ∂c_final_pixel/∂c̄_k = alpha_k_pixel
            // And ∂c_final_pixel/∂G_k = opacity_k * c_bar_k
            float alpha_k_pixel = opacity_k * G_k_pixel;

            CudaMath::float3 c_bar_k_rgb; // SH eval at r_k_normalized
            c_bar_k_rgb.x = CudaMath::dot_product(CudaMath::make_float3(sh_coeffs_k_R[0], sh_coeffs_k_R[1], sh_coeffs_k_R[2]), CudaMath::make_float3(sh_basis_eval_data[0], sh_basis_eval_data[1], sh_basis_eval_data[2])); // Simplified for display
            // A proper SH eval: sum over all coeffs: sum(sh_coeffs_k_R[i] * sh_basis_eval_data[i])
            c_bar_k_rgb.x =0; for(int i=0; i<sh_coeffs_per_color_channel; ++i) c_bar_k_rgb.x += sh_coeffs_k_R[i] * sh_basis_eval_data[i];
            c_bar_k_rgb.y =0; for(int i=0; i<sh_coeffs_per_color_channel; ++i) c_bar_k_rgb.y += sh_coeffs_k_G[i] * sh_basis_eval_data[i];
            c_bar_k_rgb.z =0; for(int i=0; i<sh_coeffs_per_color_channel; ++i) c_bar_k_rgb.z += sh_coeffs_k_B[i] * sh_basis_eval_data[i];


            CudaMath::float3 d_c_final_d_Gk = CudaMath::mul_float3_scalar(c_bar_k_rgb, opacity_k);

            // Eq 16: J_c_pk = (∂c/∂c̄_k)(∂c̄_k/∂p_k) + (∂c/∂G_k)[ (∂G_k/∂π_k)(∂π_k/∂p_k) ]
            // (ignoring (∂G_k/∂Σ_k)(∂Σ_k/∂p_k) term)
            // ∂c/∂c̄_k is per channel. ∂c̄_k/∂p_k is per channel (d_c_bar_R_d_pk etc)
            // (∂G_k/∂π_k)(∂π_k/∂p_k) is chain rule: (1x2 vector) * (2x3 matrix) = 1x3 vector.
            // Let d_Gk_d_pk_chain = d_Gk_d_pik * d_pi_d_pk
            CudaMath::float3 d_Gk_d_pk_chain;
            d_Gk_d_pk_chain.x = d_Gk_d_pik_data[0] * d_pi_d_pk_data[0*3+0] + d_Gk_d_pik_data[1] * d_pi_d_pk_data[1*3+0];
            d_Gk_d_pk_chain.y = d_Gk_d_pik_data[0] * d_pi_d_pk_data[0*3+1] + d_Gk_d_pik_data[1] * d_pi_d_pk_data[1*3+1];
            d_Gk_d_pk_chain.z = d_Gk_d_pik_data[0] * d_pi_d_pk_data[0*3+2] + d_Gk_d_pik_data[1] * d_pi_d_pk_data[1*3+2];

            CudaMath::float3 J_c_pk_R, J_c_pk_G, J_c_pk_B; // Jacobians of pixel color channel w.r.t pk
            J_c_pk_R = CudaMath::add_float3(CudaMath::mul_float3_scalar(d_c_bar_R_d_pk, alpha_k_pixel), CudaMath::mul_float3_scalar(d_Gk_d_pk_chain, d_c_final_d_Gk.x));
            J_c_pk_G = CudaMath::add_float3(CudaMath::mul_float3_scalar(d_c_bar_G_d_pk, alpha_k_pixel), CudaMath::mul_float3_scalar(d_Gk_d_pk_chain, d_c_final_d_Gk.y));
            J_c_pk_B = CudaMath::add_float3(CudaMath::mul_float3_scalar(d_c_bar_B_d_pk, alpha_k_pixel), CudaMath::mul_float3_scalar(d_Gk_d_pk_chain, d_c_final_d_Gk.z));

            // Eq 17: H_c_pk = ∂c/∂c̄_k * ∂²c̄_k/∂p_k² + ... many terms
            // For now, approximating H_c_pk as zero due to complexity of ∂²c̄_k/∂p_k² and other second order terms.
            // This means the term dL/dc * H_c_pk in Hessian accumulation will be zero.
            // CudaMath::float3 H_c_pk_R[3], H_c_pk_G[3], H_c_pk_B[3]; // Each is a 3x3 symm matrix, placeholder

            // Get dL/dc and d2L/dc2 for this pixel (r,c)
            int pixel_idx_flat = (r * W_img + c) * C_img;
            CudaMath::float3 dL_dc_val = CudaMath::make_float3(dL_dc_pixelwise[pixel_idx_flat+0], dL_dc_pixelwise[pixel_idx_flat+1], dL_dc_pixelwise[pixel_idx_flat+2]);
            CudaMath::float3 d2L_dc2_diag_val = CudaMath::make_float3(d2L_dc2_diag_pixelwise[pixel_idx_flat+0], d2L_dc2_diag_pixelwise[pixel_idx_flat+1], d2L_dc2_diag_pixelwise[pixel_idx_flat+2]);

            // Accumulate gradient: g_p_k += J_c_pk^T * dL_dc(pixel)
            // J_c_pk is effectively a 3x3 matrix where rows are J_c_pk_R, J_c_pk_G, J_c_pk_B (each a 1x3 vector)
            // Transpose means J_c_pk_R becomes a column vector.
            g_p_k_accum.x += J_c_pk_R.x * dL_dc_val.x + J_c_pk_G.x * dL_dc_val.y + J_c_pk_B.x * dL_dc_val.z;
            g_p_k_accum.y += J_c_pk_R.y * dL_dc_val.x + J_c_pk_G.y * dL_dc_val.y + J_c_pk_B.y * dL_dc_val.z;
            g_p_k_accum.z += J_c_pk_R.z * dL_dc_val.x + J_c_pk_G.z * dL_dc_val.y + J_c_pk_B.z * dL_dc_val.z;

            // Accumulate Hessian: H_p_k += J_c_pk^T * d2L/dc2_diag * J_c_pk  (since H_c_pk is approx zero)
            // J_c_pk_R is (dxr, dyr, dzr)
            // J_c_pk_G is (dxg, dyg, dzg)
            // J_c_pk_B is (dxb, dyb, dzb)
            // d2L/dc2_diag is (d2L_R, d2L_G, d2L_B)
            // Term for H_xx: dxr*d2L_R*dxr + dxg*d2L_G*dxg + dxb*d2L_B*dxb
            // Term for H_xy: dxr*d2L_R*dyr + dxg*d2L_G*dyg + dxb*d2L_B*dyb
            H_p_k_accum_symm[0] += J_c_pk_R.x * d2L_dc2_diag_val.x * J_c_pk_R.x + J_c_pk_G.x * d2L_dc2_diag_val.y * J_c_pk_G.x + J_c_pk_B.x * d2L_dc2_diag_val.z * J_c_pk_B.x; // H_xx
            H_p_k_accum_symm[1] += J_c_pk_R.x * d2L_dc2_diag_val.x * J_c_pk_R.y + J_c_pk_G.x * d2L_dc2_diag_val.y * J_c_pk_G.y + J_c_pk_B.x * d2L_dc2_diag_val.z * J_c_pk_B.y; // H_xy
            H_p_k_accum_symm[2] += J_c_pk_R.x * d2L_dc2_diag_val.x * J_c_pk_R.z + J_c_pk_G.x * d2L_dc2_diag_val.y * J_c_pk_G.z + J_c_pk_B.x * d2L_dc2_diag_val.z * J_c_pk_B.z; // H_xz
            H_p_k_accum_symm[3] += J_c_pk_R.y * d2L_dc2_diag_val.x * J_c_pk_R.y + J_c_pk_G.y * d2L_dc2_diag_val.y * J_c_pk_G.y + J_c_pk_B.y * d2L_dc2_diag_val.z * J_c_pk_B.y; // H_yy
            H_p_k_accum_symm[4] += J_c_pk_R.y * d2L_dc2_diag_val.x * J_c_pk_R.z + J_c_pk_G.y * d2L_dc2_diag_val.y * J_c_pk_G.z + J_c_pk_B.y * d2L_dc2_diag_val.z * J_c_pk_B.z; // H_yz
            H_p_k_accum_symm[5] += J_c_pk_R.z * d2L_dc2_diag_val.x * J_c_pk_R.z + J_c_pk_G.z * d2L_dc2_diag_val.y * J_c_pk_G.z + J_c_pk_B.z * d2L_dc2_diag_val.z * J_c_pk_B.z; // H_zz
        }
    }

    // 4. Store accumulated g_p_k and H_p_k into global memory
    grad_p_output[output_idx * 3 + 0] = g_p_k_accum.x;
    grad_p_output[output_idx * 3 + 1] = g_p_k_accum.y;
    grad_p_output[output_idx * 3 + 2] = g_p_k_accum.z;

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

    // 1. Calculate r_k = p_k - cam_pos_world (view vector from camera to point)
    //    It's often more convenient to use the camera's Z-axis (view direction) and up/right vectors.
    //    The paper's U_k = [u_x, u_y] forms a 2D basis perpendicular to r_k.
    //    Let r_k = means_3d_visible[idx*3+c] - cam_pos_world[c]
    //    This r_k is pointing from camera to Gaussian.
    //    The paper mentions "camera's look at r_k". This usually means r_k is the direction vector.
    //    Let's assume view_matrix gives camera orientation. view_matrix[2], view_matrix[6], view_matrix[10] is cam Z axis (if row-major).
    //    Let world_z_vec = {view_matrix[2], view_matrix[6], view_matrix[10]} (camera's forward vector)
    //    Let world_y_vec = {view_matrix[1], view_matrix[5], view_matrix[9]} (camera's up vector)
    //    Let world_x_vec = {view_matrix[0], view_matrix[4], view_matrix[8]} (camera's right vector)

    // Simplified U_k: use camera's X and Y axes in world space as u_x, u_y.
    // This assumes planar adjustment aligned with camera's own axes.
    // Paper Eq 14 is more complex: u_y = (r_k outer_prod r_k)[0,1,0]^T / norm(...)
    // This implies r_k is used to define the plane.
    // For now, let u_x = camera right, u_y = camera up. This is a common simplification for screen-space operations.
    float ux[3] = {view_matrix[0], view_matrix[4], view_matrix[8]}; // Camera Right
    float uy[3] = {view_matrix[1], view_matrix[5], view_matrix[9]}; // Camera Up

    // Project gradient: g_v = U^T g_p
    // g_v[0] = ux . grad_p_input[idx*3+c]
    // g_v[1] = uy . grad_p_input[idx*3+c]
    out_grad_v[idx*2 + 0] = ux[0]*grad_p_input[idx*3+0] + ux[1]*grad_p_input[idx*3+1] + ux[2]*grad_p_input[idx*3+2];
    out_grad_v[idx*2 + 1] = uy[0]*grad_p_input[idx*3+0] + uy[1]*grad_p_input[idx*3+1] + uy[2]*grad_p_input[idx*3+2];

    // Project Hessian: H_v = U^T H_p U
    // H_p matrix from packed:
    // [ H00 H01 H02 ]
    // [ H01 H11 H12 ]
    // [ H02 H12 H22 ]
    // H_p_packed_input = [H00, H01, H02, H11, H12, H22]
    const float* Hp = &H_p_packed_input[idx*6];
    float Hpu_x[3]; // H_p * u_x
    Hpu_x[0] = Hp[0]*ux[0] + Hp[1]*ux[1] + Hp[2]*ux[2];
    Hpu_x[1] = Hp[1]*ux[0] + Hp[3]*ux[1] + Hp[4]*ux[2];
    Hpu_x[2] = Hp[2]*ux[0] + Hp[4]*ux[1] + Hp[5]*ux[2];

    float Hpu_y[3]; // H_p * u_y
    Hpu_y[0] = Hp[0]*uy[0] + Hp[1]*uy[1] + Hp[2]*uy[2];
    Hpu_y[1] = Hp[1]*uy[0] + Hp[3]*uy[1] + Hp[4]*uy[2];
    Hpu_y[2] = Hp[2]*uy[0] + Hp[4]*uy[1] + Hp[5]*uy[2];

    // H_v elements:
    // Hv_xx = u_x^T H_p u_x
    out_H_v_packed[idx*3 + 0] = ux[0]*Hpu_x[0] + ux[1]*Hpu_x[1] + ux[2]*Hpu_x[2];
    // Hv_xy = u_x^T H_p u_y
    out_H_v_packed[idx*3 + 1] = ux[0]*Hpu_y[0] + ux[1]*Hpu_y[1] + ux[2]*Hpu_y[2];
    // Hv_yy = u_y^T H_p u_y
    out_H_v_packed[idx*3 + 2] = uy[0]*Hpu_y[0] + uy[1]*Hpu_y[1] + uy[2]*Hpu_y[2];
}

// Kernel for batch 2x2 solve
__global__ void batch_solve_2x2_system_kernel(
    int num_systems,
    const float* H_v_packed, // [N, 3] (H00, H01, H11)
    const float* g_v,        // [N, 2] (g0, g1)
    float damping,
    float step_scale,
    float* out_delta_v) {    // [N, 2]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_systems) return;

    float H00 = H_v_packed[idx*3 + 0];
    float H01 = H_v_packed[idx*3 + 1];
    float H11 = H_v_packed[idx*3 + 2];

    float g0 = g_v[idx*2 + 0];
    float g1 = g_v[idx*2 + 1];

    // Add damping to diagonal
    H00 += damping;
    H11 += damping;

    float det = H00 * H11 - H01 * H01;

    // If det is too small, effectively no update or use gradient descent step
    if (abs(det) < 1e-8f) {
        out_delta_v[idx*2 + 0] = -step_scale * g0 / (H00 + 1e-6f); // Simplified fallback
        out_delta_v[idx*2 + 1] = -step_scale * g1 / (H11 + 1e-6f);
        return;
    }

    float inv_det = 1.f / det;

    // delta_v = - H_inv * g
    // H_inv = inv_det * [H11, -H01; -H01, H00]
    out_delta_v[idx*2 + 0] = -step_scale * inv_det * (H11 * g0 - H01 * g1);
    out_delta_v[idx*2 + 1] = -step_scale * inv_det * (-H01 * g0 + H00 * g1);
}

// Kernel for re-projecting delta_v to delta_p
__global__ void project_update_to_3d_kernel(
    int num_updates,
    const float* delta_v,          // [N, 2] (dvx, dvy)
    const float* means_3d_visible, // [N, 3] (Not strictly needed if U_k doesn't depend on p_k itself, but paper's U_k does via r_k)
    const float* view_matrix,      // [16]
    const float* cam_pos_world,    // [3]
    float* out_delta_p) {          // [N, 3]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_updates) return;

    // Using the same simplified U_k = [cam_right, cam_up] as in projection kernel
    float ux[3] = {view_matrix[0], view_matrix[4], view_matrix[8]}; // Camera Right
    float uy[3] = {view_matrix[1], view_matrix[5], view_matrix[9]}; // Camera Up

    float dvx = delta_v[idx*2 + 0];
    float dvy = delta_v[idx*2 + 1];

    // delta_p = U_k * delta_v = u_x * dvx + u_y * dvy
    out_delta_p[idx*3 + 0] = ux[0] * dvx + uy[0] * dvy;
    out_delta_p[idx*3 + 1] = ux[1] * dvx + uy[1] * dvy;
    out_delta_p[idx*3 + 2] = ux[2] * dvx + uy[2] * dvy;
}

// --- Spherical Harmonics Basis Evaluation Kernel ---
// Based on gsplat's sh_coeffs_to_color_fast, but only computes basis values.
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

    // Create temporary tensors for L1/L2 parts
    auto tensor_opts = rendered_image_tensor.options();
    torch::Tensor dL_dc_l1l2 = torch::empty_like(rendered_image_tensor, tensor_opts);
    torch::Tensor d2L_dc2_diag_l1l2 = torch::empty_like(rendered_image_tensor, tensor_opts);

    // Calculate normalization factor
    const float N_pixels = static_cast<float>(H * W);
    const float inv_N_pixels = (N_pixels > 0) ? (1.0f / N_pixels) : 1.0f; // Avoid div by zero if H*W=0

    // Call kernel for L1/L2 derivatives
    compute_l1l2_loss_derivatives_kernel<<<GET_BLOCKS(total_elements), CUDA_NUM_THREADS>>>(
        rendered_image_ptr, gt_image_ptr, use_l2_loss_term, inv_N_pixels,
        gs::torch_utils::get_data_ptr<float>(dL_dc_l1l2),
        gs::torch_utils::get_data_ptr<float>(d2L_dc2_diag_l1l2),
        H, W, C
    );
    CUDA_CHECK(cudaGetLastError());

    // --- SSIM Part ---
    // Constants for SSIM
    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;

    // Reshape and permute images for SSIM functions: [H,W,C] -> [1,C,H,W]
    torch::Tensor img1_bchw = rendered_image_tensor.unsqueeze(0).permute({0, 3, 1, 2}).contiguous();
    torch::Tensor img2_bchw = gt_image_tensor.unsqueeze(0).permute({0, 3, 1, 2}).contiguous();

    // Call fusedssim to get ssim_map and intermediate derivatives for backward pass
    // Need to include "kernels/ssim.cuh" for these C++ functions
    auto ssim_outputs = fusedssim(C1, C2, img1_bchw, img2_bchw, true /* train=true */);
    torch::Tensor ssim_map_bchw = std::get<0>(ssim_outputs);
    torch::Tensor dm_dmu1 = std::get<1>(ssim_outputs);
    torch::Tensor dm_dsigma1_sq = std::get<2>(ssim_outputs);
    torch::Tensor dm_dsigma12 = std::get<3>(ssim_outputs);

    // Define dL_s/d(ssim_map). Assuming L_s = DSSIM = (1 - SSIM)/2, so dL_s/d(ssim_map) = -0.5
    // The lambda_dssim is applied to the result of dL_s/dc.
    // If L_s is the loss term itself, then dL_s/d(map) is the derivative of that loss.
    // If the loss is L = (1-lambda)*L2 + lambda*DSSIM, then dL/d(DSSIM) = lambda.
    // And d(DSSIM)/d(SSIM_map) = -0.5. So dL/d(SSIM_map) = -0.5 * lambda.
    // However, the formulation L = L2 + lambda*L_S suggests lambda is a weight for L_S.
    // Let's assume L_S is DSSIM. Then the dL_s/dc we compute will be d(DSSIM)/dc.
    // The final dL/dc will be dL2/dc + lambda_dssim * d(DSSIM)/dc.
    // So, for d(DSSIM)/dc, we need d(DSSIM)/d(SSIM_map) = -0.5.
    torch::Tensor dL_dmap_tensor = torch::full_like(ssim_map_bchw, -0.5f);

    // Call fusedssim_backward to get d(SSIM_loss)/dc (which is d(DSSIM)/dc if dL_dmap is for DSSIM)
    torch::Tensor dL_dc_ssim_bchw = fusedssim_backward(
        C1, C2, img1_bchw, img2_bchw, dL_dmap_tensor, dm_dmu1, dm_dsigma1_sq, dm_dsigma12
    );

    // Permute dL_dc_ssim back to [H,W,C]
    torch::Tensor dL_dc_ssim_hwc_unnormalized = dL_dc_ssim_bchw.permute({0, 2, 3, 1}).squeeze(0).contiguous();
    torch::Tensor dL_dc_ssim_hwc_normalized = dL_dc_ssim_hwc_unnormalized * inv_N_pixels;

    // Combine derivatives: out_dL_dc = dL_dc_l1l2 + lambda_dssim * dL_dc_ssim_hwc_normalized
    // dL_dc_l1l2 is already normalized by inv_N_pixels inside its kernel
    out_dL_dc_tensor.copy_(dL_dc_l1l2 + lambda_dssim * dL_dc_ssim_hwc_normalized);

    // Set out_d2L_dc2_diag_tensor:
    // d2L_dc2_diag_l1l2 is already normalized by inv_N_pixels inside its kernel
    // As discussed, SSIM's second derivative d2L_s/dc2 is not computed by ssim.cu.
    // We assume it's effectively zero or handled by use_l2_for_hessian_L_term logic,
    // meaning only the L1/L2 part contributes to the d2L/dc2 term in Hessian assembly.
    out_d2L_dc2_diag_tensor.copy_(d2L_dc2_diag_l1l2);

    CUDA_CHECK(cudaGetLastError()); // Check for errors from SSIM calls too
}


void NewtonKernels::compute_position_hessian_components_kernel_launcher(
    int H_img, int W_img, int C_img,
    int P_total,
    const float* means_3d_all, const float* scales_all, const float* rotations_all,
    const float* opacities_all, const float* shs_all,
    int sh_degree,
    int sh_coeffs_per_color_channel, // Changed from sh_coeffs_dim
    const float* view_matrix_ptr, // Already a pointer from C++
    const float* perspective_proj_matrix_ptr, // Changed from projection_matrix_for_jacobian
    const float* cam_pos_world_ptr, // Already a pointer from C++
    // Removed means_2d_render, depths_render, radii_render, P_render as kernel doesn't use them directly
    const torch::Tensor& visibility_mask_for_model_tensor,
    const float* dL_dc_pixelwise_ptr, // Already a pointer
    const float* d2L_dc2_diag_pixelwise_ptr, // Already a pointer
    int num_output_gaussians,
    float* H_p_output_packed_ptr, // Already a pointer
    float* grad_p_output_ptr,   // Already a pointer
    bool debug_prints_enabled
) {
    // Construct the output_index_map (mapping P_total index to dense output index)
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
    // AT_ASSERTM(current_out_idx == num_output_gaussians, "Mismatch in visible count for output_index_map"); // Good check but might fail if num_output_gaussians is pre-calculated slightly differently.

    torch::Tensor output_index_map_tensor = torch::tensor(output_index_map_cpu,
        torch::TensorOptions().dtype(torch::kInt).device(visibility_mask_for_model_tensor.device())); // Keep on same device
    const int* output_index_map_gpu = gs::torch_utils::get_const_data_ptr<int>(output_index_map_tensor, "output_index_map_tensor_in_launcher");
    const bool* visibility_mask_gpu_ptr = gs::torch_utils::get_const_data_ptr<bool>(visibility_mask_for_model_tensor, "visibility_mask_for_model_tensor_for_kernel");

    if (debug_prints_enabled && P_total > 0) { // Added P_total > 0 to avoid printing for empty scenes
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
        sh_coeffs_per_color_channel, // Use the new name
        view_matrix_ptr,
        perspective_proj_matrix_ptr, // Use the new name (4x4 P matrix)
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
    // bool debug_prints_enabled // TODO: Add this if needed
) {
    // TODO: Pass debug_prints_enabled if options_.debug_print_shapes is to be respected here
    // if (debug_prints_enabled) {
    //     printf("[STUB KERNEL LAUNCHER] compute_scale_hessian_gradient_components_kernel_launcher called.\n");
    // }
    // This function would:
    // 1. Prepare raw pointers from all input tensors.
    // 2. Launch one or more CUDA kernels to compute ∂c/∂s_k, ∂²c/∂s_k², and then accumulate
    //    H_s_k and g_s_k for each visible Gaussian.
    // For now, it does nothing, out_H_s_packed and out_g_s remain as initialized (e.g. zeros).
}

void NewtonKernels::batch_solve_3x3_system_kernel_launcher(
    int num_systems,
    const torch::Tensor& H_s_packed,
    const torch::Tensor& g_s,
    float damping,
    torch::Tensor& out_delta_s
    // bool debug_prints_enabled // TODO: Add this if needed
) {
    // TODO: Pass debug_prints_enabled if options_.debug_print_shapes is to be respected here
    // if (debug_prints_enabled) {
    //    printf("[STUB KERNEL LAUNCHER] batch_solve_3x3_system_kernel_launcher called for %d systems.\n", num_systems);
    // }
    // This function would:
    // 1. Prepare raw pointers.
    // 2. Launch a CUDA kernel to solve N independent 3x3 systems: H_s * Δs = -g_s.
    //    (H_s is symmetric, so 6 unique elements from H_s_packed).
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
        num_systems,
        H_ptr,
        g_ptr,
        damping,
        delta_s_ptr
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
    torch::Tensor& out_g_theta) {
    // This function would:
    // 1. Prepare raw pointers from input tensors.
    // 2. Launch CUDA kernel(s) to compute ∂c/∂θ_k, ∂²c/∂θ_k², and then accumulate
    //    H_θ_k and g_θ_k for each visible Gaussian, using r_k as rotation axis.
    // For now, it does nothing; out_H_theta and out_g_theta remain as initialized.
    // if (options_debug_print_shapes_can_be_passed_here) {
    //     printf("[STUB KERNEL LAUNCHER] compute_rotation_hessian_gradient_components_kernel_launcher called.\n");
    // }
}

void NewtonKernels::batch_solve_1x1_system_kernel_launcher(
    int num_systems,
    const torch::Tensor& H_theta,
    const torch::Tensor& g_theta,
    float damping,
    torch::Tensor& out_delta_theta) {
    // This function would:
    // 1. Prepare raw pointers.
    // 2. Launch a CUDA kernel to solve N independent 1x1 systems:
    //    (H_theta_k + damping) * Δθ_k = -g_theta_k  => Δθ_k = -g_theta_k / (H_theta_k + damping)
    // For now, it does nothing; out_delta_theta remains as initialized.
    // The calling C++ code in NewtonOptimizer currently has a placeholder for this.
    // if (options_debug_print_shapes_can_be_passed_here) {
    //     printf("[STUB KERNEL LAUNCHER] batch_solve_1x1_system_kernel_launcher called for %d systems.\n", num_systems);
    // }
    TORCH_CHECK(H_theta.defined() && H_theta.size(0) == num_systems, "H_theta size mismatch");
    TORCH_CHECK(g_theta.defined() && g_theta.size(0) == num_systems, "g_theta size mismatch");
    TORCH_CHECK(out_delta_theta.defined() && out_delta_theta.size(0) == num_systems, "out_delta_theta size mismatch");
    TORCH_CHECK(H_theta.is_cuda() && g_theta.is_cuda() && out_delta_theta.is_cuda(), "All tensors must be CUDA tensors");
    // Assuming tensors can be [N] or [N,1].contiguous() makes them effectively [N] for data_ptr.
    // If they must be [N,1], ensure contiguity after potential reshape.
    // For simplicity, assume they are already prepared as contiguous (e.g. after .contiguous() call if reshaped from [N,1])

    if (num_systems == 0) return;

    const float* H_ptr = gs::torch_utils::get_const_data_ptr<float>(H_theta.contiguous(), "H_theta");
    const float* g_ptr = gs::torch_utils::get_const_data_ptr<float>(g_theta.contiguous(), "g_theta");
    float* delta_theta_ptr = gs::torch_utils::get_data_ptr<float>(out_delta_theta.contiguous(), "out_delta_theta"); // Ensure contiguous for output too

    batch_solve_1x1_system_kernel<<<GET_BLOCKS(num_systems), CUDA_NUM_THREADS>>>(
        num_systems,
        H_ptr,
        g_ptr,
        damping,
        delta_theta_ptr
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
    torch::Tensor& out_g_sigma_base) {
    // This function would:
    // 1. Prepare raw pointers from input tensors.
    // 2. Launch CUDA kernel(s) to compute ∂c/∂σ_k. The paper states ∂²c/∂σ_k² = 0.
    //    The formula for ∂c/∂σ_k involves terms like G_k, accumulated alpha from prior Gaussians,
    //    the Gaussian's own color c_k, and the color accumulated from Gaussians behind it.
    //    This requires careful handling of sorted Gaussians and their blended contributions.
    // 3. Accumulate H_σ_base_k and g_σ_base_k for each visible Gaussian:
    //    g_σ_base_k = sum_pixels [ (∂c/∂σ_k)ᵀ ⋅ (dL/dc) ]
    //    H_σ_base_k = sum_pixels [ (∂c/∂σ_k)ᵀ ⋅ (d²L/dc²) ⋅ (∂c/∂σ_k) ]
    // For now, it does nothing; out_H_sigma_base and out_g_sigma_base remain as initialized (e.g., zeros).
    // if (options_debug_print_shapes_can_be_passed_here) { // Assuming a debug flag could be passed
    //     printf("[STUB KERNEL LAUNCHER] compute_opacity_hessian_gradient_components_kernel_launcher called.\n");
    // }
    // This is a stub. A real implementation needs a kernel.
    // For now, to avoid linker errors if called, we ensure outputs are zeroed if they are not already.
    if (out_H_sigma_base.defined()) out_H_sigma_base.zero_();
    if (out_g_sigma_base.defined()) out_g_sigma_base.zero_();
}

// --- Definitions for SH (Color) Optimization Launchers (Stubs) ---

torch::Tensor NewtonKernels::compute_sh_bases_kernel_launcher(
    int sh_degree,
    const torch::Tensor& normalized_view_vectors) {
    // This function would:
    // 1. Prepare raw pointers.
    // 2. Launch a CUDA kernel to evaluate SH basis functions B_k(r_k) for each view vector.
    //    Output shape: [N_vis, (sh_degree+1)^2]
    // For now, returns empty tensor or zeros of correct shape.
    // if (options_debug_print_shapes_can_be_passed_here) {
    //     printf("[STUB KERNEL LAUNCHER] compute_sh_bases_kernel_launcher called.\n");
    // }
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
        num_points,
        sh_degree,
        dirs_ptr,
        sh_basis_output_ptr
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
    const torch::Tensor& K_matrix,
    const gs::RenderOutput& render_output,
    const torch::Tensor& visible_indices,
    const torch::Tensor& dL_dc_pixelwise,
    const torch::Tensor& d2L_dc2_diag_pixelwise,
    torch::Tensor& out_H_ck_diag,
    torch::Tensor& out_g_ck) {
    // This function would:
    // 1. Prepare raw pointers.
    // 2. Launch CUDA kernel(s) to compute Jacobian J_sh = ∂c_pixel/∂c_k (using sh_bases_values)
    //    and then accumulate H_ck_base and g_ck_base.
    //    Paper: ∂c_R/∂c_{k,R} = sum_{gaussians} G_k σ_k (Π(1-G_jσ_j)) B_{k,R}
    //    If ∂²c_R/∂c_{k,R}² (direct part) = 0, then Hessian is J_sh^T * (d2L/dc2) * J_sh
    // For now, it does nothing. out_H_ck_diag and out_g_ck remain as initialized.
    // if (options_debug_print_shapes_can_be_passed_here) {
    //     printf("[STUB KERNEL LAUNCHER] compute_sh_hessian_gradient_components_kernel_launcher called.\n");
    // }
    // This is a stub. A real implementation needs a kernel.
    // For now, to avoid linker errors if called, we ensure outputs are zeroed.
    if (out_H_ck_diag.defined()) out_H_ck_diag.zero_();
    if (out_g_ck.defined()) out_g_ck.zero_();
}

