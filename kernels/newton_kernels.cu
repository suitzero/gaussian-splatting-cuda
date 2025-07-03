// kernels/newton_kernels.cu
#include "newton_kernels.cuh" // Now in the same directory
#include "kernels/ssim.cuh"   // For fusedssim, fusedssim_backward C++ functions
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/torch.h> // For AT_ASSERTM

// Basic CUDA utilities (normally in a separate header)
#define CUDA_CHECK(status) AT_ASSERTM(status == cudaSuccess, cudaGetErrorString(status))

constexpr int CUDA_NUM_THREADS = 256; // Default number of threads per block
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// --- KERNEL IMPLEMENTATIONS ---

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
    int H_img, int W_img, int C_img,
    int P_total,
    const float* means_3d_all, const float* scales_all, const float* rotations_all,
    const float* opacities_all, const float* shs_all, int sh_degree, int sh_coeffs_dim,
    const float* view_matrix, const float* projection_matrix_for_jacobian, const float* cam_pos_world,
    const float* means_2d_render, const float* depths_render, const float* radii_render,
    // visibility_indices_in_render_output (ranks) removed
    int P_render,
    const bool* visibility_mask_for_model,
    const float* dL_dc_pixelwise, const float* d2L_dc2_diag_pixelwise,
    int num_output_gaussians,
    float* H_p_output_packed, float* grad_p_output,
    // Helper: map original P_total index to dense output index (0 to num_output_gaussians-1)
    // This map should be precomputed on CPU and passed if outputs are dense.
    // If visibility_indices_in_render_output is not null, it might serve a similar purpose
    // for mapping render output data.
    const int* output_index_map, // [P_total], value is output slot or -1 if not visible
    bool debug_prints_enabled
) {
    if (debug_prints_enabled && threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("CUDA KERNEL compute_position_hessian_components_kernel started. P_total: %d, num_output_gaussians: %d\n", P_total, num_output_gaussians);
    }

    int p_idx_total = blockIdx.x * blockDim.x + threadIdx.x; // Iterate over all Gaussians in model

    if (p_idx_total >= P_total) return;
    if (!visibility_mask_for_model[p_idx_total]) return;

    int output_idx = output_index_map ? output_index_map[p_idx_total] : -1;
    if (output_idx == -1) return; // Should not happen if visibility_mask_for_model[p_idx_total] is true and map is correct

    // For each visible Gaussian p_idx_total:
    // 1. Get its parameters (means_3d_all[p_idx_total*3], etc.)
    // 2. Compute its influence on pixels (coverage, color contribution). This is part of rasterization.
    //    This step is very complex. It involves projecting the Gaussian, calculating its 2D covariance,
    //    determining which pixels it covers.
    //
    // 3. For each covered pixel (px, py):
    //    a. Get dL/dc(px,py) and d2L/dc2(px,py) from inputs.
    //    b. Compute Jacobian J_c(px,py) = 𝜕c(px,py)/𝜕p_k (how this Gaussian's position change affects this pixel color).
    //       This involves derivatives of projection, SH evaluation, Gaussian PDF, alpha blending. (Eq 16 from paper)
    //    c. Compute Hessian H_c_y(px,py) = 𝜕²c(px,py)/𝜕p_k² (second derivative). (Eq 17 from paper)
    //
    //    d. Accumulate to H_p_k and g_p_k for this Gaussian p_idx_total:
    //       g_p_k += J_c(px,py)^T * dL/dc(px,py)
    //       H_p_k += J_c(px,py)^T * d2L/dc2(px,py) * J_c(px,py) + dL/dc(px,py) * H_c_y(px,py)
    //
    // 4. Store H_p_k (packed) and g_p_k into H_p_output_packed[output_idx*6] and grad_p_output[output_idx*3].

    // --- Placeholder ---
    // This is extremely simplified. A real implementation is hundreds of lines of CUDA.
    // Initialize H_p_k and g_p_k to zero
    float g_p_k[3] = {0,0,0};
    float H_p_k_symm[6] = {0,0,0,0,0,0}; // H00, H01, H02, H11, H12, H22

    // Dummy values for demonstration
    g_p_k[0] = 1.0f * p_idx_total; g_p_k[1] = 0.5f * p_idx_total; g_p_k[2] = 0.1f * p_idx_total;
    H_p_k_symm[0] = 1.0f; // H00
    H_p_k_symm[3] = 1.0f; // H11
    H_p_k_symm[5] = 1.0f; // H22

    for(int i=0; i<3; ++i) grad_p_output[output_idx * 3 + i] = g_p_k[i];
    for(int i=0; i<6; ++i) H_p_output_packed[output_idx * 6 + i] = H_p_k_symm[i];
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
    const float* opacities_all, const float* shs_all, int sh_degree, int sh_coeffs_dim,
    const float* view_matrix, const float* projection_matrix_for_jacobian, const float* cam_pos_world,
    const float* means_2d_render, const float* depths_render, const float* radii_render,
    /* const int* visibility_indices_in_render_output, REMOVED */ int P_render,
    const torch::Tensor& visibility_mask_for_model_tensor, // Changed from const bool*
    const float* dL_dc_pixelwise, const float* d2L_dc2_diag_pixelwise,
    int num_output_gaussians,
    float* H_p_output_packed, float* grad_p_output,
    bool debug_prints_enabled // Added new parameter
) {
    // Precompute output_index_map on CPU/GPU
    // output_index_map: array of size P_total. output_index_map[i] is the dense output index for Gaussian i, or -1.
    // This map is crucial. For now, assuming it's passed or P_total is small enough for simple handling.
    // This is a placeholder for the actual kernel call, which needs the map.
    // For now, the kernel is simplified and assumes P_total is the number of output gaussians if output_index_map is null.
    // This needs to be fixed for a real scenario.

    // Construct the output_index_map using a CPU copy of the visibility mask
    TORCH_CHECK(visibility_mask_for_model_tensor.defined(), "visibility_mask_for_model_tensor is not defined in launcher.");
    TORCH_CHECK(visibility_mask_for_model_tensor.scalar_type() == torch::kBool, "visibility_mask_for_model_tensor must be Bool type.");
    TORCH_CHECK(static_cast<int>(visibility_mask_for_model_tensor.size(0)) == P_total, "visibility_mask_for_model_tensor size mismatch with P_total.");

    torch::Tensor visibility_mask_cpu = visibility_mask_for_model_tensor.to(torch::kCPU).contiguous();
    const bool* cpu_visibility_ptr = visibility_mask_cpu.data_ptr<bool>();

    std::vector<int> output_index_map_cpu(P_total);
    int current_out_idx = 0;
    for(int i=0; i<P_total; ++i) {
        if(cpu_visibility_ptr[i]) { // Using CPU pointer
            output_index_map_cpu[i] = current_out_idx++;
        } else {
            output_index_map_cpu[i] = -1;
        }
    }
    // AT_ASSERTM(current_out_idx == num_output_gaussians, "Mismatch in visible count for output_index_map");

    torch::Tensor output_index_map_tensor = torch::tensor(output_index_map_cpu, torch::kInt).to(torch::kCUDA);

    // Verbose check for output_index_map_tensor
    if (debug_prints_enabled) {
        const std::string name = "output_index_map_tensor_in_launcher";
        const torch::Tensor& tensor = output_index_map_tensor;
        const std::string expected_type_str = "int";
        std::cout << "[VERBOSE_CHECK_LAUNCHER] Tensor: " << name << std::endl;
        if (!tensor.defined()) {
            std::cout << "  - Defined: No" << std::endl;
        } else {
            std::cout << "  - Defined: Yes" << std::endl;
            std::cout << "  - Device: " << tensor.device() << std::endl;
            std::cout << "  - Dtype: " << tensor.scalar_type() << " (Expected: " << expected_type_str << ")" << std::endl;
            std::cout << "  - Contiguous: " << tensor.is_contiguous() << std::endl;
            std::cout << "  - Sizes: " << tensor.sizes() << std::endl;
            std::cout << "  - Numel: " << tensor.numel() << std::endl;
            try {
                if (tensor.numel() > 0) {
                    tensor.data_ptr<int>(); // Test call
                    std::cout << "  - data_ptr<" << expected_type_str << "> call: OK (or returned nullptr for empty)" << std::endl;
                } else {
                    std::cout << "  - data_ptr<" << expected_type_str << "> call: Skipped (numel is 0)" << std::endl;
                }
            } catch (const c10::Error& e) {
                std::cout << "  - data_ptr<" << expected_type_str << "> call: FAILED (c10::Error): " << e.what_without_backtrace() << std::endl;
            } catch (const std::exception& e) {
                std::cout << "  - data_ptr<" << expected_type_str << "> call: FAILED (std::exception): " << e.what() << std::endl;
            } catch (...) {
                std::cout << "  - data_ptr<" << expected_type_str << "> call: FAILED (unknown exception)" << std::endl;
            }
        }
    }

    const int* output_index_map_gpu = gs::torch_utils::get_const_data_ptr<int>(output_index_map_tensor, "output_index_map_tensor_in_launcher");

    // Get the GPU pointer for the visibility_mask_for_model_tensor to pass to the CUDA kernel
    const bool* visibility_mask_gpu_ptr = gs::torch_utils::get_const_data_ptr<bool>(visibility_mask_for_model_tensor, "visibility_mask_for_model_tensor_for_kernel");

    compute_position_hessian_components_kernel<<<GET_BLOCKS(P_total), CUDA_NUM_THREADS>>>(
        H_img, W_img, C_img, P_total, means_3d_all, scales_all, rotations_all, opacities_all,
        shs_all, sh_degree, sh_coeffs_dim, view_matrix, projection_matrix_for_jacobian, cam_pos_world,
        means_2d_render, depths_render, radii_render, /* visibility_indices_in_render_output REMOVED */ P_render,
        visibility_mask_gpu_ptr, dL_dc_pixelwise, d2L_dc2_diag_pixelwise, // Use GPU pointer
        num_output_gaussians, H_p_output_packed, grad_p_output,
        output_index_map_gpu, // Pass the map
        debug_prints_enabled // Pass the flag to the kernel
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

// --- Kernel for batch 1x1 solve ---
// Solves (H + damping) * x = -g for x (scalar case)
__global__ void batch_solve_1x1_system_kernel(
    int num_systems,
    const float* H_scalar, // [N] or [N,1]
    const float* g_scalar, // [N] or [N,1]
    float damping,
    float* out_x) {        // [N] or [N,1]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_systems) return;

    float h_val = H_scalar[idx];
    float g_val = g_scalar[idx];

    float h_damped = h_val + damping;

    if (abs(h_damped) < 1e-9f) { // Avoid division by zero
        out_x[idx] = 0.0f; // Or some other fallback, like -g_val / (small_epsilon)
    } else {
        out_x[idx] = -g_val / h_damped;
    }
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

// --- Kernel for batch 3x3 solve ---
// Solves (H + damping*I) * x = -g for x, where H is symmetric 3x3
// H_packed = [H00, H01, H02, H11, H12, H22]
__global__ void batch_solve_3x3_symmetric_system_kernel(
    int num_systems,
    const float* H_packed, // [N, 6]
    const float* g,        // [N, 3]
    float damping,
    float* out_x) {        // [N, 3]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_systems) return;

    const float* Hp = &H_packed[idx * 6];
    const float* gp = &g[idx * 3];
    float* xp = &out_x[idx * 3];

    // Construct matrix A = H + damping*I
    // A = [ Hp[0]+d  Hp[1]    Hp[2]   ]
    //     [ Hp[1]    Hp[3]+d  Hp[4]   ]
    //     [ Hp[2]    Hp[4]    Hp[5]+d ]
    float a00 = Hp[0] + damping; float a01 = Hp[1];         float a02 = Hp[2];
    float a10 = Hp[1];         float a11 = Hp[3] + damping; float a12 = Hp[4];
    float a20 = Hp[2];         float a21 = Hp[4];         float a22 = Hp[5] + damping;

    // Calculate determinant of A
    float detA = a00 * (a11 * a22 - a12 * a21) -
                 a01 * (a10 * a22 - a12 * a20) +
                 a02 * (a10 * a21 - a11 * a20);

    if (abs(detA) < 1e-9f) { // Check for singularity
        // Fallback: e.g., scaled gradient descent step or zero update
        // x = -g / (diag(H) + damping)
        xp[0] = -gp[0] / (a00 + 1e-6f); // Add small epsilon to avoid div by zero if a00 was zero before damping
        xp[1] = -gp[1] / (a11 + 1e-6f);
        xp[2] = -gp[2] / (a22 + 1e-6f);
        return;
    }

    float invDetA = 1.0f / detA;

    // Calculate adjugate(A) and multiply by -g / detA
    // adj(A)_00 = (a11*a22 - a12*a21)
    // adj(A)_01 = (a02*a21 - a01*a22)
    // adj(A)_02 = (a01*a12 - a02*a11)
    // ... (transpose for cofactor matrix)
    // x = A_inv * (-g)
    xp[0] = invDetA * (
        (a11 * a22 - a12 * a21) * (-gp[0]) +
        (a02 * a21 - a01 * a22) * (-gp[1]) +
        (a01 * a12 - a02 * a11) * (-gp[2])
    );
    xp[1] = invDetA * (
        (a12 * a20 - a10 * a22) * (-gp[0]) +
        (a00 * a22 - a02 * a20) * (-gp[1]) +
        (a02 * a10 - a00 * a12) * (-gp[2])
    );
    xp[2] = invDetA * (
        (a10 * a21 - a11 * a20) * (-gp[0]) +
        (a01 * a20 - a00 * a21) * (-gp[1]) +
        (a00 * a11 - a01 * a10) * (-gp[2])
    );
}

// --- Kernel for batch 1x1 solve ---
// Solves (H + damping) * x = -g for x (scalar case)
__global__ void batch_solve_1x1_system_kernel(
    int num_systems,
    const float* H_scalar, // [N] or [N,1]
    const float* g_scalar, // [N] or [N,1]
    float damping,
    float* out_x) {        // [N] or [N,1]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_systems) return;

    float h_val = H_scalar[idx];
    float g_val = g_scalar[idx];

    float h_damped = h_val + damping;

    if (abs(h_damped) < 1e-9f) { // Avoid division by zero
        out_x[idx] = 0.0f; // Or some other fallback, like -g_val / (small_epsilon)
    } else {
        out_x[idx] = -g_val / h_damped;
    }
}

// Make sure torch_utils.hpp has these definitions or similar:
// namespace gs { namespace torch_utils {
// template <typename T>
// inline const T* get_const_data_ptr(const torch::Tensor& tensor) {
//     TORCH_CHECK(tensor.is_cuda(), "Tensor must be CUDA tensor");
//     TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
//     return tensor.data_ptr<T>();
// }
// template <typename T>
// inline T* get_data_ptr(torch::Tensor& tensor) {
//     TORCH_CHECK(tensor.is_cuda(), "Tensor must be CUDA tensor");
//     TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");
//     return tensor.data_ptr<T>();
// }
// }}

// --- Spherical Harmonics Basis Evaluation Kernel ---
// Based on gsplat's sh_coeffs_to_color_fast, but only computes basis values.
__global__ void eval_sh_basis_kernel(
    const int num_points,
    const int degree,          // Max degree to compute up to
    const float* dirs,         // [num_points, 3], normalized directions
    float* sh_basis_output) {  // [num_points, (degree+1)^2]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float x = dirs[idx * 3 + 0];
    float y = dirs[idx * 3 + 1];
    float z = dirs[idx * 3 + 2];

    // Output is [num_points, max_sh_coeffs_for_degree]
    // max_sh_coeffs_for_degree depends on the input 'degree'
    int num_sh_coeffs = (degree + 1) * (degree + 1);
    float* current_sh_output = sh_basis_output + idx * num_sh_coeffs;

    // Degree 0
    current_sh_output[0] = 0.2820947917738781f; // Y_0_0
    if (degree == 0) return;

    // Degree 1
    // Y_1_-1, Y_1_0, Y_1_1
    // gsplat order seems to be: (y, z, x) for l=1 components based on sh_coeffs_to_color_fast:
    // result += 0.48860251190292f * (-y * coeffs[1*3+c] + z * coeffs[2*3+c] - x * coeffs[3*3+c]);
    // Coeff indices 1,2,3 map to SH components for l=1.
    // Original SH order: Y_1^-1 ~ y, Y_1^0 ~ z, Y_1^1 ~ x
    // Let's follow this convention for basis output.
    current_sh_output[1] = -0.48860251190292f * y; // Corresponds to term with coeffs[1]
    current_sh_output[2] = 0.48860251190292f * z;  // Corresponds to term with coeffs[2]
    current_sh_output[3] = -0.48860251190292f * x; // Corresponds to term with coeffs[3]
    if (degree == 1) return;

    // Degree 2
    // Coeff indices 4,5,6,7,8
    // pSH4 (xy), pSH5 (yz), pSH6 (zz), pSH7 (xz), pSH8 (xx-yy)
    float z2 = z * z;
    float fTmp0B = -1.092548430592079f * z;
    float fC1 = x * x - y * y;
    float fS1 = 2.f * x * y;

    current_sh_output[4] = 0.5462742152960395f * fS1;           // pSH4 -> xy
    current_sh_output[5] = fTmp0B * y;                          // pSH5 -> yz
    current_sh_output[6] = (0.9461746957575601f * z2 - 0.3153915652525201f); // pSH6 -> 3z^2-1 type
    current_sh_output[7] = fTmp0B * x;                          // pSH7 -> xz
    current_sh_output[8] = 0.5462742152960395f * fC1;           // pSH8 -> x^2-y^2
    if (degree == 2) return;

    // Degree 3
    // Coeff indices 9..15
    // pSH9 (S2x), pSH10 (S1z), pSH11 (C0z), pSH12 (z(zz)), pSH13 (C0x), pSH14 (C1z), pSH15 (C2x)
    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    float fC2 = x * fC1 - y * fS1; // x(x^2-y^2) - y(2xy) = x^3 - 3xy^2
    float fS2 = x * fS1 + y * fC1; // x(2xy) + y(x^2-y^2) = 3x^2y - y^3

    current_sh_output[9]  = -0.5900435899266435f * fS2;          // pSH9
    current_sh_output[10] = fTmp1B * fS1;                         // pSH10
    current_sh_output[11] = fTmp0C * y;                           // pSH11
    current_sh_output[12] = z * (1.865881662950577f * z2 - 1.119528997770346f); // pSH12
    current_sh_output[13] = fTmp0C * x;                           // pSH13
    current_sh_output[14] = fTmp1B * fC1;                         // pSH14
    current_sh_output[15] = -0.5900435899266435f * fC2;          // pSH15
    if (degree == 3) return;

    // Degree 4
    // Coeff indices 16..24
    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fC3 = x * fC2 - y * fS2;
    float fS3 = x * fS2 + y * fC2;
    // pSH6 was (0.9461746957575601f * z2 - 0.3153915652525201f)
    // pSH12 was z * (1.865881662950577f * z2 - 1.119528997770346f)
    float pSH6_val = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH12_val = z * (1.865881662950577f * z2 - 1.119528997770346f);


    current_sh_output[16] = 0.6258357354491763f * fS3;            // pSH16
    current_sh_output[17] = fTmp2B * fS2;                         // pSH17
    current_sh_output[18] = fTmp1C * fS1;                         // pSH18
    current_sh_output[19] = fTmp0D * y;                           // pSH19
    current_sh_output[20] = (1.984313483298443f * z * pSH12_val - 1.006230589874905f * pSH6_val); // pSH20
    current_sh_output[21] = fTmp0D * x;                           // pSH21
    current_sh_output[22] = fTmp1C * fC1;                         // pSH22
    current_sh_output[23] = fTmp2B * fC2;                         // pSH23
    current_sh_output[24] = 0.6258357354491763f * fC3;            // pSH24
    // Degree 4 is max supported by this structure
}