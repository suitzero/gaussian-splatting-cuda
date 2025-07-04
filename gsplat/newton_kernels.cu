// kernels/newton_kernels.cu
#include "newton_kernels.cuh"
#include "kernels/ssim.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/torch.h>
#include <cmath>

// GLM includes
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/norm.hpp>

#define CUDA_CHECK(status) AT_ASSERTM(status == cudaSuccess, cudaGetErrorString(status))

constexpr int CUDA_NUM_THREADS = 256;
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// --- CudaMath namespace: Reduced, matrix ops on raw pointers kept for transition ---
namespace CudaMath {

__device__ __forceinline__ void mat3_transpose_inplace(float* M) {
    float temp;
    temp = M[1]; M[1] = M[3]; M[3] = temp;
    temp = M[2]; M[2] = M[6]; M[6] = temp;
    temp = M[5]; M[5] = M[7]; M[7] = temp;
}

__device__ __forceinline__ void outer_product_3x3_ptr_row_major(const glm::vec3& a, const glm::vec3& b, float* out_M_ptr) {
    glm::mat3 result_mat_col_major = glm::outerProduct(a,b);
    const float* p = glm::value_ptr(result_mat_col_major);
    out_M_ptr[0] = p[0]; out_M_ptr[1] = p[3]; out_M_ptr[2] = p[6];
    out_M_ptr[3] = p[1]; out_M_ptr[4] = p[4]; out_M_ptr[5] = p[7];
    out_M_ptr[6] = p[2]; out_M_ptr[7] = p[5]; out_M_ptr[8] = p[8];
}

__device__ __forceinline__ void mul_mat4_vec4_ptr(const float* PW_row_major_ptr, const float* p_k_h_ptr, float* result_ptr) {
    glm::mat4 PW_col_major = glm::transpose(glm::make_mat4(PW_row_major_ptr));
    glm::vec4 p_k_h_glm = glm::make_vec4(p_k_h_ptr[0],p_k_h_ptr[1],p_k_h_ptr[2],p_k_h_ptr[3]);
    glm::vec4 result_glm = PW_col_major * p_k_h_glm;
    result_ptr[0] = result_glm.x; result_ptr[1] = result_glm.y; result_ptr[2] = result_glm.z; result_ptr[3] = result_glm.w;
}

__device__ __forceinline__ void mat_mul_mat_ptr(const float* A_ptr_row_major, const float* B_ptr_row_major, float* C_ptr_row_major,
                                             int A_rows, int A_cols_B_rows, int B_cols) {
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols_B_rows; ++k) {
                sum += A_ptr_row_major[i * A_cols_B_rows + k] * B_ptr_row_major[k * B_cols + j];
            }
            C_ptr_row_major[i * B_cols + j] = sum;
        }
    }
}
} // namespace CudaMath

namespace ProjectionDerivs {

__device__ __forceinline__ void compute_h_vec(const glm::vec3& p_k, const glm::mat4& PW_col_major, glm::vec4& h_vec_out) {
    h_vec_out = PW_col_major * glm::vec4(p_k, 1.0f);
}

__device__ __forceinline__ void compute_projection_jacobian(
    const glm::mat4& PW_col_major, float W_I_t, float H_I_t,
    const glm::vec4& h_vec, float* jacobian_out_2x3_row_major
) {
    float hx = h_vec.x; float hy = h_vec.y; float hw = h_vec.w;
    float inv_hw = 1.0f / (hw + 1e-8f);
    float inv_hw_sq = inv_hw * inv_hw;
    float term_x_coeff = W_I_t / 2.0f; float term_y_coeff = H_I_t / 2.0f;

    jacobian_out_2x3_row_major[0] = term_x_coeff * (inv_hw * PW_col_major[0][0] - hx * inv_hw_sq * PW_col_major[0][3]);
    jacobian_out_2x3_row_major[1] = term_x_coeff * (inv_hw * PW_col_major[1][0] - hx * inv_hw_sq * PW_col_major[1][3]);
    jacobian_out_2x3_row_major[2] = term_x_coeff * (inv_hw * PW_col_major[2][0] - hx * inv_hw_sq * PW_col_major[2][3]);
    jacobian_out_2x3_row_major[3] = term_y_coeff * (inv_hw * PW_col_major[0][1] - hy * inv_hw_sq * PW_col_major[0][3]);
    jacobian_out_2x3_row_major[4] = term_y_coeff * (inv_hw * PW_col_major[1][1] - hy * inv_hw_sq * PW_col_major[1][3]);
    jacobian_out_2x3_row_major[5] = term_y_coeff * (inv_hw * PW_col_major[2][1] - hy * inv_hw_sq * PW_col_major[2][3]);
}

__device__ __forceinline__ void compute_projection_hessian(
    const glm::mat4& PW_col_major, float W_I_t, float H_I_t,
    const glm::vec4& h_vec,
    float* hessian_out_pi_x_3x3_row_major, float* hessian_out_pi_y_3x3_row_major
) {
    float hx = h_vec.x; float hy = h_vec.y; float hw = h_vec.w;
    float inv_hw_sq = 1.0f / (hw * hw + 1e-8f);
    float hw_cubed = hw * hw * hw + 1e-9f;

    glm::vec3 PWr0 = glm::vec3(PW_col_major[0][0], PW_col_major[1][0], PW_col_major[2][0]);
    glm::vec3 PWr1 = glm::vec3(PW_col_major[0][1], PW_col_major[1][1], PW_col_major[2][1]);
    glm::vec3 PWr3 = glm::vec3(PW_col_major[0][3], PW_col_major[1][3], PW_col_major[2][3]);

    glm::mat3 PW3_outer_PW3 = glm::outerProduct(PWr3, PWr3);
    glm::mat3 PW3_outer_PW0 = glm::outerProduct(PWr3, PWr0);
    glm::mat3 PW0_outer_PW3 = glm::transpose(PW3_outer_PW0);
    glm::mat3 PW3_outer_PW1 = glm::outerProduct(PWr3, PWr1);
    glm::mat3 PW1_outer_PW3 = glm::transpose(PW3_outer_PW1);

    float factor_x1 = W_I_t * (2.0f * hx / hw_cubed);
    float factor_x2 = W_I_t * (-1.0f * inv_hw_sq);
    glm::mat3 H_pi_x_col_major = factor_x1 * PW3_outer_PW3 + factor_x2 * (PW3_outer_PW0 + PW0_outer_PW3);
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) hessian_out_pi_x_3x3_row_major[i*3+j] = H_pi_x_col_major[j][i];

    float factor_y1 = H_I_t * (2.0f * hy / hw_cubed);
    float factor_y2 = H_I_t * (-1.0f * inv_hw_sq);
    glm::mat3 H_pi_y_col_major = factor_y1 * PW3_outer_PW3 + factor_y2 * (PW3_outer_PW1 + PW1_outer_PW3);
    for(int i=0; i<3; ++i) for(int j=0; j<3; ++j) hessian_out_pi_y_3x3_row_major[i*3+j] = H_pi_y_col_major[j][i];
}
} // namespace ProjectionDerivs

namespace SHDerivs {

__device__ __forceinline__ void eval_sh_basis_up_to_degree3(
    int degree, const glm::vec3& r_k_normalized, float* basis_out
) {
    float x = r_k_normalized.x; float y = r_k_normalized.y; float z = r_k_normalized.z;
    basis_out[0]=0.2820947917738781f; if(degree==0)return;
    basis_out[1]=-0.48860251190292f*y; basis_out[2]=0.48860251190292f*z; basis_out[3]=-0.48860251190292f*x; if(degree==1)return;
    float x2=x*x; float y2=y*y; float z2=z*z;
    basis_out[4]=0.5462742152960395f*(2.f*x*y); basis_out[5]=-1.092548430592079f*y*z;
    basis_out[6]=0.3153915652525201f*(3.f*z2-1.f); basis_out[7]=-1.092548430592079f*x*z;
    basis_out[8]=0.5462742152960395f*(x2-y2); if(degree==2)return;
    float fC1=x2-y2; float fS1=2.f*x*y; float fC2=x*fC1-y*fS1; float fS2=y*fC1+x*fS1;
    float fTmp0C=-2.285228997322329f*z2+0.4570457994644658f; float fTmp1B=1.445305721320277f*z;
    basis_out[9]=-0.5900435899266435f*fS2; basis_out[10]=fTmp1B*fS1; basis_out[11]=fTmp0C*y;
    basis_out[12]=z*(1.865881662950577f*z2-1.119528997770346f); basis_out[13]=fTmp0C*x;
    basis_out[14]=fTmp1B*fC1; basis_out[15]=-0.5900435899266435f*fC2;
}

__device__ __forceinline__ glm::mat3 compute_drk_dpk_glm(
    const glm::vec3& r_k_normalized, float r_k_norm
) {
    float inv_r_k_norm = 1.0f / (r_k_norm + 1e-8f);
    glm::mat3 I_minus_rktrk = glm::mat3(1.0f) - glm::outerProduct(r_k_normalized, r_k_normalized);
    return inv_r_k_norm * I_minus_rktrk;
}

__device__ __forceinline__ void compute_dphi_drk_up_to_degree3(
    int degree, const glm::vec3& r_k_normalized, float* dPhi_drk_out
) {
    float x=r_k_normalized.x; float y=r_k_normalized.y; float z=r_k_normalized.z; float x2=x*x; float y2=y*y; float z2=z*z;
    dPhi_drk_out[0]=0.f; dPhi_drk_out[1]=0.f; dPhi_drk_out[2]=0.f; if(degree==0)return; // Coeff 0
    dPhi_drk_out[3]=0.f; dPhi_drk_out[4]=-0.48860251190292f; dPhi_drk_out[5]=0.f; // Coeff 1
    dPhi_drk_out[6]=0.f; dPhi_drk_out[7]=0.f; dPhi_drk_out[8]=0.48860251190292f; // Coeff 2
    dPhi_drk_out[9]=-0.48860251190292f; dPhi_drk_out[10]=0.f; dPhi_drk_out[11]=0.f; if(degree==1)return; // Coeff 3
    const float C2_0v=1.092548430592079f; const float C2_1v=-1.092548430592079f; const float C2_2vs=0.9461746957575601f;
    dPhi_drk_out[12]=(C2_0v/2.f)*y; dPhi_drk_out[13]=(C2_0v/2.f)*x; dPhi_drk_out[14]=0.f; // Coeff 4
    dPhi_drk_out[15]=0.f; dPhi_drk_out[16]=C2_1v*z; dPhi_drk_out[17]=C2_1v*y; // Coeff 5
    dPhi_drk_out[18]=0.f; dPhi_drk_out[19]=0.f; dPhi_drk_out[20]=(C2_2vs/3.f)*(6.f*z); // Coeff 6
    dPhi_drk_out[21]=C2_1v*z; dPhi_drk_out[22]=0.f; dPhi_drk_out[23]=C2_1v*x; // Coeff 7
    dPhi_drk_out[24]=(C2_0v/2.f)*(2.f*x); dPhi_drk_out[25]=(C2_0v/2.f)*(-2.f*y); dPhi_drk_out[26]=0.f; if(degree==2)return; // Coeff 8
    const float K9v=-0.5900435899266435f; const float K10zcv=1.445305721320277f;
    const float K11acv=-2.285228997322329f; const float K11bcv=0.4570457994644658f;
    const float K12acv=1.865881662950577f; const float K12bcv=-1.119528997770346f; const float K15v=-0.5900435899266435f;
    dPhi_drk_out[27]=K9v*(6.f*x*y); dPhi_drk_out[28]=K9v*(3.f*x2-3.f*y2); dPhi_drk_out[29]=0.f; // Coeff 9
    dPhi_drk_out[30]=K10zcv*z*(2.f*y); dPhi_drk_out[31]=K10zcv*z*(2.f*x); dPhi_drk_out[32]=K10zcv*(2.f*x*y); // Coeff 10
    dPhi_drk_out[33]=0.f; dPhi_drk_out[34]=K11acv*z2+K11bcv; dPhi_drk_out[35]=K11acv*y*(2.f*z);  // Coeff 11
    dPhi_drk_out[36]=0.f; dPhi_drk_out[37]=0.f; dPhi_drk_out[38]=K12acv*3.f*z2+K12bcv; // Coeff 12
    dPhi_drk_out[39]=K11acv*z2+K11bcv; dPhi_drk_out[40]=0.f; dPhi_drk_out[41]=K11acv*x*(2.f*z); // Coeff 13
    dPhi_drk_out[42]=K10zcv*z*(2.f*x); dPhi_drk_out[43]=K10zcv*z*(-2.f*y); dPhi_drk_out[44]=K10zcv*(x2-y2); // Coeff 14
    dPhi_drk_out[45]=K15v*(3.f*x2-3.f*y2); dPhi_drk_out[46]=K15v*(-6.f*x*y); dPhi_drk_out[47]=0.f; // Coeff 15
}

__device__ __forceinline__ void compute_sh_color_jacobian_single_channel(
    const float* sh_coeffs_single_channel, const float* sh_basis_values,
    const float* dPhi_drk_ptr, const glm::mat3& drk_dpk_mat_col_major,
    int num_basis_coeffs, glm::vec3& jac_out_glm
) {
    jac_out_glm = glm::vec3(0.0f);
    for (int i = 0; i < num_basis_coeffs; ++i) {
        glm::vec3 dPhi_drk_row_i = glm::vec3(dPhi_drk_ptr[i*3+0], dPhi_drk_ptr[i*3+1], dPhi_drk_ptr[i*3+2]);
        glm::vec3 m_prod_row_i = dPhi_drk_row_i * drk_dpk_mat_col_major;
        float v_i = sh_basis_values[i] * sh_coeffs_single_channel[i];
        jac_out_glm += v_i * m_prod_row_i;
    }
}
} // namespace SHDerivs

__device__ __forceinline__ void get_projected_cov2d_and_derivs_placeholder(
    const glm::vec3& p_k_world,
    const float* scales_k, const float* rotations_k,
    const float* view_matrix_ptr, const float* proj_matrix_ptr,
    const float* jacobian_d_pi_d_pk_row_major,
    float img_W, float img_H,
    float* cov2d_sym_row_major, float* inv_cov2d_sym_row_major, float* det_cov2d,
    glm::vec2& d_Gk_d_pik, glm::mat2& d2_Gk_d_pik2_col_major
) {
    cov2d_sym_row_major[0]=1.f; cov2d_sym_row_major[1]=0.f; cov2d_sym_row_major[2]=1.f;
    inv_cov2d_sym_row_major[0]=1.f; inv_cov2d_sym_row_major[1]=0.f; inv_cov2d_sym_row_major[2]=1.f;
    *det_cov2d=1.f;
    d_Gk_d_pik = glm::vec2(0.f,0.f);
    // Storing into column-major glm::mat2
    d2_Gk_d_pik2_col_major[0][0] = -1.f*inv_cov2d_sym_row_major[0]; // xx
    d2_Gk_d_pik2_col_major[1][0] = -1.f*inv_cov2d_sym_row_major[1]; // xy (col 0, row 1)
    d2_Gk_d_pik2_col_major[0][1] = -1.f*inv_cov2d_sym_row_major[1]; // yx (col 1, row 0)
    d2_Gk_d_pik2_col_major[1][1] = -1.f*inv_cov2d_sym_row_major[2]; // yy
}

__global__ void compute_l1l2_loss_derivatives_kernel(
    const float* rendered_image, const float* gt_image, bool use_l2_loss_term,
    float inv_N_pixels, float* out_dL_dc_l1l2, float* out_d2L_dc2_diag_l1l2,
    int H, int W, int C) {
    int idx = blockIdx.x*blockDim.x+threadIdx.x; int total_elements=H*W*C; if(idx>=total_elements)return;
    float diff=rendered_image[idx]-gt_image[idx];
    if(use_l2_loss_term){out_dL_dc_l1l2[idx]=inv_N_pixels*2.f*diff; out_d2L_dc2_diag_l1l2[idx]=inv_N_pixels*2.f;}
    else{out_dL_dc_l1l2[idx]=inv_N_pixels*((diff>1e-6f)?1.f:((diff<-1e-6f)?-1.f:0.f)); out_d2L_dc2_diag_l1l2[idx]=0.f;}
}

__global__ void compute_position_hessian_components_kernel(
    int H_img, int W_img, int C_img, int P_total,
    const float* means_3d_all, const float* scales_all, const float* rotations_all,
    const float* opacities_all, const float* shs_all, int sh_degree,
    int sh_coeffs_per_color_channel,
    const float* view_matrix_ptr, const float* perspective_proj_matrix_ptr,
    const float* cam_pos_world_ptr, const bool* visibility_mask_for_model,
    const float* dL_dc_pixelwise, const float* d2L_dc2_diag_pixelwise,
    int num_output_gaussians, float* H_p_output_packed, float* grad_p_output,
    const int* output_index_map, bool debug_prints_enabled
) {
    int p_idx_total=blockIdx.x*blockDim.x+threadIdx.x;
    if(p_idx_total>=P_total || !visibility_mask_for_model[p_idx_total])return;
    int output_idx=output_index_map[p_idx_total]; if(output_idx<0||output_idx>=num_output_gaussians)return;

    glm::vec3 pk_vec3(means_3d_all[p_idx_total*3+0], means_3d_all[p_idx_total*3+1], means_3d_all[p_idx_total*3+2]);
    const float* scales_k=scales_all+p_idx_total*3; const float* rotations_k=rotations_all+p_idx_total*4;
    float opacity_k=opacities_all[p_idx_total];
    const float* sh_coeffs_k_all_channels=shs_all+p_idx_total*sh_coeffs_per_color_channel*3;
    glm::vec3 cam_pos_world_vec3(cam_pos_world_ptr[0],cam_pos_world_ptr[1],cam_pos_world_ptr[2]);
    glm::vec3 view_dir_to_pk_unnormalized=pk_vec3-cam_pos_world_vec3;
    float r_k_norm=glm::length(view_dir_to_pk_unnormalized);
    glm::vec3 r_k_normalized=glm::normalize(view_dir_to_pk_unnormalized);

    glm::mat4 V_col_major = glm::transpose(glm::make_mat4(view_matrix_ptr));
    glm::mat4 P_col_major = glm::transpose(glm::make_mat4(perspective_proj_matrix_ptr));
    glm::mat4 PW_col_major = P_col_major * V_col_major;

    glm::vec4 h_vec4_data;
    ProjectionDerivs::compute_h_vec(pk_vec3, PW_col_major, h_vec4_data);
    float d_pi_d_pk_data_row_major[6];
    ProjectionDerivs::compute_projection_jacobian(PW_col_major, (float)W_img, (float)H_img, h_vec4_data, d_pi_d_pk_data_row_major);
    float d2_pi_d_pk2_x_data_row_major[9]; float d2_pi_d_pk2_y_data_row_major[9];
    ProjectionDerivs::compute_projection_hessian(PW_col_major, (float)W_img, (float)H_img, h_vec4_data, d2_pi_d_pk2_x_data_row_major, d2_pi_d_pk2_y_data_row_major);

    float sh_basis_eval_data[16]; SHDerivs::eval_sh_basis_up_to_degree3(sh_degree,r_k_normalized,sh_basis_eval_data);
    glm::mat3 drk_dpk_mat_col_major = SHDerivs::compute_drk_dpk_glm(r_k_normalized,r_k_norm);
    float d_phi_d_rk_data_row_major[16*3]; SHDerivs::compute_dphi_drk_up_to_degree3(sh_degree,r_k_normalized,d_phi_d_rk_data_row_major);

    glm::vec3 d_c_bar_R_d_pk_val, d_c_bar_G_d_pk_val, d_c_bar_B_d_pk_val;
    float sh_coeffs_k_R[16],sh_coeffs_k_G[16],sh_coeffs_k_B[16];
    for(int i=0;i<sh_coeffs_per_color_channel;++i){sh_coeffs_k_R[i]=sh_coeffs_k_all_channels[i*3+0]; sh_coeffs_k_G[i]=sh_coeffs_k_all_channels[i*3+1]; sh_coeffs_k_B[i]=sh_coeffs_k_all_channels[i*3+2];}
    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_R,sh_basis_eval_data,d_phi_d_rk_data_row_major,drk_dpk_mat_col_major,sh_coeffs_per_color_channel,d_c_bar_R_d_pk_val);
    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_G,sh_basis_eval_data,d_phi_d_rk_data_row_major,drk_dpk_mat_col_major,sh_coeffs_per_color_channel,d_c_bar_G_d_pk_val);
    SHDerivs::compute_sh_color_jacobian_single_channel(sh_coeffs_k_B,sh_basis_eval_data,d_phi_d_rk_data_row_major,drk_dpk_mat_col_major,sh_coeffs_per_color_channel,d_c_bar_B_d_pk_val);

    glm::vec3 g_p_k_accum_val(0.f); float H_p_k_accum_symm[6]={0.f};
    for(int r=0;r<H_img;++r){for(int c=0;c<W_img;++c){
        float pixel_ndc_x=(2.f*(c+0.5f)/W_img-1.f); float pixel_ndc_y=(2.f*(r+0.5f)/H_img-1.f);
        float pi_k_ndc_x=h_vec4_data.x/(h_vec4_data.w+1e-7f); float pi_k_ndc_y=h_vec4_data.y/(h_vec4_data.w+1e-7f);
        glm::vec2 diff_ndc(pi_k_ndc_x-pixel_ndc_x, pi_k_ndc_y-pixel_ndc_y);
        float cov2d_sym_rm[3],inv_cov2d_sym_rm[3],det_cov2d; glm::vec2 dGk_dPi; glm::mat2 d2Gk_dPi2_cm;
        get_projected_cov2d_and_derivs_placeholder(pk_vec3,scales_k,rotations_k,view_matrix_ptr,perspective_proj_matrix_ptr,d_pi_d_pk_data_row_major,(float)W_img,(float)H_img,cov2d_sym_rm,inv_cov2d_sym_rm,&det_cov2d,dGk_dPi,d2Gk_dPi2_cm);
        float G_k_pixel=expf(-0.5f*(diff_ndc.x*diff_ndc.x*inv_cov2d_sym_rm[0]+2*diff_ndc.x*diff_ndc.y*inv_cov2d_sym_rm[1]+diff_ndc.y*diff_ndc.y*inv_cov2d_sym_rm[2]));
        if(det_cov2d<=1e-7f)G_k_pixel=0.f; if(G_k_pixel<1e-4f)continue;

        glm::vec2 sigma_inv_diff(inv_cov2d_sym_rm[0]*diff_ndc.x+inv_cov2d_sym_rm[1]*diff_ndc.y, inv_cov2d_sym_rm[1]*diff_ndc.x+inv_cov2d_sym_rm[2]*diff_ndc.y);
        dGk_dPi = -G_k_pixel * sigma_inv_diff;
        d2Gk_dPi2_cm[0][0]=G_k_pixel*(sigma_inv_diff.x*sigma_inv_diff.x-inv_cov2d_sym_rm[0]);
        d2Gk_dPi2_cm[1][0]=G_k_pixel*(sigma_inv_diff.x*sigma_inv_diff.y-inv_cov2d_sym_rm[1]); // For col-major mat2, this is (0,1)
        d2Gk_dPi2_cm[0][1]=d2Gk_dPi2_cm[1][0]; // For col-major mat2, this is (1,0)
        d2Gk_dPi2_cm[1][1]=G_k_pixel*(sigma_inv_diff.y*sigma_inv_diff.y-inv_cov2d_sym_rm[2]);

        float alpha_k_pixel=opacity_k*G_k_pixel;
        glm::vec3 c_bar_k_rgb_val(0.f);
        for(int i=0;i<sh_coeffs_per_color_channel;++i)c_bar_k_rgb_val.x+=sh_coeffs_k_R[i]*sh_basis_eval_data[i];
        for(int i=0;i<sh_coeffs_per_color_channel;++i)c_bar_k_rgb_val.y+=sh_coeffs_k_G[i]*sh_basis_eval_data[i];
        for(int i=0;i<sh_coeffs_per_color_channel;++i)c_bar_k_rgb_val.z+=sh_coeffs_k_B[i]*sh_basis_eval_data[i];
        glm::vec3 d_c_final_d_Gk_val=c_bar_k_rgb_val*opacity_k;
        glm::vec3 d_Gk_d_pk_chain_val;
        d_Gk_d_pk_chain_val.x=dGk_dPi.x*d_pi_d_pk_data_row_major[0]+dGk_dPi.y*d_pi_d_pk_data_row_major[3];
        d_Gk_d_pk_chain_val.y=dGk_dPi.x*d_pi_d_pk_data_row_major[1]+dGk_dPi.y*d_pi_d_pk_data_row_major[4];
        d_Gk_d_pk_chain_val.z=dGk_dPi.x*d_pi_d_pk_data_row_major[2]+dGk_dPi.y*d_pi_d_pk_data_row_major[5];
        glm::vec3 J_c_pk_R=d_c_bar_R_d_pk_val*alpha_k_pixel+d_Gk_d_pk_chain_val*d_c_final_d_Gk_val.x;
        glm::vec3 J_c_pk_G=d_c_bar_G_d_pk_val*alpha_k_pixel+d_Gk_d_pk_chain_val*d_c_final_d_Gk_val.y;
        glm::vec3 J_c_pk_B=d_c_bar_B_d_pk_val*alpha_k_pixel+d_Gk_d_pk_chain_val*d_c_final_d_Gk_val.z;
        int pixel_idx_flat=(r*W_img+c)*C_img;
        glm::vec3 dL_dc_val(dL_dc_pixelwise[pixel_idx_flat+0],dL_dc_pixelwise[pixel_idx_flat+1],dL_dc_pixelwise[pixel_idx_flat+2]);
        glm::vec3 d2L_dc2_diag_val(d2L_dc2_diag_pixelwise[pixel_idx_flat+0],d2L_dc2_diag_pixelwise[pixel_idx_flat+1],d2L_dc2_diag_pixelwise[pixel_idx_flat+2]);
        g_p_k_accum_val.x+=J_c_pk_R.x*dL_dc_val.x+J_c_pk_G.x*dL_dc_val.y+J_c_pk_B.x*dL_dc_val.z;
        g_p_k_accum_val.y+=J_c_pk_R.y*dL_dc_val.x+J_c_pk_G.y*dL_dc_val.y+J_c_pk_B.y*dL_dc_val.z;
        g_p_k_accum_val.z+=J_c_pk_R.z*dL_dc_val.x+J_c_pk_G.z*dL_dc_val.y+J_c_pk_B.z*dL_dc_val.z;
        H_p_k_accum_symm[0]+=J_c_pk_R.x*d2L_dc2_diag_val.x*J_c_pk_R.x+J_c_pk_G.x*d2L_dc2_diag_val.y*J_c_pk_G.x+J_c_pk_B.x*d2L_dc2_diag_val.z*J_c_pk_B.x;
        H_p_k_accum_symm[1]+=J_c_pk_R.x*d2L_dc2_diag_val.x*J_c_pk_R.y+J_c_pk_G.x*d2L_dc2_diag_val.y*J_c_pk_G.y+J_c_pk_B.x*d2L_dc2_diag_val.z*J_c_pk_B.y;
        H_p_k_accum_symm[2]+=J_c_pk_R.x*d2L_dc2_diag_val.x*J_c_pk_R.z+J_c_pk_G.x*d2L_dc2_diag_val.y*J_c_pk_G.z+J_c_pk_B.x*d2L_dc2_diag_val.z*J_c_pk_B.z;
        H_p_k_accum_symm[3]+=J_c_pk_R.y*d2L_dc2_diag_val.x*J_c_pk_R.y+J_c_pk_G.y*d2L_dc2_diag_val.y*J_c_pk_G.y+J_c_pk_B.y*d2L_dc2_diag_val.z*J_c_pk_B.y;
        H_p_k_accum_symm[4]+=J_c_pk_R.y*d2L_dc2_diag_val.x*J_c_pk_R.z+J_c_pk_G.y*d2L_dc2_diag_val.y*J_c_pk_G.z+J_c_pk_B.y*d2L_dc2_diag_val.z*J_c_pk_B.z;
        H_p_k_accum_symm[5]+=J_c_pk_R.z*d2L_dc2_diag_val.x*J_c_pk_R.z+J_c_pk_G.z*d2L_dc2_diag_val.y*J_c_pk_G.z+J_c_pk_B.z*d2L_dc2_diag_val.z*J_c_pk_B.z;
    }}
    grad_p_output[output_idx*3+0]=g_p_k_accum_val.x; grad_p_output[output_idx*3+1]=g_p_k_accum_val.y; grad_p_output[output_idx*3+2]=g_p_k_accum_val.z;
    for(int i=0;i<6;++i)H_p_output_packed[output_idx*6+i]=H_p_k_accum_symm[i];
}

__global__ void project_position_hessian_gradient_kernel(
    int num_visible_gaussians,
    const float* H_p_packed_input,
    const float* grad_p_input,
    const float* means_3d_visible,
    const float* view_matrix,
    const float* cam_pos_world,
    float* out_H_v_packed,
    float* out_grad_v) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=num_visible_gaussians)return;
    float ux[3]={view_matrix[0],view_matrix[4],view_matrix[8]}; float uy[3]={view_matrix[1],view_matrix[5],view_matrix[9]};
    out_grad_v[idx*2+0]=ux[0]*grad_p_input[idx*3+0]+ux[1]*grad_p_input[idx*3+1]+ux[2]*grad_p_input[idx*3+2];
    out_grad_v[idx*2+1]=uy[0]*grad_p_input[idx*3+0]+uy[1]*grad_p_input[idx*3+1]+uy[2]*grad_p_input[idx*3+2];
    const float* Hp=&H_p_packed_input[idx*6];
    float Hpu_x[3]; Hpu_x[0]=Hp[0]*ux[0]+Hp[1]*ux[1]+Hp[2]*ux[2]; Hpu_x[1]=Hp[1]*ux[0]+Hp[3]*ux[1]+Hp[4]*ux[2]; Hpu_x[2]=Hp[2]*ux[0]+Hp[4]*ux[1]+Hp[5]*ux[2];
    float Hpu_y[3]; Hpu_y[0]=Hp[0]*uy[0]+Hp[1]*uy[1]+Hp[2]*uy[2]; Hpu_y[1]=Hp[1]*uy[0]+Hp[3]*uy[1]+Hp[4]*uy[2]; Hpu_y[2]=Hp[2]*uy[0]+Hp[4]*uy[1]+Hp[5]*uy[2];
    out_H_v_packed[idx*3+0]=ux[0]*Hpu_x[0]+ux[1]*Hpu_x[1]+ux[2]*Hpu_x[2];
    out_H_v_packed[idx*3+1]=ux[0]*Hpu_y[0]+ux[1]*Hpu_y[1]+ux[2]*Hpu_y[2];
    out_H_v_packed[idx*3+2]=uy[0]*Hpu_y[0]+uy[1]*Hpu_y[1]+uy[2]*Hpu_y[2];
}

__global__ void batch_solve_2x2_system_kernel(
    int num_systems, const float* H_v_packed, const float* g_v, float damping, float step_scale, float* out_delta_v
) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=num_systems)return;
    float H00=H_v_packed[idx*3+0]; float H01=H_v_packed[idx*3+1]; float H11=H_v_packed[idx*3+2];
    float g0=g_v[idx*2+0]; float g1=g_v[idx*2+1];
    H00+=damping; H11+=damping; float det=H00*H11-H01*H01;
    if(abs(det)<1e-8f){out_delta_v[idx*2+0]=-step_scale*g0/(H00+1e-6f); out_delta_v[idx*2+1]=-step_scale*g1/(H11+1e-6f); return;}
    float inv_det=1.f/det;
    out_delta_v[idx*2+0]=-step_scale*inv_det*(H11*g0-H01*g1); out_delta_v[idx*2+1]=-step_scale*inv_det*(-H01*g0+H00*g1);
}

__global__ void project_update_to_3d_kernel(
    int num_updates, const float* delta_v, const float* means_3d_visible,
    const float* view_matrix, const float* cam_pos_world, float* out_delta_p
) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=num_updates)return;
    float ux[3]={view_matrix[0],view_matrix[4],view_matrix[8]}; float uy[3]={view_matrix[1],view_matrix[5],view_matrix[9]};
    float dvx=delta_v[idx*2+0]; float dvy=delta_v[idx*2+1];
    out_delta_p[idx*3+0]=ux[0]*dvx+uy[0]*dvy; out_delta_p[idx*3+1]=ux[1]*dvx+uy[1]*dvy; out_delta_p[idx*3+2]=ux[2]*dvx+uy[2]*dvy;
}

__global__ void eval_sh_basis_kernel(
    const int num_points, const int degree, const float* dirs, float* sh_basis_output
) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=num_points)return;
    float x=dirs[idx*3+0]; float y=dirs[idx*3+1]; float z=dirs[idx*3+2];
    int num_sh_coeffs=(degree+1)*(degree+1); float*current_sh_output=sh_basis_output+idx*num_sh_coeffs;
    current_sh_output[0]=0.2820947917738781f; if(degree==0)return;
    current_sh_output[1]=-0.48860251190292f*y; current_sh_output[2]=0.48860251190292f*z; current_sh_output[3]=-0.48860251190292f*x; if(degree==1)return;
    float z2=z*z; float fTmp0B=-1.092548430592079f*z; float fC1=x*x-y*y; float fS1=2.f*x*y;
    current_sh_output[4]=0.5462742152960395f*fS1; current_sh_output[5]=fTmp0B*y;
    current_sh_output[6]=(0.9461746957575601f*z2-0.3153915652525201f); current_sh_output[7]=fTmp0B*x; current_sh_output[8]=0.5462742152960395f*fC1; if(degree==2)return;
    float fTmp0C=-2.285228997322329f*z2+0.4570457994644658f; float fTmp1B=1.445305721320277f*z; float fC2=x*fC1-y*fS1; float fS2=x*fS1+y*fC1;
    current_sh_output[9]=-0.5900435899266435f*fS2; current_sh_output[10]=fTmp1B*fS1; current_sh_output[11]=fTmp0C*y;
    current_sh_output[12]=z*(1.865881662950577f*z2-1.119528997770346f); current_sh_output[13]=fTmp0C*x;
    current_sh_output[14]=fTmp1B*fC1; current_sh_output[15]=-0.5900435899266435f*fC2; if(degree==3)return;
    float fTmp0D=z*(-4.683325804901025f*z2+2.007139630671868f); float fTmp1C=3.31161143515146f*z2-0.47308734787878f; float fTmp2B=-1.770130769779931f*z;
    float fC3=x*fC2-y*fS2; float fS3=x*fS2+y*fC2; float pSH6_val=(0.9461746957575601f*z2-0.3153915652525201f); float pSH12_val=z*(1.865881662950577f*z2-1.119528997770346f);
    current_sh_output[16]=0.6258357354491763f*fS3; current_sh_output[17]=fTmp2B*fS2; current_sh_output[18]=fTmp1C*fS1; current_sh_output[19]=fTmp0D*y;
    current_sh_output[20]=(1.984313483298443f*z*pSH12_val-1.006230589874905f*pSH6_val); current_sh_output[21]=fTmp0D*x; current_sh_output[22]=fTmp1C*fC1;
    current_sh_output[23]=fTmp2B*fC2; current_sh_output[24]=0.6258357354491763f*fC3;
}

__global__ void batch_solve_3x3_symmetric_system_kernel(
    int num_systems, const float* H_packed, const float* g, float damping, float* out_x
) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=num_systems)return;
    const float*Hp=&H_packed[idx*6]; const float*gp=&g[idx*3]; float*xp=&out_x[idx*3];
    float a00=Hp[0]+damping; float a01=Hp[1]; float a02=Hp[2];
    float a10=Hp[1]; float a11=Hp[3]+damping; float a12=Hp[4];
    float a20=Hp[2]; float a21=Hp[4]; float a22=Hp[5]+damping;
    float detA=a00*(a11*a22-a12*a21)-a01*(a10*a22-a12*a20)+a02*(a10*a21-a11*a20);
    if(abs(detA)<1e-9f){xp[0]=-gp[0]/(a00+1e-6f);xp[1]=-gp[1]/(a11+1e-6f);xp[2]=-gp[2]/(a22+1e-6f);return;}
    float invDetA=1.0f/detA;
    xp[0]=invDetA*((a11*a22-a12*a21)*(-gp[0])+(a02*a21-a01*a22)*(-gp[1])+(a01*a12-a02*a11)*(-gp[2]));
    xp[1]=invDetA*((a12*a20-a10*a22)*(-gp[0])+(a00*a22-a02*a20)*(-gp[1])+(a02*a10-a00*a12)*(-gp[2]));
    xp[2]=invDetA*((a10*a21-a11*a20)*(-gp[0])+(a01*a20-a00*a21)*(-gp[1])+(a00*a11-a01*a10)*(-gp[2]));
}

__global__ void batch_solve_1x1_system_kernel(
    int num_systems, const float* H_scalar, const float* g_scalar, float damping, float* out_x
) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=num_systems)return;
    float h_val=H_scalar[idx]; float g_val=g_scalar[idx]; float h_damped=h_val+damping;
    if(abs(h_damped)<1e-9f){out_x[idx]=0.0f;}else{out_x[idx]=-g_val/h_damped;}
}

// --- LAUNCHER FUNCTIONS ---
namespace NewtonKernels { // Start namespace for launcher definitions

void compute_loss_derivatives_kernel_launcher(
    const torch::Tensor& rendered_image_tensor, const torch::Tensor& gt_image_tensor,
    float lambda_dssim, bool use_l2_loss_term,
    torch::Tensor& out_dL_dc_tensor, torch::Tensor& out_d2L_dc2_diag_tensor
) {
    int H=rendered_image_tensor.size(0); int W=rendered_image_tensor.size(1); int C=rendered_image_tensor.size(2);
    int total_elements=H*W*C;
    const float*rendered_image_ptr=gs::torch_utils::get_const_data_ptr<float>(rendered_image_tensor);
    const float*gt_image_ptr=gs::torch_utils::get_const_data_ptr<float>(gt_image_tensor);
    float*out_dL_dc_ptr=gs::torch_utils::get_data_ptr<float>(out_dL_dc_tensor);
    float*out_d2L_dc2_diag_ptr=gs::torch_utils::get_data_ptr<float>(out_d2L_dc2_diag_tensor);
    auto tensor_opts=rendered_image_tensor.options();
    torch::Tensor dL_dc_l1l2=torch::empty_like(rendered_image_tensor,tensor_opts);
    torch::Tensor d2L_dc2_diag_l1l2=torch::empty_like(rendered_image_tensor,tensor_opts);
    const float N_pixels=static_cast<float>(H*W);
    const float inv_N_pixels=(N_pixels>0)?(1.0f/N_pixels):1.0f;
    compute_l1l2_loss_derivatives_kernel<<<GET_BLOCKS(total_elements),CUDA_NUM_THREADS>>>(
        rendered_image_ptr,gt_image_ptr,use_l2_loss_term,inv_N_pixels,
        gs::torch_utils::get_data_ptr<float>(dL_dc_l1l2),
        gs::torch_utils::get_data_ptr<float>(d2L_dc2_diag_l1l2),
        H,W,C);
    CUDA_CHECK(cudaGetLastError());
    const float C1=0.01f*0.01f; const float C2=0.03f*0.03f;
    torch::Tensor img1_bchw=rendered_image_tensor.unsqueeze(0).permute({0,3,1,2}).contiguous();
    torch::Tensor img2_bchw=gt_image_tensor.unsqueeze(0).permute({0,3,1,2}).contiguous();
    auto ssim_outputs=fusedssim(C1,C2,img1_bchw,img2_bchw,true);
    torch::Tensor ssim_map_bchw=std::get<0>(ssim_outputs);
    torch::Tensor dm_dmu1=std::get<1>(ssim_outputs);
    torch::Tensor dm_dsigma1_sq=std::get<2>(ssim_outputs);
    torch::Tensor dm_dsigma12=std::get<3>(ssim_outputs);
    torch::Tensor dL_dmap_tensor=torch::full_like(ssim_map_bchw,-0.5f);
    torch::Tensor dL_dc_ssim_bchw=fusedssim_backward(C1,C2,img1_bchw,img2_bchw,dL_dmap_tensor,dm_dmu1,dm_dsigma1_sq,dm_dsigma12);
    torch::Tensor dL_dc_ssim_hwc_unnormalized=dL_dc_ssim_bchw.permute({0,2,3,1}).squeeze(0).contiguous();
    torch::Tensor dL_dc_ssim_hwc_normalized=dL_dc_ssim_hwc_unnormalized*inv_N_pixels;
    out_dL_dc_tensor.copy_(dL_dc_l1l2+lambda_dssim*dL_dc_ssim_hwc_normalized);
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
    if (out_H_s_packed.defined()) out_H_s_packed.zero_();
    if (out_g_s.defined()) out_g_s.zero_();
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
    if (out_H_theta.defined()) out_H_theta.zero_();
    if (out_g_theta.defined()) out_g_theta.zero_();
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
    const torch::Tensor& proj_param_for_sh_hess,
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

[end of gsplat/newton_kernels.cu]
