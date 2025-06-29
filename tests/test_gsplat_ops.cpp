#include "Ops.h"
#include "core/debug_utils.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <torch/torch.h>

class GsplatOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        device = torch::kCUDA;
    }

    torch::Device device{torch::kCPU};
};

TEST_F(GsplatOpsTest, RelocationTest) {
    torch::manual_seed(42);

    // Test data matching Python test setup
    int N = 100;
    auto opacities = torch::rand({N}, device) * 0.8f + 0.1f; // [0.1, 0.9]
    auto scales = torch::rand({N, 3}, device) * 0.5f + 0.1f; // [0.1, 0.6]

    // Create ratios as in Python - must be int32!
    auto ratios = torch::randint(1, 10, {N}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Create binomial coefficients
    const int n_max = 51;
    auto binoms = torch::zeros({n_max, n_max}, torch::kFloat32);
    auto binoms_accessor = binoms.accessor<float, 2>();
    for (int n = 0; n < n_max; ++n) {
        for (int k = 0; k <= n; ++k) {
            float binom = 1.0f;
            for (int i = 0; i < k; ++i) {
                binom *= static_cast<float>(n - i) / static_cast<float>(i + 1);
            }
            binoms_accessor[n][k] = binom;
        }
    }
    binoms = binoms.to(device);

    // Test relocation function
    auto [new_opacities, new_scales] = gsplat::relocation(
        opacities,
        scales,
        ratios,
        binoms,
        n_max);

    // Basic sanity checks
    EXPECT_EQ(new_opacities.sizes(), opacities.sizes());
    EXPECT_EQ(new_scales.sizes(), scales.sizes());
    EXPECT_FALSE(new_opacities.isnan().any().item<bool>());
    EXPECT_FALSE(new_scales.isnan().any().item<bool>());

    // Values should be in reasonable ranges
    EXPECT_TRUE((new_opacities >= 0).all().item<bool>());
    EXPECT_TRUE((new_opacities <= 1).all().item<bool>());
    EXPECT_TRUE((new_scales > 0).all().item<bool>());
}

TEST_F(GsplatOpsTest, QuatScaleToCovarPreciGradientTest) {
    torch::manual_seed(42);

    int N = 100;
    auto quats = torch::randn({N, 4}, device);
    auto scales = torch::rand({N, 3}, device) * 0.1f;

    quats.set_requires_grad(true);
    scales.set_requires_grad(true);

    // Forward pass
    auto [covars, precis] = gsplat::quat_scale_to_covar_preci_fwd(
        quats,
        scales,
        true, // compute_covar
        true, // compute_preci
        false // triu
    );

    // Create gradients
    auto v_covars = torch::randn_like(covars);
    auto v_precis = torch::randn_like(precis) * 0.01f; // Small gradient for precis

    // Backward pass
    auto [v_quats, v_scales] = gsplat::quat_scale_to_covar_preci_bwd(
        quats,
        scales,
        false, // triu
        v_covars,
        v_precis);

    // Check gradients are valid
    EXPECT_TRUE(v_quats.defined());
    EXPECT_TRUE(v_scales.defined());
    EXPECT_FALSE(v_quats.isnan().any().item<bool>());
    EXPECT_FALSE(v_scales.isnan().any().item<bool>());
    EXPECT_FALSE(v_quats.isinf().any().item<bool>());
    EXPECT_FALSE(v_scales.isinf().any().item<bool>());
}

TEST_F(GsplatOpsTest, SphericalHarmonicsGradientTest) {
    torch::manual_seed(42);

    // Test with different SH degrees like Python
    std::vector<int> sh_degrees = {0, 1, 2, 3};

    for (int sh_degree : sh_degrees) {
        int N = 1000;
        int K = (sh_degree + 1) * (sh_degree + 1);

        auto coeffs = torch::randn({N, K, 3}, device);
        auto dirs = torch::randn({N, 3}, device);
        auto masks = torch::ones({N}, torch::TensorOptions().dtype(torch::kBool).device(device));

        coeffs.set_requires_grad(true);
        dirs.set_requires_grad(true);

        // Forward
        auto colors = gsplat::spherical_harmonics_fwd(sh_degree, dirs, coeffs, masks);

        // Check forward pass
        EXPECT_EQ(colors.sizes(), torch::IntArrayRef({N, 3}));
        EXPECT_FALSE(colors.isnan().any().item<bool>());
        EXPECT_FALSE(colors.isinf().any().item<bool>());

        // Backward
        auto v_colors = torch::randn_like(colors);
        auto [v_coeffs, v_dirs] = gsplat::spherical_harmonics_bwd(
            K, sh_degree, dirs, coeffs, masks,
            v_colors, sh_degree > 0 // compute_dirs_grad only for degree > 0
        );

        // Check backward pass
        EXPECT_EQ(v_coeffs.sizes(), coeffs.sizes());
        EXPECT_FALSE(v_coeffs.isnan().any().item<bool>());

        if (sh_degree > 0) {
            EXPECT_EQ(v_dirs.sizes(), dirs.sizes());
            EXPECT_FALSE(v_dirs.isnan().any().item<bool>());
        }
    }
}

TEST_F(GsplatOpsTest, ProjectionEWATest) {
    torch::manual_seed(42);

    // Setup matching Python test
    int N = 100;
    int C = 2;
    int width = 640, height = 480;

    auto means = torch::randn({N, 3}, device);
    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto scales = torch::rand({N, 3}, device) * 0.1f;
    auto opacities = torch::rand({N}, device);

    auto viewmats = torch::eye(4, device).unsqueeze(0).repeat({C, 1, 1});
    auto Ks = torch::tensor({{300.0f, 0.0f, 320.0f},
                             {0.0f, 300.0f, 240.0f},
                             {0.0f, 0.0f, 1.0f}},
                            device)
                  .unsqueeze(0)
                  .repeat({C, 1, 1});

    // Empty covars tensor
    auto empty_covars = torch::empty({0, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Test projection
    auto [radii, means2d, depths, conics, compensations] = gsplat::projection_ewa_3dgs_fused_fwd(
        means,
        empty_covars,
        quats,
        scales,
        opacities,
        viewmats,
        Ks,
        width,
        height,
        0.3f,     // eps2d
        0.01f,    // near_plane
        10000.0f, // far_plane
        0.0f,     // radius_clip
        false,    // calc_compensations
        gsplat::CameraModelType::PINHOLE);

    // Check outputs
    EXPECT_EQ(radii.sizes(), torch::IntArrayRef({C, N, 2}));
    EXPECT_EQ(means2d.sizes(), torch::IntArrayRef({C, N, 2}));
    EXPECT_EQ(depths.sizes(), torch::IntArrayRef({C, N}));
    EXPECT_EQ(conics.sizes(), torch::IntArrayRef({C, N, 3}));

    // Check for valid values
    EXPECT_FALSE(means2d.isnan().any().item<bool>());
    EXPECT_FALSE(depths.isnan().any().item<bool>());
    EXPECT_FALSE(conics.isnan().any().item<bool>());

    // At least some Gaussians should be visible
    auto valid = (radii > 0).all(-1);
    EXPECT_GT(valid.sum().item<int64_t>(), 0);
}

TEST_F(GsplatOpsTest, RasterizationPipelineTest) {
    torch::manual_seed(42);

    // Simple scene setup matching Python
    int N = 100;
    int width = 256, height = 256;
    int tile_size = 16;

    // Create test Gaussians
    auto means = torch::randn({N, 3}, device) * 2.0f;
    auto quats = torch::randn({N, 4}, device);
    quats = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto scales = torch::rand({N, 3}, device) * 0.5f;
    auto opacities = torch::rand({N}, device);
    auto colors = torch::rand({N, 3}, device);

    // Camera setup
    auto viewmat = torch::eye(4, device).unsqueeze(0);
    auto K = torch::tensor({{200.0f, 0.0f, 128.0f},
                            {0.0f, 200.0f, 128.0f},
                            {0.0f, 0.0f, 1.0f}},
                           device)
                 .unsqueeze(0);

    auto empty_covars = torch::empty({0, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Project
    auto [radii, means2d, depths, conics, compensations] = gsplat::projection_ewa_3dgs_fused_fwd(
        means,
        empty_covars,
        quats,
        scales,
        opacities,
        viewmat,
        K,
        width,
        height,
        0.3f,
        0.01f,
        1000.0f,
        0.0f,
        false,
        gsplat::CameraModelType::PINHOLE);

    // Tile intersection
    int tile_width = (width + tile_size - 1) / tile_size;
    int tile_height = (height + tile_size - 1) / tile_size;

    auto empty_orders = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto empty_tiles_per_gauss = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    auto [tiles_per_gauss, isect_ids, flatten_ids] = gsplat::intersect_tile(
        means2d, radii, depths,
        empty_orders,
        empty_tiles_per_gauss,
        1, tile_size, tile_width, tile_height,
        true // sort
    );

    auto isect_offsets = gsplat::intersect_offset(isect_ids, 1, tile_width, tile_height);
    isect_offsets = isect_offsets.reshape({1, tile_height, tile_width});

    // Prepare for rasterization
    colors = colors.unsqueeze(0);
    opacities = opacities.unsqueeze(0);
    auto background = torch::zeros({1, 3}, device);
    auto empty_masks = torch::empty({0}, torch::TensorOptions().dtype(torch::kBool).device(device));

    // Rasterize
    auto [render_colors, render_alphas, last_ids] = gsplat::rasterize_to_pixels_3dgs_fwd(
        means2d,
        conics,
        colors,
        opacities,
        background,
        empty_masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids);

    // Check output
    EXPECT_EQ(render_colors.sizes(), torch::IntArrayRef({1, height, width, 3}));
    EXPECT_EQ(render_alphas.sizes(), torch::IntArrayRef({1, height, width, 1}));

    // Check values are valid
    EXPECT_TRUE((render_colors >= 0).all().item<bool>());
    EXPECT_TRUE((render_colors <= 1).all().item<bool>());
    EXPECT_TRUE((render_alphas >= 0).all().item<bool>());
    EXPECT_TRUE((render_alphas <= 1).all().item<bool>());
    EXPECT_FALSE(render_colors.isnan().any().item<bool>());
    EXPECT_FALSE(render_alphas.isnan().any().item<bool>());
}


// Helper function for CPU-based Morton code calculation (for test verification)
uint32_t expandBits_cpu(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

uint32_t morton3D_cpu(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t xx = expandBits_cpu(x);
    uint32_t yy = expandBits_cpu(y);
    uint32_t zz = expandBits_cpu(z);
    return xx | (yy << 1) | (zz << 2);
}

uint32_t compute_expected_morton_code(float x, float y, float z,
                                      const gsplat::vec3& world_min,
                                      const gsplat::vec3& world_max) {
    float norm_x = (x - world_min.x) / (world_max.x - world_min.x);
    float norm_y = (y - world_min.y) / (world_max.y - world_min.y);
    float norm_z = (z - world_min.z) / (world_max.z - world_min.z);

    uint32_t morton_x = static_cast<uint32_t>(std::clamp(norm_x * 1023.0f, 0.0f, 1023.0f));
    uint32_t morton_y = static_cast<uint32_t>(std::clamp(norm_y * 1023.0f, 0.0f, 1023.0f));
    uint32_t morton_z = static_cast<uint32_t>(std::clamp(norm_z * 1023.0f, 0.0f, 1023.0f));

    return morton3D_cpu(morton_x, morton_y, morton_z);
}


TEST_F(GsplatOpsTest, MortonCodeTest) {
    torch::manual_seed(42);

    // Define some sample 3D means
    auto means3d_tensor = torch::tensor({
        {0.0f, 0.0f, 0.0f},    // Point 1
        {1.0f, 1.0f, 1.0f},    // Point 2
        {0.5f, 0.5f, 0.5f},    // Point 3
        {-1.0f, -1.0f, -1.0f}, // Point 4 (outside, should clamp)
        {2.0f, 0.25f, 0.75f}   // Point 5 (partially outside)
    }, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Define world bounds
    gsplat::vec3 world_min_val(0.0f, 0.0f, 0.0f);
    gsplat::vec3 world_max_val(1.0f, 1.0f, 1.0f);

    auto world_min_tensor = torch::tensor({world_min_val.x, world_min_val.y, world_min_val.z}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto world_max_tensor = torch::tensor({world_max_val.x, world_max_val.y, world_max_val.z}, torch::TensorOptions().dtype(torch::kFloat32).device(device));


    // Call the CUDA Morton code function (assuming it's in gsplat namespace)
    // Need to include MortonCodes.h in this test file, or ensure Ops.h includes it or declares the wrapper.
    // For now, let's assume gsplat::compute_morton_codes_tensor is available.
    // This function is defined in gsplat/MortonCodes.cpp but needs to be declared in a header included by Ops.h or directly here.
    // Let's add a temporary declaration here if not in Ops.h yet.
    // namespace gsplat { at::Tensor compute_morton_codes_tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&); }

    at::Tensor morton_codes_output;
    // Check if the function is declared, if not, this won't compile.
    // It should be declared in MortonCodes.h and MortonCodes.cpp should be linked.
    // The C++ wrapper `compute_morton_codes_tensor` is in `gsplat` namespace.
    // We need to add its declaration to `gsplat/Ops.h` or include `gsplat/MortonCodes.h` here.
    // Let's assume it will be added to Ops.h for now.
    // To make this test self-contained for now, I'll call the launcher if the wrapper is not yet exposed via Ops.h
    // However, the wrapper `gsplat::compute_morton_codes_tensor` is the target.
    // For now, to proceed, I might need to call the launcher if the wrapper isn't in Ops.h
    // For the actual test, we'd want to call the gsplat:: C++ API function.

    // Placeholder: Add declaration for gsplat::compute_morton_codes_tensor if not in Ops.h
    // This is typically handled by CMake linking and Ops.h including relevant function declarations.
    // I will assume Ops.h is updated to include the declaration from MortonCodes.h or MortonCodes.cpp's function.
    morton_codes_output = gsplat::compute_morton_codes_tensor(means3d_tensor, world_min_tensor, world_max_tensor);


    ASSERT_EQ(morton_codes_output.numel(), 5);
    ASSERT_EQ(morton_codes_output.dtype(), torch::kInt32); // As per our C++ wrapper, PyTorch kInt is int32_t

    auto output_accessor = morton_codes_output.accessor<int32_t, 1>();

    // Calculate expected values
    std::vector<uint32_t> expected_codes;
    auto means_cpu = means3d_tensor.to(torch::kCPU);
    auto means_acc = means_cpu.accessor<float, 2>();

    for (int i = 0; i < means_cpu.size(0); ++i) {
        expected_codes.push_back(compute_expected_morton_code(
            means_acc[i][0], means_acc[i][1], means_acc[i][2],
            world_min_val, world_max_val
        ));
    }

    // Compare
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(static_cast<uint32_t>(output_accessor[i]), expected_codes[i]) << "Mismatch at index " << i;
    }

    // Test with empty input
    auto empty_means = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto empty_codes = gsplat::compute_morton_codes_tensor(empty_means, world_min_tensor, world_max_tensor);
    EXPECT_EQ(empty_codes.numel(), 0);
}


TEST_F(GsplatOpsTest, TighterBoundingBoxTest) {
    torch::manual_seed(42);

    int N = 1; // One Gaussian
    int C = 1; // One camera
    int width = 640, height = 480;

    // Gaussian centered, pointing down Z axis in camera space
    auto means_tensor = torch::tensor({{0.0f, 0.0f, 3.0f}}, torch::TensorOptions().dtype(torch::kFloat32).device(device)); // [N,3]

    // Define scales and quaternion for a rotated ellipse
    // Scales: major axis = 0.2, minor axis = 0.1, depth_axis = 0.1
    // Rotate by 45 degrees around Z-axis (viewing axis)
    // q = [cos(pi/8), 0, 0, sin(pi/8)] -> rotation of pi/4 around Z
    float angle_rad = M_PI / 4.0f; // 45 degrees
    auto quats_tensor = torch::tensor({{std::cos(angle_rad / 2.0f), 0.0f, 0.0f, std::sin(angle_rad / 2.0f)}},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(device)); // [N,4]
    auto scales_tensor = torch::tensor({{0.2f, 0.1f, 0.1f}}, torch::TensorOptions().dtype(torch::kFloat32).device(device)); // [N,3]
    auto opacities_tensor = torch::tensor({1.0f}, torch::TensorOptions().dtype(torch::kFloat32).device(device)); // [N]

    // Standard camera
    auto viewmats_tensor = torch::eye(4, device).unsqueeze(0); // [C,4,4] (identity view matrix, means are in camera space)
    auto Ks_tensor = torch::tensor({{{500.0f, 0.0f, static_cast<float>(width)/2.0f},
                                   {0.0f, 500.0f, static_cast<float>(height)/2.0f},
                                   {0.0f, 0.0f, 1.0f}}},
                                   torch::TensorOptions().dtype(torch::kFloat32).device(device)); // [C,3,3]

    // Call projection
    auto [radii, means2d, depths, conics, compensations] = gsplat::projection_ewa_3dgs_fused_fwd(
        means_tensor,
        torch::optional<at::Tensor>(), // covars (use quats/scales)
        quats_tensor,
        scales_tensor,
        opacities_tensor,
        viewmats_tensor,
        Ks_tensor,
        width, height,
        0.0f,    // eps2d - set to 0 to isolate covariance effect
        0.1f,    // near_plane
        100.0f,  // far_plane
        0.0f,    // radius_clip
        false,   // calc_compensations
        gsplat::CameraModelType::PINHOLE
    );

    ASSERT_EQ(radii.sizes(), torch::IntArrayRef({C, N, 2}));
    auto radii_acc = radii.accessor<int32_t, 3>();
    int32_t radius_x_new = radii_acc[0][0][0];
    int32_t radius_y_new = radii_acc[0][0][1];

    // --- Calculate expected old radii (axis-aligned based on diagonal of projected covar) ---
    // This requires getting the covar2d that was used internally by the projection.
    // For simplicity, we'll re-calculate parts of the projection logic here on CPU/Torch for verification.
    // 1. World to Camera (already in camera space in this test)
    auto means_cam = means_tensor[0]; // {0,0,3}
    // 2. 3D Covariance from quat/scale
    auto R_mat = gsplat::detail::quat_to_rotmat(
        gsplat::vec4(quats_tensor[0][0].item<float>(), quats_tensor[0][1].item<float>(),
                     quats_tensor[0][2].item<float>(), quats_tensor[0][3].item<float>())
    );
    auto S_mat = glm::diagonal3x3(gsplat::vec3(scales_tensor[0][0].item<float>(), scales_tensor[0][1].item<float>(), scales_tensor[0][2].item<float>()));
    auto cov3D_world_g = R_mat * S_mat * S_mat * glm::transpose(R_mat); // gsplat::mat3

    // View matrix is identity, so cov3D_cam = cov3D_world_g
    auto cov3D_cam_g = cov3D_world_g;

    // 3. Perspective Projection of covariance
    float fx = Ks_tensor[0][0][0].item<float>();
    float fy = Ks_tensor[0][1][1].item<float>();
    float X = means_cam[0].item<float>();
    float Y = means_cam[1].item<float>();
    float Z = means_cam[2].item<float>();
    float Zinv = 1.0f / Z;
    float Zinv2 = Zinv * Zinv;

    gsplat::mat3x2 J; // Jacobian of perspective projection
    J[0][0] = fx * Zinv; J[0][1] = 0.0f;
    J[1][0] = 0.0f;      J[1][1] = fy * Zinv;
    J[2][0] = -fx * X * Zinv2; J[2][1] = -fy * Y * Zinv2;

    gsplat::mat2 covar2d_g = glm::transpose(J) * cov3D_cam_g * J; // This is glm::transpose(J) * V * J for math convention
                                                              // In gsplat code it's J * V * Jt where J is (2,3)
                                                              // Let's use the gsplat convention for J (2x3)
    glm::mat2x3 J_gsplat_conv(fx * Zinv, 0.0f, -fx * X * Zinv2,
                              0.0f, fy * Zinv, -fy * Y * Zinv2);

    covar2d_g = J_gsplat_conv * cov3D_cam_g * glm::transpose(J_gsplat_conv);
    // covar2d_g should have no blur (eps2d=0)

    float extend_f = 3.33f; // Default extend factor if opacity is 1.0 and no compensation
    // With opacity = 1.0f, ratio = 1.0f / ALPHA_THRESHOLD. If ALPHA_THRESHOLD is 1/255, ratio = 255.
    // extend_factor = min(3.33f, sqrtf(max(0.1f, 2.0f * __logf(255.0f))))
    // log(255) approx 5.54. sqrt(2*5.54) = sqrt(11.08) = 3.328. So extend_f is ~3.328
    float ratio_op = opacities_tensor[0].item<float>() / (1.f / 255.f);
    if (ratio_op > 1.0f) {
         extend_f = std::min(extend_f, sqrtf(std::max(0.1f, 2.0f * logf(ratio_op))));
    } else {
         extend_f = std::min(extend_f, 1.0f);
    }


    float old_radius_x = std::ceil(extend_f * std::sqrt(covar2d_g[0][0]));
    float old_radius_y = std::ceil(extend_f * std::sqrt(covar2d_g[1][1]));

    // --- Calculate expected new radii (tighter AABB) ---
    float A_g = covar2d_g[0][0];
    float B_g = covar2d_g[0][1];
    float C_g = covar2d_g[1][1];
    float new_radius_x_expected, new_radius_y_expected;

    if (std::abs(B_g) < 1e-5f) {
        new_radius_x_expected = std::ceil(extend_f * std::sqrt(A_g));
        new_radius_y_expected = std::ceil(extend_f * std::sqrt(C_g));
    } else {
        float T_g = A_g + C_g;
        float D_sqrt_val_g = (A_g - C_g) * (A_g - C_g) + 4.0f * B_g * B_g;
        float D_sqrt_g = std::sqrt(std::max(0.0f, D_sqrt_val_g));
        float lambda1_g = (T_g + D_sqrt_g) / 2.0f;
        float lambda2_g = (T_g - D_sqrt_g) / 2.0f;

        float cos_phi_sq_g, sin_phi_sq_g;
         if (std::abs(A_g - C_g) < 1e-5f && std::abs(B_g) < 1e-5f) {
             cos_phi_sq_g = 1.0f; sin_phi_sq_g = 0.0f;
        } else if (std::abs(B_g) < 1e-5f) {
             cos_phi_sq_g = 1.0f; sin_phi_sq_g = 0.0f;
        } else {
            float cos2phi_g = (A_g - C_g) / D_sqrt_g;
            cos_phi_sq_g = (1.0f + cos2phi_g) / 2.0f;
            sin_phi_sq_g = (1.0f - cos2phi_g) / 2.0f;
        }

        float var_x_aabb_g = lambda1_g * cos_phi_sq_g + lambda2_g * sin_phi_sq_g;
        float var_y_aabb_g = lambda1_g * sin_phi_sq_g + lambda2_g * cos_phi_sq_g;

        new_radius_x_expected = std::ceil(extend_f * std::sqrt(std::max(0.0f, var_x_aabb_g)));
        new_radius_y_expected = std::ceil(extend_f * std::sqrt(std::max(0.0f, var_y_aabb_g)));
    }

    EXPECT_LE(radius_x_new, old_radius_x);
    EXPECT_LE(radius_y_new, old_radius_y);
    EXPECT_EQ(radius_x_new, static_cast<int32_t>(new_radius_x_expected));
    EXPECT_EQ(radius_y_new, static_cast<int32_t>(new_radius_y_expected));

    // Example: For 45 deg rotation, major axis 0.2, minor 0.1 (before Z scaling)
    // s_x = 0.2, s_y = 0.1. Z = 3. fx=fy=500.
    // Projected scales approx: sx_proj = 0.2 * 500/3 = 33.33, sy_proj = 0.1 * 500/3 = 16.67
    // Rotated by 45 deg.
    // old_radius_x based on var_x of transformed covar.
    // new_radius_x based on rotated ellipse projection.
    // For a 45-degree rotated ellipse with semi-axes 'a' and 'b', the AABB half-widths are:
    // W = H = sqrt(0.5*(a^2+b^2))
    // Scaled by extend_f.
    // Here a = extend_f * sx_proj, b = extend_f * sy_proj (if lambda1, lambda2 were directly sx_proj^2, sy_proj^2)
    // This is a sanity check - the detailed calculation above is more accurate.
    float a_test = extend_f * (500.0f/Z) * scales_tensor[0][0].item<float>(); // major axis projected
    float b_test = extend_f * (500.0f/Z) * scales_tensor[0][1].item<float>(); // minor axis projected
    float expected_aabb_halfwidth_45deg = std::ceil(std::sqrt(0.5f * (a_test*a_test + b_test*b_test)));

    // This simple expectation is only if the principal axes of the 3D Gaussian are aligned with world axes
    // before the 45deg Z rotation. Our setup has scales along local axes, then rotated.
    // The full calculation above is more robust.
    // For the given scales (0.2, 0.1) and 45deg rotation, lambda1 and lambda2 will be related to (0.2^2) and (0.1^2)
    // and cos_phi_sq and sin_phi_sq will be ~0.5 if the projection doesn't shear much.
    // So var_x_aabb approx 0.5*(lambda1+lambda2).
    // Let's print values if they mismatch.
    if (radius_x_new != static_cast<int32_t>(new_radius_x_expected) || radius_y_new != static_cast<int32_t>(new_radius_y_expected)) {
        std::cout << "New radii: (" << radius_x_new << ", " << radius_y_new << ")" << std::endl;
        std::cout << "Expected new: (" << new_radius_x_expected << ", " << new_radius_y_expected << ")" << std::endl;
        std::cout << "Old radii: (" << old_radius_x << ", " << old_radius_y << ")" << std::endl;
        std::cout << "Covar2D_g: [[" << covar2d_g[0][0] << ", " << covar2d_g[0][1] << "], [" << covar2d_g[1][0] << ", " << covar2d_g[1][1] << "]]" << std::endl;
        std::cout << "Extend factor used in test: " << extend_f << std::endl;
    }

}
