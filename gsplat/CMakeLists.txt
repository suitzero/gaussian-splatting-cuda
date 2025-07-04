cmake_minimum_required(VERSION 3.24...3.30)

# Get torch from parent
if(NOT DEFINED TORCH_INCLUDE_DIRS)
    find_package(Torch REQUIRED)
endif()

# All gsplat sources together
set(GSPLAT_SOURCES
        # C++ files
        Adam.cpp
        Intersect.cpp
        Null.cpp
        Projection.cpp
        QuatScaleToCovar.cpp
        Rasterization.cpp
        Relocation.cpp
        SphericalHarmonics.cpp

        # CUDA files
        AdamCUDA.cu
        IntersectTile.cu
        NullCUDA.cu
        ProjectionEWA3DGSFused.cu
        ProjectionEWA3DGSPacked.cu
        ProjectionEWASimple.cu
        ProjectionUT3DGSFused.cu
        QuatScaleToCovarCUDA.cu
        RasterizeToIndices3DGS.cu
        RasterizeToPixels3DGSBwd.cu
        RasterizeToPixels3DGSFwd.cu
        RasterizeToPixelsFromWorld3DGSBwd.cu
        RasterizeToPixelsFromWorld3DGSFwd.cu
        RelocationCUDA.cu
        SphericalHarmonicsCUDA.cu

        # Added Newton optimizer and kernel files
        newton_optimizer.cpp
        newton_kernels.cu
)

# One unified library
add_library(gsplat_backend STATIC ${GSPLAT_SOURCES}) # Reverted target name to gsplat_backend

set_target_properties(gsplat_backend PROPERTIES # Reverted target name to gsplat_backend
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(gsplat_backend # Reverted target name to gsplat_backend
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}            # For headers within gsplat/ (e.g., newton_kernels.cuh)
        "${CMAKE_SOURCE_DIR}/include"          # For project root includes (e.g., core/newton_optimizer.hpp)
        "${CMAKE_SOURCE_DIR}/external/glm"     # Added for GLM
        ${TORCH_INCLUDE_DIRS}                  # Already public, good
        PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}            # Added CUDA toolkit includes
        ${Python3_INCLUDE_DIRS}
)

target_link_libraries(gsplat_backend # Reverted target name to gsplat_backend
        PUBLIC
        ${TORCH_LIBRARIES}
        CUDA::cudart
        CUDA::curand
        CUDA::cublas
        glm
)

# Compile options for both CUDA and C++
if(MSVC)
    target_compile_options(gsplat_backend PRIVATE # Reverted target name to gsplat_backend
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G -lineinfo -O0>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 --use_fast_math>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Od /Zi /EHsc /D_DEBUG /DEBUG_BUILD>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2 /EHsc /DNDEBUG /RELEASE_BUILD>
    )
    target_compile_definitions(gsplat_backend PRIVATE $<$<COMPILE_LANGUAGE:CXX>:_CRT_SECURE_NO_WARNINGS>) # Reverted target
else()
    target_compile_options(gsplat_backend PRIVATE # Reverted target name to gsplat_backend
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -g -G -lineinfo>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 --use_fast_math --expt-relaxed-constexpr -diag-suppress=20012,186>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g -DDEBUG_BUILD>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3 -DNDEBUG -DRELEASE_BUILD>
    )
endif()

if(NOT MSVC)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(gsplat_backend PRIVATE DEBUG_BUILD) # Reverted target
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_definitions(gsplat_backend PRIVATE RELEASE_BUILD) # Reverted target
    endif()
endif()
