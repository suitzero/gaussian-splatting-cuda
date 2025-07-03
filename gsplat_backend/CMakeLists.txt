cmake_minimum_required(VERSION 3.18)

if(NOT PROJ_ROOT_DIR)
    get_filename_component(PROJ_ROOT_DIR "${CMAKE_SOURCE_DIR}" ABSOLUTE)
endif()

add_library(gsplat_backend STATIC
    newton_optimizer.cpp  # Corrected path
    newton_kernels.cu    # Corrected path
)
set_target_properties(gsplat_backend PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(gsplat_backend
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"          # For newton_kernels.cuh (now directly in gsplat_backend/)
        "${PROJ_ROOT_DIR}/include"             # For core headers like newton_optimizer.hpp, splat_data.hpp etc.
        "${PROJ_ROOT_DIR}/external/glm"        # For GLM
        "${PROJ_ROOT_DIR}/gsplat"              # For gsplat headers if newton_kernels.cu needs them
    PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(gsplat_backend
    PUBLIC
        glm
    PRIVATE
        ${TORCH_LIBRARIES}
        CUDA::cudart
)

if(CMAKE_CUDA_STANDARD)
    set_target_properties(gsplat_backend PROPERTIES CUDA_STANDARD ${CMAKE_CUDA_STANDARD})
endif()
if(CMAKE_CUDA_STANDARD_REQUIRED)
    set_target_properties(gsplat_backend PROPERTIES CUDA_STANDARD_REQUIRED ${CMAKE_CUDA_STANDARD_REQUIRED})
endif()

if(CMAKE_CXX_STANDARD)
    set_target_properties(gsplat_backend PROPERTIES CXX_STANDARD ${CMAKE_CXX_STANDARD})
endif()
if(CMAKE_CXX_STANDARD_REQUIRED)
    set_target_properties(gsplat_backend PROPERTIES CXX_STANDARD_REQUIRED ${CMAKE_CXX_STANDARD_REQUIRED})
endif()

message(STATUS "Configuring gsplat_backend library with updated file paths.")
