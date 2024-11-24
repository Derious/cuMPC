cmake_minimum_required (VERSION 2.8.11)
project (gpt-cuda LANGUAGES C CXX CUDA)
set(NAME "gpt-cuda")

set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# Set CUDA compile flags
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets --expt-relaxed-constexpr -diag-suppress=20012")

find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/emp-base.cmake)

FIND_PACKAGE(emp-ot REQUIRED)
INCLUDE_DIRECTORIES(${EMP-OT_INCLUDE_DIRS})


FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})

FIND_PACKAGE(OpenSSL REQUIRED)


install(FILES cmake/emp-agmpc-config.cmake DESTINATION cmake/)
install(DIRECTORY emp-agmpc DESTINATION include/)

ENABLE_TESTING()
ADD_SUBDIRECTORY(test/test_mpc)
ADD_SUBDIRECTORY(test/test_CORE)

# 添加CUDA相关配置
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 添加cuBLAS
find_library(CUBLAS_LIBRARIES cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
if(NOT CUBLAS_LIBRARIES)
    message(FATAL_ERROR "cuBLAS library not found")
endif()

# 设置CUDA编译选项
set(CMAKE_CUDA_STANDARD 16)  # 或者更高版本，根据需要
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_85")  # 根据您的GPU架构调整

# 允许在CUDA文件中使用C++特性
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()