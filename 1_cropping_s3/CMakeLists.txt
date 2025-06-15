cmake_minimum_required(VERSION 3.16.3)
project(snowflake_classifier LANGUAGES CXX CUDA)

# Require C++17, and disable compiler-specific extensions (if possible).

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Set build flags for CXX
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}")

# Set build flags for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELEASE} ${CMAKE_CUDA_FLAGS_DEBUG}")
set(CMAKE_CUDA_ARCHITECTURES "87")

# Add executables
file(GLOB_RECURSE PROJECT_SOURCE_FILES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")
add_executable(snowflake_classifier ${PROJECT_SOURCE_FILES})

# Set directory and load OpenCV built binaries
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../opencv-install/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

target_include_directories(snowflake_classifier PRIVATE ${PROJECT_SOURCE_DIR}/src /usr/local/cuda-11.4/include)
link_directories(snowflake_classifier /usr/local/cuda-11.4/lib64)

target_link_libraries(snowflake_classifier PRIVATE
                                            ${OpenCV_LIBS}
                                            stdc++fs
                                            #  culibos -- Only needed for static cuda linking
                                            metis_static
                                            cusolver
                                            cudart
                                            cublas
                                            cublasLt
                                            cusparse   
                                            )