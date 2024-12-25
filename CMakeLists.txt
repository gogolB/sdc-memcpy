cmake_minimum_required(VERSION 3.25.2)
project(sdc-memcpy LANGUAGES CUDA CXX)

# Set CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# Add CUDA executable
add_executable(sdc-memcpy main.cu)

# Add CUDA libraries (if needed)
# target_link_libraries(my_cuda_app CUDA::cudart)