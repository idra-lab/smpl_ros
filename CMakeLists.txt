cmake_minimum_required(VERSION 3.10)
project(fitting_smpl)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

# In the top-level CMakeLists.txt of the consumer project
list(APPEND CMAKE_PREFIX_PATH "/home/nardi/torchure_smplx/build")

find_package(torchure_smplx REQUIRED)
find_package(Torch REQUIRED)
find_package(Open3D REQUIRED)

# Create your executable
add_executable(fitting_smpl fitting_smpl.cpp)
target_compile_definitions(fitting_smpl PRIVATE USE_OPEN3D=ON)


# Link against the smplx library
message(STATUS "Fitting includes: ${FITTING_INCLUDES}")
target_link_libraries(fitting_smpl PRIVATE torchure_smplx::smplx
    Open3D::Open3D
)
