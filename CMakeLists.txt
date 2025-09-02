cmake_minimum_required(VERSION 3.10)
project(fitting_smpl)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

# Torchure SMPLX
list(APPEND CMAKE_PREFIX_PATH "/home/nardi/torchure_smplx/build")
find_package(torchure_smplx REQUIRED)

# LibTorch
find_package(Torch REQUIRED)
# Open3D
find_package(Open3D REQUIRED)

# ROS 2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
# PCL
find_package(PCL REQUIRED COMPONENTS io filters)


# Executable
add_executable(fitting_smpl src/fitting_smpl_ros.cpp)

target_include_directories(fitting_smpl PRIVATE
  ${PCL_INCLUDE_DIRS} ${Open3D_INCLUDE_DIRS} include
)

target_link_libraries(fitting_smpl
  torchure_smplx::smplx
  ${TORCH_LIBRARIES}
  ${PCL_LIBRARIES}
  ${Open3D_LIBRARIES}
)

ament_target_dependencies(fitting_smpl
  rclcpp
  visualization_msgs
  geometry_msgs
  sensor_msgs
)

# Install
install(TARGETS fitting_smpl
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
