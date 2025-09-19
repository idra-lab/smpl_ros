cmake_minimum_required(VERSION 3.10)
project(smpl_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)


# Torchure SMPLX
find_package(torchure_smplx REQUIRED)
# LibTorch
find_package(Torch REQUIRED)
# Open3D
find_package(Open3D REQUIRED)
# Eigen
find_package(Eigen3 REQUIRED)
# ROS 2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(smpl_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
# ZED SMPL Tracking
find_package(ZED REQUIRED)
find_package(CUDA REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
SET(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(OpenCV)

set(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

# Yolo
add_library(yolov8 src/zed_smpl_tracking/yolov8/yolov8_seg.cpp src/zed_smpl_tracking/yolov8/yolov8_utils.cpp)
target_include_directories(yolov8 PUBLIC include/zed_smpl_tracking/yolov8 ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov8 ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY}
  ${OpenCV_LIBRARIES})

# Executable
add_executable(fitting_smpl src/body_fitting/fitting_smpl_ros.cpp)
add_executable(smpl_ros_viewer src/smpl_ros_viewer/smpl_ros_viewer.cpp)
add_executable(zed_smpl_tracking
  src/zed_smpl_tracking/zed_smpl_tracking.cpp
  src/zed_smpl_tracking/ClientPublisher.cpp
)

# Fitting SMPL
target_include_directories(fitting_smpl PRIVATE
  ${Open3D_INCLUDE_DIRS} include
)
target_link_libraries(fitting_smpl
  torchure_smplx::smplx
  ${TORCH_LIBRARIES}
  ${Open3D_LIBRARIES}
)

# Viewer
target_include_directories(smpl_ros_viewer PRIVATE include)
target_link_libraries(smpl_ros_viewer
  torchure_smplx::smplx
  ${TORCH_LIBRARIES}
)


# ZED SMPL Tracking
target_include_directories(zed_smpl_tracking PUBLIC ${CUDA_INCLUDE_DIRS})
target_include_directories(zed_smpl_tracking PUBLIC ${ZED_INCLUDE_DIRS})
target_include_directories(zed_smpl_tracking PUBLIC ${GLEW_INCLUDE_DIRS})
target_include_directories(zed_smpl_tracking PUBLIC ${GLUT_INCLUDE_DIR})
target_include_directories(zed_smpl_tracking PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(zed_smpl_tracking PUBLIC ${OpenCV_INCLUDE_DIRS})
link_directories(zed_smpl_tracking ${ZED_LIBRARY_DIR})
link_directories(zed_smpl_tracking ${CUDA_LIBRARY_DIRS})
link_directories(zed_smpl_tracking ${GLEW_LIBRARY_DIRS})
link_directories(zed_smpl_tracking ${GLUT_LIBRARY_DIRS})
link_directories(zed_smpl_tracking ${OpenGL_LIBRARY_DIRS})
link_directories(zed_smpl_tracking ${OpenCV_LIBRARY_DIRS})
target_link_libraries(zed_smpl_tracking ${ZED_LIBS} ${UTILS_LIB} ${SPECIAL_OS_LIBS} ${OpenCV_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES}
  Eigen3::Eigen
  yolov8
  torchure_smplx::smplx
  ${TORCH_LIBRARIES}
)

ament_target_dependencies(fitting_smpl
  rclcpp
  visualization_msgs
  geometry_msgs
  sensor_msgs
)

# Directly Link message to node
ament_target_dependencies(smpl_ros_viewer
  rclcpp
  visualization_msgs
  smpl_msgs
  geometry_msgs
)
ament_target_dependencies(zed_smpl_tracking
  rclcpp
  smpl_msgs
  sensor_msgs
  tf2_ros
  cv_bridge
)

# Install
install(TARGETS fitting_smpl smpl_ros_viewer zed_smpl_tracking
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
