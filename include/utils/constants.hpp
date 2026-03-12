#pragma once
#include <Eigen/Dense>

inline Eigen::Matrix4d smpl_to_ros_transform() {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) << 0, 1, 0, 0, 0, 1, 1, 0, 0;
  return T;
}

// SMPL to ROS homogenous transformation of coordinates
inline Eigen::Matrix4d ros_to_image_transform() {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  return T;
}

// ---- SMPL -> ZED mapping
static constexpr std::array<int, 24> SMPL_TO_ZED = {
    0,  // 0
    18, // 1
    19, // 2
    1,  // 3
    20, // 4
    21, // 5
    2,  // 6
    22, // 7
    23, // 8
    3,  // 9
    24, // 10
    25, // 11
    4,  // 12
    10, // 13
    11, // 14
    5,  // 15
    12, // 16
    13, // 17
    14, // 18
    15, // 19
    16, // 20
    17, // 21
    30, // 22
    31  // 23
};

// SMPL parents (standard 24-joint kinematic tree). -1 is root.
static const int SMPL_PARENTS[24] = {
    -1,
    0,  // 1
    0,  // 2
    0,  // 3
    1,  // 4
    2,  // 5
    3,  // 6
    4,  // 7
    5,  // 8
    6,  // 9
    7,  // 10
    8,  // 11
    9,  // 12
    9,  // 13
    9,  // 14
    12, // 15
    13, // 16
    14, // 17
    16, // 18
    17, // 19
    18, // 20
    19, // 21
    20, // 22
    21  // 23
};
// load json with betas