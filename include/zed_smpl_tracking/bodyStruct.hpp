#ifndef BODYSTRUCT_HPP
#define BODYSTRUCT_HPP
#include <Eigen/Dense>
// define the Body structure with minimal fields from sl::Body
typedef struct Body {
  Eigen::Quaterniond global_orientation;   // root/global orientation
  Eigen::Vector3d root_position;           // root translation

  std::array<Eigen::Quaterniond, 24> local_orient; // local joint orientations
  std::array<Eigen::Vector3d, 24> keypoints;       // keypoints in body space

  Body() {
    local_orient.fill(Eigen::Quaterniond::Identity());
    keypoints.fill(Eigen::Vector3d::Zero());
    global_orientation = Eigen::Quaterniond::Identity();
    root_position = Eigen::Vector3d::Zero();
  }
};
#endif // BODYSTRUCT_HPP