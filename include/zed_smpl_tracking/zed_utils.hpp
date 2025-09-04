#pragma once
#include <Eigen/Dense>
#include <atomic>
#include <map>
#include <memory>
#include <sl/Camera.hpp>
#include <thread>
#include <vector>

static Eigen::Vector3d quatToRotVec(const Eigen::Quaterniond &q_in) {
  Eigen::Vector4d q;
  q(0) = q_in.x();
  q(1) = q_in.y();
  q(2) = q_in.z();
  q(3) = q_in.w();
  double n = q.norm();
  if (n < 1e-8f)
    return Eigen::Vector3d::Zero();
  q /= n;

  double x = q(0), y = q(1), z = q(2), w = std::clamp(q(3), -1.0, 1.0);

  double angle = 2.0f * std::acos(w);
  double s = std::sqrt(1.0f - w * w);

  Eigen::Vector3d axis;
  if (s < 1e-6f)
    axis = Eigen::Vector3d(x, y, z) * 2.0f;
  else
    axis = Eigen::Vector3d(x / s, y / s, z / s);

  if (angle > M_PI) {
    angle = 2.0f * M_PI - angle;
    axis = -axis;
  }

  Eigen::Vector3d rvec = axis * angle;
  // wrap angles to [-pi, pi]
  for (int i = 0; i < 3; ++i) {
    while (rvec[i] > M_PI) {
      rvec[i] -= 2 * M_PI;
    }

    while (rvec[i] < -M_PI) {
      rvec[i] += 2 * M_PI;
    }
  }
  return rvec;
}

// ---- SMPL -> ZED mapping
static const std::map<int, int> SMPL_TO_ZED = {
    {0, 0},   {1, 18},  {2, 19},  {3, 1},   {4, 20},  {5, 21},
    {6, 2},   {7, 22},  {8, 23},  {9, 3},   {10, 28}, {11, 29},
    {12, 4},  {13, 10}, {14, 11}, {15, 5},  {16, 12}, {17, 13},
    {18, 14}, {19, 15}, {20, 16}, {21, 17}, {22, 30}, {23, 31},
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

// ---- Helper: retrieve quaternion from ZED fused body
Eigen::Quaterniond getZEDGlobalQuaternion(const sl::Bodies &bodies,
                                          int body_idx) {
  const sl::BodyData &body = bodies.body_list[body_idx];
  auto q = body.global_root_orientation;
  Eigen::Quaterniond out_q;
  out_q.w() = q.w;
  out_q.x() = q.x;
  out_q.y() = q.y;
  out_q.z() = q.z;
  return out_q;
}
Eigen::Quaterniond getZEDLocalQuaternion(const sl::Bodies &bodies, int body_idx,
                                         int zed_kp_index) {
  const sl::BodyData &body = bodies.body_list[body_idx];
  auto q = body.local_orientation_per_joint[zed_kp_index];
  Eigen::Quaterniond out_q;
  out_q.w() = q.w;
  out_q.x() = q.x;
  out_q.y() = q.y;
  out_q.z() = q.z;
  return out_q;
}

inline smpl_msgs::msg::Smpl
build_smpl_msg(const sl::Bodies &fused_bodies, int body_idx,
               const Eigen::Matrix4d &T_smpl_to_ros,
               const std::map<int, int> &SMPL_TO_ZED) {
  smpl_msgs::msg::Smpl msg;
  const auto &body = fused_bodies.body_list[body_idx];
  Eigen::Matrix3d R_change = T_smpl_to_ros.block<3, 3>(0, 0);

  // --- Root joint ---
  Eigen::Vector3d root_pos(body.keypoint[0].x, body.keypoint[0].y,
                           body.keypoint[0].z);
  Eigen::Quaterniond root_quat(getZEDGlobalQuaternion(fused_bodies, body_idx));
  Eigen::Matrix4d T_root = Eigen::Matrix4d::Identity();
  T_root.block<3, 3>(0, 0) = root_quat.toRotationMatrix();
  T_root.block<3, 1>(0, 3) = root_pos;
  Eigen::Matrix4d T_root_smpl =
      T_smpl_to_ros * T_root * T_smpl_to_ros.inverse();
  Eigen::Vector3d root_pos_smpl = T_root_smpl.block<3, 1>(0, 3);
  Eigen::Quaterniond root_quat_smpl(T_root_smpl.block<3, 3>(0, 0));
  Eigen::Vector3d root_rvec_smpl = quatToRotVec(root_quat_smpl);

  msg.global_orient[0] = root_rvec_smpl.x();
  msg.global_orient[1] = root_rvec_smpl.y();
  msg.global_orient[2] = root_rvec_smpl.z();
  msg.transl[0] = root_pos_smpl.x();
  msg.transl[1] = root_pos_smpl.y();
  msg.transl[2] = root_pos_smpl.z();

  // --- Local joints ---
  for (int j = 1; j < 24; ++j) {
    Eigen::Quaterniond q_local_ros = Eigen::Quaterniond::Identity();
    auto it = SMPL_TO_ZED.find(j);
    if (it != SMPL_TO_ZED.end()) {
      int zed_idx = it->second;
      q_local_ros = getZEDLocalQuaternion(fused_bodies, body_idx, zed_idx);
    }
    Eigen::Matrix3d R_local_smpl =
        R_change * q_local_ros.toRotationMatrix() * R_change.inverse();
    Eigen::Vector3d rvec_local_smpl =
        quatToRotVec(Eigen::Quaterniond(R_local_smpl));
    msg.body_pose[(j - 1) * 3 + 0] = rvec_local_smpl.x();
    msg.body_pose[(j - 1) * 3 + 1] = rvec_local_smpl.y();
    msg.body_pose[(j - 1) * 3 + 2] = rvec_local_smpl.z();
  }

  // --- Keypoints ---
  for (int j = 0; j < 24; ++j) {
    const auto &kp = body.keypoint.at(SMPL_TO_ZED.at(j));
    Eigen::Vector4d kp_h(kp.x, kp.y, kp.z, 1.0);
    Eigen::Vector4d kp_smpl = T_smpl_to_ros * kp_h;
    msg.keypoints[j * 3 + 0] = kp_smpl.x();
    msg.keypoints[j * 3 + 1] = kp_smpl.y();
    msg.keypoints[j * 3 + 2] = kp_smpl.z();
  }

  return msg;
}

inline Eigen::Matrix4d smpl_to_ros_transform() {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) << 0, 1, 0, 0, 0, 1, 1, 0, 0;
  return T;
}