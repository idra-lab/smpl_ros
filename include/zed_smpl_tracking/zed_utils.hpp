#pragma once
#include <Eigen/Dense>
#include <atomic>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <map>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sl/Camera.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <thread>
#include <vector>

// SMPL to ROS homogenous transformation of coordinates
inline Eigen::Matrix4d smpl_to_ros_transform() {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) << 0, 1, 0, 0, 0, 1, 1, 0, 0;
  return T;
}
inline Eigen::Matrix4d ros_to_image_transform() {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  return T;
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

// Converts quaternion to rotation vector (axis-angle)
// The output rotation vector is in the range [-pi, pi]
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

// ---- Helper: retrieve quaternion from ZED fused body  ----
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
// Note: ZED local orientations are relative to the parent joint, so to get the
// absolute orientation you need to chain-multiply the quaternions up to the
// root.
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

// ---- Build SMPL message from ZED fused body and apply coordinate transforms
// ----
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
    // check for NaNs and set them to zero
    if (std::isnan(kp_smpl.x()) || std::isnan(kp_smpl.y()) ||
        std::isnan(kp_smpl.z())) {
      kp_smpl.x() = 0.0;
      kp_smpl.y() = 0.0;
      kp_smpl.z() = 0.0;
    }
    msg.keypoints[j * 3 + 0] = kp_smpl.x();
    msg.keypoints[j * 3 + 1] = kp_smpl.y();
    msg.keypoints[j * 3 + 2] = kp_smpl.z();
  }

  return msg;
}
// ---- Merge multiple point clouds into one ----
inline std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
mergePointClouds(
    const std::vector<std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>>
        &pcs) {
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> merged;

  for (const auto &pc : pcs) {
    merged.insert(merged.end(), pc.begin(), pc.end());
  }

  return merged;
}

Eigen::Matrix4d slTransformToEigen(const sl::Transform &T) {
  Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();

  // Rotation part
  sl::Matrix3f r = T.getRotationMatrix(); // returns sl::Matrix3f
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      mat(i, j) = static_cast<double>(r(i, j));

  // Translation part
  sl::Translation t = T.getTranslation();
  mat(0, 3) = static_cast<double>(t.x);
  mat(1, 3) = static_cast<double>(t.y);
  mat(2, 3) = static_cast<double>(t.z);

  return mat;
}
static void broadcastStaticCameras(
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster,
    const std::vector<Eigen::Matrix4d> &T_cams_extrinsics, std::vector<int> cam_ids,
    const std::string &parent_frame = "map") {
    
  int i = 0;
  for (const auto &T : T_cams_extrinsics) {
    std::string sn = std::to_string(cam_ids[i]);
    i++;
    // create transform message
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = rclcpp::Clock().now();
    t.header.frame_id = parent_frame;
    t.child_frame_id = "cam" + std::to_string(i) + "_" + sn;

    // translation
    t.transform.translation.x = T(0, 3);
    t.transform.translation.y = T(1, 3);
    t.transform.translation.z = T(2, 3);

    // rotation
    Eigen::Quaterniond q(T.block<3, 3>(0, 0));
    t.transform.rotation.x = q.x();
    t.transform.rotation.y = q.y();
    t.transform.rotation.z = q.z();
    t.transform.rotation.w = q.w();

    // send immediately
    tf_broadcaster->sendTransform(t);
  }
}
void publishMergedPointCloud(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
        &merged_cloud,
    const std::string &frame_id = "map") {
  sensor_msgs::msg::PointCloud2 cloud_msg;
  cloud_msg.header.stamp = rclcpp::Clock().now();
  cloud_msg.header.frame_id = frame_id;
  cloud_msg.height = 1;
  cloud_msg.width = static_cast<uint32_t>(merged_cloud.size());
  cloud_msg.is_dense = true;
  cloud_msg.point_step = 16; // 4 floats: x,y,z + rgb
  cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
  cloud_msg.data.resize(cloud_msg.row_step);

  // Define fields
  cloud_msg.fields.resize(4);
  cloud_msg.fields[0].name = "x";
  cloud_msg.fields[0].offset = 0;
  cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[0].count = 1;
  cloud_msg.fields[1].name = "y";
  cloud_msg.fields[1].offset = 4;
  cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[1].count = 1;
  cloud_msg.fields[2].name = "z";
  cloud_msg.fields[2].offset = 8;
  cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[2].count = 1;
  cloud_msg.fields[3].name = "rgb";
  cloud_msg.fields[3].offset = 12;
  cloud_msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
  cloud_msg.fields[3].count = 1;

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_rgb(cloud_msg, "rgb");

  for (const auto &p : merged_cloud) {
    const Eigen::Vector3d &pt = p.first;
    const Eigen::Vector3d &col = p.second;

    *iter_x = static_cast<float>(pt.x());
    *iter_y = static_cast<float>(pt.y());
    *iter_z = static_cast<float>(pt.z());

    uint8_t r =
        static_cast<uint8_t>(std::min(1.0, std::max(0.0, col.x())) * 255.0);
    uint8_t g =
        static_cast<uint8_t>(std::min(1.0, std::max(0.0, col.y())) * 255.0);
    uint8_t b =
        static_cast<uint8_t>(std::min(1.0, std::max(0.0, col.z())) * 255.0);

    uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                    static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));

    *iter_rgb = *reinterpret_cast<float *>(&rgb);

    ++iter_x;
    ++iter_y;
    ++iter_z;
    ++iter_rgb;
  }

  pub->publish(cloud_msg);
}
