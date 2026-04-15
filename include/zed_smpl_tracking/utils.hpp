#pragma once
#include "bodyStruct.hpp"
#include "utils/constants.hpp"
#include <Eigen/Dense>
#include <atomic>
// #include <geometry_msgs/msg/transform_stamped.hpp>
#include "utils/geom_utils.hpp"
#include <map>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sl/Camera.hpp>
#include <smpl_msgs/msg/fixed_size_image.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <thread>
#include <vector>

#define NUM_BETAS 10
std::vector<double> load_smpl_betas(const std::string &smpl_params_path) {
  std::ifstream smpl_file(smpl_params_path);
  if (!smpl_file.is_open()) {
    RCLCPP_FATAL(rclcpp::get_logger("zed_smpl_tracking"),
                 "Could not open smpl params file %s",
                 smpl_params_path.c_str());
    throw std::runtime_error("Failed to open SMPL params file");
  }

  nlohmann::json smpl_json;
  smpl_file >> smpl_json;

  std::vector<double> betas;
  if (smpl_json.contains("betas") && smpl_json["betas"].is_array()) {
    auto betas_json = smpl_json["betas"];
    if (betas_json.size() != NUM_BETAS) {
      RCLCPP_FATAL(rclcpp::get_logger("zed_smpl_tracking"),
                   "SMPL betas array must contain %d values", NUM_BETAS);
      throw std::runtime_error("Invalid betas size in SMPL params");
    }
    betas.reserve(NUM_BETAS);
    for (size_t i = 0; i < NUM_BETAS; i++) {
      betas.push_back(static_cast<double>(betas_json[i]));
    }
  } else {
    RCLCPP_FATAL(rclcpp::get_logger("zed_smpl_tracking"),
                 "SMPL params missing 'betas' array");
    throw std::runtime_error("Missing betas in SMPL params");
  }

  RCLCPP_INFO(rclcpp::get_logger("zed_smpl_tracking"),
              "Loaded SMPL params from %s", smpl_params_path.c_str());
  std::string betas_str;
  for (const auto &b : betas) {
    betas_str += std::to_string(b) + " ";
  }
  RCLCPP_INFO(rclcpp::get_logger("zed_smpl_tracking"), "Betas: %s",
              betas_str.c_str());
  return betas;
}

sl::RESOLUTION resolutionToEnum(const sl::Resolution &res) {
  if (res.width == 2208 && res.height == 1242)
    return sl::RESOLUTION::HD2K;
  else if (res.width == 1920 && res.height == 1080)
    return sl::RESOLUTION::HD1080;
  else if (res.width == 1280 && res.height == 720)
    return sl::RESOLUTION::HD720;
  else if (res.width == 672 && res.height == 376)
    return sl::RESOLUTION::VGA;
  else
    throw std::runtime_error("Unsupported resolution");
}

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

// ---- Build SMPL message from ZED fused body and apply transforms ----
inline smpl_msgs::msg::Smpl
buildSMPLMessage(const Body &body, const Eigen::Matrix4d &T_ros_from_smpl,
                 const std::vector<double> &betas,
                 const rclcpp::Clock::SharedPtr &clock) {

  static const auto logger = rclcpp::get_logger("zed_smpl_tracking");

  smpl_msgs::msg::Smpl msg;

  // ============================================================
  // VALIDATE & DECOMPOSE T_ros_from_smpl
  // ============================================================

  // Guard: if the basis-change matrix is garbage, every downstream
  // computation will be wrong — bail out with an identity-pose message.
  if (!T_ros_from_smpl.allFinite()) {
    RCLCPP_ERROR(logger, "T_ros_from_smpl contains non-finite values, "
                         "returning default message.");
    msg.header.stamp = clock->now();
    msg.header.frame_id = "map";
    return msg;
  }

  const Eigen::Matrix3d R_basis = T_ros_from_smpl.block<3, 3>(0, 0);

  // For a pure rotation/rigid matrix R^{-1} == R^T — avoid the costly
  // and numerically noisier general inverse().
  const Eigen::Matrix3d R_basis_inv = R_basis.transpose();

  // Project R_basis onto SO(3) if floating-point drift has pushed it off.
  // We reuse the validated copy for all subsequent change-of-basis ops.
  Eigen::Matrix3d R_basis_clean = R_basis;
  if (std::abs(R_basis.determinant() - 1.0) > 1e-4 || !R_basis.allFinite()) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_basis, Eigen::ComputeFullU |
                                                       Eigen::ComputeFullV);
    R_basis_clean = svd.matrixU() * svd.matrixV().transpose();
    if (R_basis_clean.determinant() < 0)
      R_basis_clean = svd.matrixU() *
                      Eigen::DiagonalMatrix<double, 3>(1, 1, -1) *
                      svd.matrixV().transpose();
    RCLCPP_WARN(logger, "R_basis was not a valid rotation matrix; "
                        "projected onto SO(3).");
  }
  const Eigen::Matrix3d R_basis_clean_inv = R_basis_clean.transpose();

  // ============================================================
  // ROOT TRANSFORM  (SMPL world → ROS world)
  // ============================================================

  // Validate global orientation before using it.
  Eigen::Quaterniond q_global = body.global_orientation;
  if (!isValidQuaternion(q_global)) {
    RCLCPP_WARN(logger, "global_orientation is invalid, using Identity.");
    q_global = Eigen::Quaterniond::Identity();
  }
  q_global.normalize();

  // Validate root position.
  Eigen::Vector3d root_pos = body.root_position;
  if (!root_pos.allFinite()) {
    RCLCPP_WARN(logger, "root_position contains non-finite values, "
                        "using zero.");
    root_pos.setZero();
  }

  // Build root transform in SMPL frame.
  Eigen::Matrix4d T_root_smpl = Eigen::Matrix4d::Identity();
  T_root_smpl.block<3, 3>(0, 0) = q_global.toRotationMatrix();
  T_root_smpl.block<3, 1>(0, 3) = root_pos;

  // Change of basis:  T_root_ros = T_smpl_to_ros * T_root_smpl * T_ros_to_smpl
  // Use the explicit rigid inverse instead of .inverse():
  //   T^{-1} = [ R^T  | -R^T * t ]
  //            [  0   |     1    ]
  Eigen::Matrix4d T_ros_from_smpl_inv = Eigen::Matrix4d::Identity();
  T_ros_from_smpl_inv.block<3, 3>(0, 0) = R_basis_clean_inv;
  T_ros_from_smpl_inv.block<3, 1>(0, 3) =
      -R_basis_clean_inv * T_ros_from_smpl.block<3, 1>(0, 3);

  const Eigen::Matrix4d T_root_ros =
      T_ros_from_smpl * T_root_smpl * T_ros_from_smpl_inv;

  // Extract and validate root rotation.
  const Eigen::Matrix3d R_root_ros = T_root_ros.block<3, 3>(0, 0);
  Eigen::Quaterniond q_root_ros;
  if (isValidRotationMatrix(R_root_ros)) {
    q_root_ros = Eigen::Quaterniond(R_root_ros).normalized();
  } else {
    RCLCPP_WARN(logger, "Root rotation after basis change is not in SO(3), "
                        "projecting.");
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_root_ros, Eigen::ComputeFullU |
                                                          Eigen::ComputeFullV);
    Eigen::Matrix3d R_fixed = svd.matrixU() * svd.matrixV().transpose();
    if (R_fixed.determinant() < 0)
      R_fixed = svd.matrixU() * Eigen::DiagonalMatrix<double, 3>(1, 1, -1) *
                svd.matrixV().transpose();
    q_root_ros = Eigen::Quaterniond(R_fixed).normalized();
  }

  const Eigen::Vector3d rvec_root = quatToRotVec(q_root_ros);
  const Eigen::Vector3d t_root = T_root_ros.block<3, 1>(0, 3);

  msg.global_orient[0] = rvec_root.x();
  msg.global_orient[1] = rvec_root.y();
  msg.global_orient[2] = rvec_root.z();

  msg.transl[0] = t_root.x();
  msg.transl[1] = t_root.y();
  msg.transl[2] = t_root.z();

  // ============================================================
  // LOCAL JOINT ROTATIONS  (change of basis, joints 1–23)
  // ============================================================

  // How many joints are actually available — avoid out-of-bounds access.
  const int available_joints = static_cast<int>(body.local_orient.size());

  for (int j = 1; j < 24; ++j) {
    Eigen::Vector3d rvec_local = Eigen::Vector3d::Zero(); // safe default

    if (j < available_joints) {
      Eigen::Quaterniond q_local = body.local_orient[j];

      if (!isValidQuaternion(q_local)) {
        RCLCPP_WARN(logger, "local_orient[%d] is invalid, using Identity.", j);
        q_local = Eigen::Quaterniond::Identity();
      }
      q_local.normalize();

      // Change of basis for a rotation:  R_ros = R_b * R_smpl * R_b^T
      const Eigen::Matrix3d R_local_ros =
          R_basis_clean * q_local.toRotationMatrix() * R_basis_clean_inv;

      // Validate result before converting to rotation vector.
      Eigen::Quaterniond q_local_ros;
      if (isValidRotationMatrix(R_local_ros)) {
        q_local_ros = Eigen::Quaterniond(R_local_ros).normalized();
      } else {
        RCLCPP_WARN(logger,
                    "local_orient[%d] result is not in SO(3), "
                    "projecting.",
                    j);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            R_local_ros, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R_fixed = svd.matrixU() * svd.matrixV().transpose();
        if (R_fixed.determinant() < 0)
          R_fixed = svd.matrixU() * Eigen::DiagonalMatrix<double, 3>(1, 1, -1) *
                    svd.matrixV().transpose();
        q_local_ros = Eigen::Quaterniond(R_fixed).normalized();
      }

      rvec_local = quatToRotVec(q_local_ros);
    } else {
      RCLCPP_WARN_ONCE(logger,
                       "body.local_orient has fewer than 24 entries "
                       "(%d available); missing joints default to zero.",
                       available_joints);
    }

    msg.body_pose[(j - 1) * 3 + 0] = rvec_local.x();
    msg.body_pose[(j - 1) * 3 + 1] = rvec_local.y();
    msg.body_pose[(j - 1) * 3 + 2] = rvec_local.z();
  }

  // ============================================================
  // KEYPOINTS  (point transformation)
  // ============================================================

  const int available_kps = static_cast<int>(body.keypoints.size());

  for (int j = 0; j < 24; ++j) {
    Eigen::Vector3d kp_ros = Eigen::Vector3d::Zero();

    if (j < available_kps) {
      const Eigen::Vector3d &kp = body.keypoints[j];

      if (kp.allFinite()) {
        Eigen::Vector4d kp_h(kp.x(), kp.y(), kp.z(), 1.0);
        Eigen::Vector4d kp_ros_h = T_ros_from_smpl * kp_h;

        // Guard against degenerate homogeneous division.
        if (std::abs(kp_ros_h.w()) > 1e-9 && kp_ros_h.head<3>().allFinite()) {
          kp_ros = kp_ros_h.head<3>() / kp_ros_h.w();
        } else {
          RCLCPP_WARN(logger,
                      "Keypoint %d produced non-finite result "
                      "after transform, setting to zero.",
                      j);
        }
      } else {
        RCLCPP_WARN(logger,
                    "Keypoint %d contains non-finite input, "
                    "setting to zero.",
                    j);
      }
    }

    msg.keypoints[j * 3 + 0] = kp_ros.x();
    msg.keypoints[j * 3 + 1] = kp_ros.y();
    msg.keypoints[j * 3 + 2] = kp_ros.z();
  }

  // ============================================================
  // BETAS  (shape parameters, frame-invariant)
  // ============================================================

  const size_t num_betas_to_copy =
      std::min(betas.size(), static_cast<size_t>(NUM_BETAS));
  if (betas.size() < static_cast<size_t>(NUM_BETAS)) {
    RCLCPP_WARN_ONCE(logger,
                     "betas vector has %zu elements, expected %d; "
                     "missing values default to 0.",
                     betas.size(), NUM_BETAS);
  }

  for (size_t i = 0; i < num_betas_to_copy; ++i)
    msg.betas[i] = betas[i];
  // Remaining slots are already zero-initialised by the message constructor.

  // ============================================================
  // HEADER
  // ============================================================

  msg.header.stamp = clock->now();
  msg.header.frame_id = "map";

  return msg;
}
// ---- Merge multiple point clouds into one ----
inline std::vector<
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>>
mergePointClouds(
    const std::vector<std::vector<
        std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>>> &pcs) {

  std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>>
      merged;

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
    const std::vector<Eigen::Matrix4d> &T_map_cam,
    const std::vector<std::string> &cam_frames,
    const std::string &parent_frame = "map",
    const std::string &image_suffix = "_image") {
  std::vector<geometry_msgs::msg::TransformStamped> transforms;
  rclcpp::Time stamp = rclcpp::Clock().now();

  // Transform ROS -> Image (same for all cameras)
  Eigen::Matrix4d T_img = ros_to_image_transform(); //.inverse();

  for (size_t i = 0; i < cam_frames.size(); ++i) {
    const auto &frame = cam_frames[i];
    const auto &T = T_map_cam[i];

    // -------------------------
    // map -> camera_frame
    // -------------------------
    geometry_msgs::msg::TransformStamped t_map_cam;
    t_map_cam.header.stamp = stamp;
    t_map_cam.header.frame_id = parent_frame;
    t_map_cam.child_frame_id = frame;

    t_map_cam.transform.translation.x = T(0, 3);
    t_map_cam.transform.translation.y = T(1, 3);
    t_map_cam.transform.translation.z = T(2, 3);

    Eigen::Quaterniond q(T.block<3, 3>(0, 0));
    t_map_cam.transform.rotation.x = q.x();
    t_map_cam.transform.rotation.y = q.y();
    t_map_cam.transform.rotation.z = q.z();
    t_map_cam.transform.rotation.w = q.w();

    transforms.push_back(t_map_cam);

    // -------------------------
    // camera_frame -> camera_frame_image
    // -------------------------
    geometry_msgs::msg::TransformStamped t_cam_img;
    t_cam_img.header.stamp = stamp;
    t_cam_img.header.frame_id = frame;
    t_cam_img.child_frame_id = frame + image_suffix;

    t_cam_img.transform.translation.x = T_img(0, 3);
    t_cam_img.transform.translation.y = T_img(1, 3);
    t_cam_img.transform.translation.z = T_img(2, 3);

    Eigen::Quaterniond q_img(T_img.block<3, 3>(0, 0));
    t_cam_img.transform.rotation.x = q_img.x();
    t_cam_img.transform.rotation.y = q_img.y();
    t_cam_img.transform.rotation.z = q_img.z();
    t_cam_img.transform.rotation.w = q_img.w();

    transforms.push_back(t_cam_img);
  }

  tf_broadcaster->sendTransform(transforms);
}

void save_ply(const std::string &filename,
              const std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d,
                                           Eigen::Vector3d>> &pc,
              bool include_normals = false) {

  std::ofstream ply_file(filename);
  if (!ply_file.is_open()) {
    throw std::runtime_error("Could not open PLY file for writing.");
  }

  // Header
  ply_file << "ply\n";
  ply_file << "format ascii 1.0\n";
  ply_file << "element vertex " << pc.size() << "\n";
  ply_file << "property float x\n";
  ply_file << "property float y\n";
  ply_file << "property float z\n";
  ply_file << "property uchar red\n";
  ply_file << "property uchar green\n";
  ply_file << "property uchar blue\n";
  if (include_normals) {
    ply_file << "property float nx\n";
    ply_file << "property float ny\n";
    ply_file << "property float nz\n";
  }
  ply_file << "end_header\n";

  // Write points
  for (const auto &p : pc) {
    const Eigen::Vector3d &pt = std::get<0>(p);
    const Eigen::Vector3d &col = std::get<1>(p);
    const Eigen::Vector3d &normal = std::get<2>(p);

    uint8_t r =
        static_cast<uint8_t>(std::min(1.0, std::max(0.0, col.x())) * 255.0);
    uint8_t g =
        static_cast<uint8_t>(std::min(1.0, std::max(0.0, col.y())) * 255.0);
    uint8_t b =
        static_cast<uint8_t>(std::min(1.0, std::max(0.0, col.z())) * 255.0);

    ply_file << pt.x() << " " << pt.y() << " " << pt.z() << " "
             << static_cast<int>(r) << " " << static_cast<int>(g) << " "
             << static_cast<int>(b);

    if (include_normals) {
      ply_file << " " << normal.x() << " " << normal.y() << " " << normal.z();
    }

    ply_file << "\n";
  }

  ply_file.close();
  std::cout << "Saved " << pc.size() << " points to " << filename << std::endl;
}

void publishPointCloud(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub,
    const std::vector<
        std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>> &cloud,
    const std::string &frame_id = "map", bool include_normals = false) {

  // sensor_msgs::msg::PointCloud2 cloud_msg;
  auto loaned_msg = pub->borrow_loaned_message();
  auto &cloud_msg = loaned_msg.get();
  cloud_msg.header.stamp = rclcpp::Clock().now();
  cloud_msg.header.frame_id = frame_id;
  cloud_msg.height = 1;
  cloud_msg.width = static_cast<uint32_t>(cloud.size());
  cloud_msg.is_dense = true;

  // Determine point step: 4 floats for xyz+rgb, +3 floats if normals included
  cloud_msg.point_step = 16; // xyz(12) + rgb(4)
  if (include_normals)
    cloud_msg.point_step += 12; // nx, ny, nz as floats
  cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;

  cloud_msg.data.resize(cloud_msg.row_step);

  // Define fields
  cloud_msg.fields.clear();
  uint32_t offset = 0;

  auto add_field = [&](const std::string &name) {
    sensor_msgs::msg::PointField field;
    field.name = name;
    field.offset = offset;
    field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field.count = 1;
    cloud_msg.fields.push_back(field);
    offset += 4;
  };

  add_field("x");
  add_field("y");
  add_field("z");
  add_field("rgb");

  if (include_normals) {
    add_field("normal_x");
    add_field("normal_y");
    add_field("normal_z");
  }

  // Iterators
  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<float> iter_rgb(cloud_msg, "rgb");

  sensor_msgs::PointCloud2Iterator<float> iter_nx(cloud_msg, "normal_x");
  sensor_msgs::PointCloud2Iterator<float> iter_ny(cloud_msg, "normal_y");
  sensor_msgs::PointCloud2Iterator<float> iter_nz(cloud_msg, "normal_z");

  sensor_msgs::PointCloud2Iterator<float> *p_nx = nullptr;
  sensor_msgs::PointCloud2Iterator<float> *p_ny = nullptr;
  sensor_msgs::PointCloud2Iterator<float> *p_nz = nullptr;

  if (include_normals) {
    p_nx = new sensor_msgs::PointCloud2Iterator<float>(cloud_msg, "normal_x");
    p_ny = new sensor_msgs::PointCloud2Iterator<float>(cloud_msg, "normal_y");
    p_nz = new sensor_msgs::PointCloud2Iterator<float>(cloud_msg, "normal_z");
  }

  for (const auto &p : cloud) {
    const Eigen::Vector3d &pt = std::get<0>(p);
    const Eigen::Vector3d &col = std::get<1>(p);
    const Eigen::Vector3d &normal = std::get<2>(p);

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

    if (include_normals && p_nx && p_ny && p_nz) {
      *(*p_nx) = static_cast<float>(normal.x());
      *(*p_ny) = static_cast<float>(normal.y());
      *(*p_nz) = static_cast<float>(normal.z());

      ++(*p_nx);
      ++(*p_ny);
      ++(*p_nz);
    }

    ++iter_x;
    ++iter_y;
    ++iter_z;
    ++iter_rgb;
  }
  RCLCPP_INFO(rclcpp::get_logger("zed_smpl_tracking"),
              "Publishing point cloud with %zu points (include_normals=%s)",
              cloud.size(), include_normals ? "true" : "false");

  pub->publish(std::move(loaned_msg));
}
// void publish_image_msg(
//     rclcpp::Publisher<smpl_msgs::msg::FixedSizeImage>::SharedPtr image_pub,
//     const cv::Mat &image, const std::string &frame_id = "map") {
//   if (image.empty())
//     return;
//   // Borrow loaned message
//   auto loaned_msg = image_pub->borrow_loaned_message();
//   auto &msg = loaned_msg.get();

//   int default_width = msg.width;
//   int default_height = msg.height;

//   if (image.cols != default_width || image.rows != default_height) {
//     RCLCPP_ERROR_STREAM(rclcpp::get_logger("image_pub"),
//                         "Image size must be " << default_width << "x"
//                                               << default_height);
//     return;
//   }

//   if (image.type() != CV_16UC1) {
//     RCLCPP_ERROR(rclcpp::get_logger("image_pub"),
//                  "Image must be CV_16UC1 (uint16)!");
//     return;
//   }

//   // Header
//   msg.header.stamp = rclcpp::Clock().now();
//   msg.header.frame_id = frame_id;

//   // Metadata
//   // msg.height = image.rows;
//   // msg.width = image.cols;
//   msg.encoding = 0; // 0 = uint16
//   // msg.is_bigendian = false;
//   msg.step = image.cols * sizeof(uint16_t);

//   // Copy data into fixed-size array
//   std::memcpy(msg.data.data(), image.data, 252672 * sizeof(uint16_t));

//   // Publish
//   image_pub->publish(std::move(loaned_msg));
// }

void publish_image_msg(
    const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& image_pub,
    const cv::Mat& image,
    const std::string& frame_id = "map")
{
  if (image.empty()) {
    RCLCPP_WARN(rclcpp::get_logger("image_pub"), "Empty image, skipping publish.");
    return;
  }

  sensor_msgs::msg::Image msg;

  // Header
  msg.header.stamp = rclcpp::Clock().now();
  msg.header.frame_id = frame_id;

  // Dimensions
  msg.height = image.rows;
  msg.width  = image.cols;

  // Encoding + step
  switch (image.type()) {
    case CV_8UC1:
      msg.encoding = "mono8";
      msg.step = image.cols * sizeof(uint8_t);
      break;

    case CV_8UC3:
      msg.encoding = "bgr8";
      msg.step = image.cols * 3 * sizeof(uint8_t);
      break;

    case CV_16UC1:
      msg.encoding = "16UC1";
      msg.step = image.cols * sizeof(uint16_t);
      break;

    default:
      RCLCPP_ERROR(rclcpp::get_logger("image_pub"),
                   "Unsupported image type: %d", image.type());
      return;
  }

  msg.is_bigendian = false;

  // Copy data
  size_t size = msg.step * msg.height;
  msg.data.resize(size);
  std::memcpy(msg.data.data(), image.data, size);

  // Publish
  image_pub->publish(msg);
}

void publish_depth_msg(
    rclcpp::Publisher<smpl_msgs::msg::FixedSizeImage>::SharedPtr pub,
    const cv::Mat &depth_mat, const std::string &frame_id = "map",
    const float *normals = nullptr) {
  if (depth_mat.empty()) {
    RCLCPP_WARN(rclcpp::get_logger("depth_pub"), "Depth matrix is empty!");
    return;
  }

  if (depth_mat.type() != CV_32FC1) {
    RCLCPP_ERROR(rclcpp::get_logger("depth_pub"),
                 "Depth must be CV_32FC1 (float meters)");
    return;
  }

  const int height = depth_mat.rows;
  const int width = depth_mat.cols;

  // Loaned message (shared memory zero-copy)
  auto loaned_msg = pub->borrow_loaned_message();
  auto &msg = loaned_msg.get();
  int default_width = msg.width;
  int default_height = msg.height;

  if (depth_mat.cols != default_width || depth_mat.rows != default_height) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("image_pub"),
                        "Depth image size must be " << default_width << "x"
                                              << default_height);
    return;
  }

  // Header
  msg.header.stamp = rclcpp::Clock().now();
  msg.header.frame_id = frame_id;

  // Metadata
  // msg.height = height;
  // msg.width = width;
  // msg.encoding = 1; // FLOAT32
  msg.is_bigendian = false;
  msg.step = width * sizeof(float);

  // Copy depth float data
  std::memcpy(msg.data.data(), depth_mat.data, height * width * sizeof(float));

  // Copy normals if available
  if (normals) {
    float *dst = msg.normals.data();
    for (int y = 0; y < height; y++) {
      const float *src =
          normals +
          y * width * 4; // zed saves normals as 4 floats (nx, ny, nz, 0)

      for (int x = 0; x < width; x++) {
        // apply ros -> image transform (x = -y, y = -z, z = x)
        dst[(y * width + x) * 3 + 0] = -src[x * 4 + 1];
        dst[(y * width + x) * 3 + 1] = -src[x * 4 + 2];
        dst[(y * width + x) * 3 + 2] = src[x * 4 + 0];
      }
    }
  } else {
    std::fill(msg.normals.begin(), msg.normals.end(), 0.0f);
  }

  float min_n = 999, max_n = -999;
  for (int i = 0; i < height * width * 3; i++) {
    float v = msg.normals[i];
    if (v < min_n)
      min_n = v;
    if (v > max_n)
      max_n = v;
  }
  pub->publish(std::move(loaned_msg));

  // Debug info
  double minVal, maxVal;
  cv::minMaxLoc(depth_mat, &minVal, &maxVal);
}

// Simple Timer Class for measuring elapsed time
class SimpleTimer {
public:
  void tik() {
    last_ = clock::now();
    running_ = true;
  }

  void tok(const char *label = "Elapsed") {
    if (!running_)
      return;

    auto now = clock::now();
    auto elapsed =
        std::chrono::duration<double, std::milli>(now - last_).count();

    std::cout << label << ": " << elapsed << " ms\n";

    // reset per la prossima misura
    last_ = now;
  }

private:
  using clock = std::chrono::steady_clock;
  clock::time_point last_;
  bool running_ = false;
};

sensor_msgs::msg::CameraInfo
buildCameraInfoMsg(const sl::CalibrationParameters &cam_params,
                   const std::string &frame_id, const int width,
                   const int height) {
  sensor_msgs::msg::CameraInfo msg;
  msg.header.frame_id = frame_id;
  // ---- Intrinsics (K) ----
  msg.k = {cam_params.left_cam.fx,
           0.0,
           cam_params.left_cam.cx,
           0.0,
           cam_params.left_cam.fy,
           cam_params.left_cam.cy,
           0.0,
           0.0,
           1.0};

  // ---- Rectification (R) ----
  msg.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  // ---- Projection (P) ----
  msg.p = {cam_params.left_cam.fx,
           0.0,
           cam_params.left_cam.cx,
           0.0,
           0.0,
           cam_params.left_cam.fy,
           cam_params.left_cam.cy,
           0.0,
           0.0,
           0.0,
           1.0,
           0.0};

  // ---- Distortion ----
  msg.distortion_model = "plumb_bob";
  msg.d = {cam_params.left_cam.disto[0], cam_params.left_cam.disto[1],
           cam_params.left_cam.disto[2], cam_params.left_cam.disto[3],
           cam_params.left_cam.disto[4]};

  return msg;
}
