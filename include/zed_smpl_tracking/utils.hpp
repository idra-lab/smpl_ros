#pragma once
#include "bodyStruct.hpp"
#include "utils/constants.hpp"
#include <Eigen/Dense>
#include <atomic>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <map>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sl/Camera.hpp>
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
buildSMPLMessage(const Body &body, const Eigen::Matrix4d &T_smpl_to_ros,
                 const std::vector<double> &betas) {
  smpl_msgs::msg::Smpl msg;
  Eigen::Matrix3d R_change = T_smpl_to_ros.block<3, 3>(0, 0);

  // --- Root joint ---
  Eigen::Matrix4d T_root = Eigen::Matrix4d::Identity();
  T_root.block<3, 3>(0, 0) = body.global_orientation.toRotationMatrix();
  T_root.block<3, 1>(0, 3) = body.root_position;
  // RCLCPP_INFO_STREAM(rclcpp::get_logger("zed_smpl_tracking"),
  //                    "Root: " << T_root);

  // change of basis to convert SMPL world to ROS world
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
    Eigen::Quaterniond q_local_ros = body.local_orient.at(j).normalized();
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
    const auto &kp = body.keypoints.at(j); // already in SMPL order
    Eigen::Vector4d kp_h(kp.x(), kp.y(), kp.z(), 1.0);
    Eigen::Vector4d kp_smpl = T_smpl_to_ros * kp_h;

    if (std::isnan(kp_smpl.x()) || std::isnan(kp_smpl.y()) ||
        std::isnan(kp_smpl.z())) {
      RCLCPP_WARN(rclcpp::get_logger("zed_smpl_tracking"),
                  "NaN keypoint detected, setting to zero.");
      kp_smpl.head<3>().setZero();
    }

    msg.keypoints[j * 3 + 0] = kp_smpl.x();
    msg.keypoints[j * 3 + 1] = kp_smpl.y();
    msg.keypoints[j * 3 + 2] = kp_smpl.z();
  }
  for (size_t i = 0; i < NUM_BETAS; i++) {
    msg.betas[i] = betas[i];
  }
  msg.header.stamp = rclcpp::Clock().now();
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

  sensor_msgs::msg::PointCloud2 cloud_msg;
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

  pub->publish(cloud_msg);
}

void publish_image_msg(
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub,
    const cv::Mat &image, const std::string &frame_id) {
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock().now();
  header.frame_id = frame_id;
  // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Publishing image with %d
  // channels",
  //             image.channels());
  // cv::cvtColor(image, image, cv::COLOR_BGRA2RGBA);
  auto image_msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
  image_pub->publish(*image_msg);
}

void publish_depth_msg(
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub,
    const cv::Mat &depth, const std::string &frame_id) {

  if (depth.empty())
    return;

  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock().now();
  header.frame_id = frame_id;

  std::string encoding;

  if (depth.type() == CV_32FC1)
    encoding = "32FC1";
  else if (depth.type() == CV_16UC1)
    encoding = "16UC1";
  else {
    RCLCPP_WARN(rclcpp::get_logger("depth_pub"), "Unsupported depth type");
    return;
  }

  auto msg = cv_bridge::CvImage(header, encoding, depth).toImageMsg();
  depth_pub->publish(*msg);
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

void publishCameraInfo(
    const rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr &pub,
    sl::Camera &zed, const rclcpp::Time &stamp, const std::string &frame_id,
    int width, int height) {
  auto cam_params =
      zed.getCameraInformation().camera_configuration.calibration_parameters;

  sensor_msgs::msg::CameraInfo msg;

  msg.header.stamp = stamp;
  msg.header.frame_id = frame_id;

  msg.width = width;
  msg.height = height;

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

  pub->publish(msg);
}
