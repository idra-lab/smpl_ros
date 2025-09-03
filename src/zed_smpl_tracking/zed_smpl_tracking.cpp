// smpl_publisher_main.cpp
#include <atomic>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include "smpl_msgs/msg/smpl.hpp"

#include "ClientPublisher.hpp"
#include "GLViewer.hpp"
#include "zed_utils.hpp"

static Eigen::Vector3f quatToRotVec(const Eigen::Quaterniond &q_in) {
  Eigen::Vector4f q;
  q(0) = q_in.x();
  q(1) = q_in.y();
  q(2) = q_in.z();
  q(3) = q_in.w();
  float n = q.norm();
  if (n < 1e-8f)
    return Eigen::Vector3f::Zero();
  q /= n;

  float x = q(0), y = q(1), z = q(2), w = std::clamp(q(3), -1.0f, 1.0f);

  float angle = 2.0f * std::acos(w);
  float s = std::sqrt(1.0f - w * w);

  Eigen::Vector3f axis;
  if (s < 1e-6f)
    axis = Eigen::Vector3f(x, y, z) * 2.0f;
  else
    axis = Eigen::Vector3f(x / s, y / s, z / s);

  if (angle > M_PI) {
    angle = 2.0f * M_PI - angle;
    axis = -axis;
  }

  return axis * angle;
}

// ---- SMPL -> ZED mapping
static const std::map<int, int> SMPL_TO_ZED = {
    {0, 0},   {1, 18},  {2, 19},  {3, 1},   {4, 20},  {5, 21},
    {6, 2},   {7, 22},  {8, 23},  {9, 3},   {10, 28}, {11, 29},
    {12, 4},  {13, 10}, {14, 11}, {15, 5},  {16, 12}, {17, 13},
    {18, 14}, {19, 15}, {20, 16}, {21, 17}, {22, 30}, {23, 31},
};

// ---- Helper: retrieve quaternion from ZED fused body
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
Eigen::Quaterniond getZEDGlobalQuaternion(const sl::Bodies &bodies, int body_idx) {
  const sl::BodyData &body = bodies.body_list[body_idx];
  auto q = body.global_root_orientation;
  Eigen::Quaterniond out_q;
  out_q.w() = q.w;
  out_q.x() = q.x;
  out_q.y() = q.y;
  out_q.z() = q.z;
  return out_q;
}
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("smpl_publisher_node");
  // Create publisher for custom SMPL message
  auto publisher =
      node->create_publisher<smpl_msgs::msg::Smpl>("/smpl_params", 10);

#ifdef _SL_JETSON_
  const bool isJetson = true;
#else
  const bool isJetson = false;
#endif

  constexpr sl::COORDINATE_SYSTEM COORDINATE_SYSTEM =
      sl::COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP;
  constexpr sl::UNIT UNIT = sl::UNIT::METER;

  std::string path = "/home/nardi/smpl_ros/zed_calib3.json";
  auto configurations =
      sl::readFusionConfigurationFile(path, COORDINATE_SYSTEM, UNIT);
  if (configurations.empty())
    return EXIT_FAILURE;

  Trigger trigger;
  std::vector<ClientPublisher> clients(configurations.size());
  int id_ = 0, gpu_id = 0, nb_gpu = 0;
  if (!isJetson)
    cudaGetDeviceCount(&nb_gpu);

  std::map<int, std::string> svo_files;
  for (auto conf : configurations) {
    if (conf.communication_parameters.getType() ==
        sl::CommunicationParameters::COMM_TYPE::INTRA_PROCESS) {
      gpu_id = id_ % nb_gpu;
      auto state = clients[id_].open(conf.input_type, &trigger, gpu_id);
      if (!state)
        continue;
      if (conf.input_type.getType() == sl::InputType::INPUT_TYPE::SVO_FILE)
        svo_files[id_] = conf.input_type.getConfiguration();
      id_++;
    }
  }
  for (auto &it : clients)
    it.start();

  sl::InitFusionParameters init_params;
  init_params.coordinate_units = UNIT;
  init_params.coordinate_system = COORDINATE_SYSTEM;
  init_params.verbose = true;
  sl::Resolution resolution(1280, 720);
  init_params.maximum_working_resolution = resolution;

  sl::Fusion fusion;
  fusion.init(init_params);

  std::vector<sl::CameraIdentifier> cameras;
  for (auto &it : configurations) {
    sl::CameraIdentifier uuid(it.serial_number);
    auto state = fusion.subscribe(uuid, it.communication_parameters, it.pose,
                                  it.override_gravity);
    if (state == sl::FUSION_ERROR_CODE::SUCCESS)
      cameras.push_back(uuid);
  }
  if (cameras.empty())
    return EXIT_FAILURE;

  sl::BodyTrackingFusionParameters body_fusion_init_params;
  body_fusion_init_params.enable_tracking = true;
  body_fusion_init_params.enable_body_fitting = !isJetson;
  fusion.enableBodyTracking(body_fusion_init_params);

  sl::BodyTrackingFusionRuntimeParameters body_tracking_runtime_parameters;
  body_tracking_runtime_parameters.skeleton_minimum_allowed_keypoints = 7;
  body_tracking_runtime_parameters.skeleton_minimum_allowed_camera =
      cameras.size() / 2.;

  GLViewer viewer;
  viewer.init(argc, argv);

  sl::Bodies fused_bodies; // reuse to avoid copy / double free
  std::map<sl::CameraIdentifier, sl::Bodies> camera_raw_data;
  sl::FusionMetrics metrics;
  std::map<sl::CameraIdentifier, sl::Mat> views;
  std::map<sl::CameraIdentifier, sl::Mat> pointClouds;

  // --- ROS spinning in background thread ---
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  std::atomic<bool> exec_running{true};
  std::thread ros_spin_thread([&]() {
    exec.spin();
    exec_running = false;
  });

  // --- Main ZED + SMPL loop ---
  while (rclcpp::ok() && viewer.isAvailable()) {
    trigger.notifyZED();

    if (fusion.process() == sl::FUSION_ERROR_CODE::SUCCESS) {

      // Retrieve fused bodies
      if (fusion.retrieveBodies(fused_bodies,
                                body_tracking_runtime_parameters) ==
          sl::FUSION_ERROR_CODE::SUCCESS) {
      }

      if (!fused_bodies.body_list.empty()) {
        const int body_idx = 0;

        smpl_msgs::msg::Smpl msg; // Custom SMPL message

        // --- Global orientation (joint 0) ---
        Eigen::Quaterniond q_global =
            getZEDGlobalQuaternion(fused_bodies, body_idx);
        Eigen::Vector3f rod_global = quatToRotVec(q_global);
        msg.global_orient[0] = rod_global.x();
        msg.global_orient[1] = rod_global.y();
        msg.global_orient[2] = rod_global.z();

        // --- Body joints 1-23 ---
        for (int j = 1; j <= 23; ++j) {
          Eigen::Quaterniond q =
              getZEDLocalQuaternion(fused_bodies, body_idx, SMPL_TO_ZED.at(j));
          Eigen::Vector3f rod = quatToRotVec(q);
          msg.body_pose[(j - 1) * 3 + 0] = rod.x();
          msg.body_pose[(j - 1) * 3 + 1] = rod.y();
          msg.body_pose[(j - 1) * 3 + 2] = rod.z();
        }

        // --- Translation from first keypoint ---
        const auto &kp0 = fused_bodies.body_list[body_idx].keypoint[0];
        msg.transl[0] = kp0.x;
        msg.transl[1] = kp0.y;
        msg.transl[2] = kp0.z;

        // Publish the message
        publisher->publish(msg);

        RCLCPP_INFO(node->get_logger(), "Published SMPL params");
      }
      //   --- Safely retrieve per-camera data ---
      for (auto &id : cameras) {

        // Retrieve per-camera bodies
        sl::Bodies temp_camera_bodies;
        if (fusion.retrieveBodies(temp_camera_bodies,
                                  body_tracking_runtime_parameters,
                                  id) == sl::FUSION_ERROR_CODE::SUCCESS) {
          camera_raw_data[id] = std::move(temp_camera_bodies);
        }

        // Camera pose
        sl::Pose pose;
        if (fusion.getPosition(pose, sl::REFERENCE_FRAME::WORLD, id,
                               sl::POSITION_TYPE::RAW) ==
            sl::POSITIONAL_TRACKING_STATE::OK) {
          viewer.setCameraPose(id.sn, pose.pose_data);
        }

        // Retrieve camera images safely
        auto state_view = fusion.retrieveImage(views[id], id, resolution);
        auto state_pc = fusion.retrieveMeasure(
            pointClouds[id], id, sl::MEASURE::XYZBGRA, resolution);

        if (state_view == sl::FUSION_ERROR_CODE::SUCCESS &&
            state_pc == sl::FUSION_ERROR_CODE::SUCCESS) {
          viewer.updateCamera(id.sn, views[id], pointClouds[id]);
        }
      }

      //   Fusion metrics
      fusion.getProcessMetrics(metrics);
    }

    // Update the 3D viewer
    viewer.updateBodies(fused_bodies, camera_raw_data, metrics);
  }
  // --- Shutdown ---
  trigger.running = false;
  trigger.notifyZED();
  for (auto &it : clients)
    it.stop();
  fusion.close();

  exec.cancel();
  if (ros_spin_thread.joinable())
    ros_spin_thread.join();

  rclcpp::shutdown();
  return 0;
}
