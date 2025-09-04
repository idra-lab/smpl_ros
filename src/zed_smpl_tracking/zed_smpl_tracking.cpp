// smpl_publisher_main.cpp
#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "smpl_msgs/msg/smpl.hpp"
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>

#include "zed_smpl_tracking/ClientPublisher.hpp"
#include "zed_smpl_tracking/zed_utils.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("smpl_publisher_node");
  // Create publisher for custom SMPL message
  auto publisher =
      node->create_publisher<smpl_msgs::msg::Smpl>("/smpl_params", 10);
  // --- ROS spinning in background thread ---
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  std::atomic<bool> exec_running{true};
  std::thread ros_spin_thread([&]() {
    exec.spin();
    exec_running = false;
  });
  RCLCPP_INFO(node->get_logger(), "ROS spinning thread started.");
  // ----------------------------------------
  constexpr sl::COORDINATE_SYSTEM COORDINATE_SYSTEM =
      sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  // Transform from SMPL(x left, y up, z forward) to Camera (x fward, y left, z
  // up)
  Eigen::Matrix4d T_SMPL_TO_ROS = smpl_to_ros_transform();

  constexpr sl::UNIT UNIT = sl::UNIT::METER;

  std::string path = "/home/nardi/smpl_ros/zed_calib3.json";
  auto configurations =
      sl::readFusionConfigurationFile(path, COORDINATE_SYSTEM, UNIT);
  if (configurations.empty())
    return EXIT_FAILURE;

    RCLCPP_INFO(node->get_logger(), "Starting ZED SMPL tracking...");
  Trigger trigger;
  std::vector<ClientPublisher> clients(configurations.size());
  int id_ = 0, gpu_id = 0, nb_gpu = 0;
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
  RCLCPP_INFO(node->get_logger(), "%ld ZED cameras connected.", cameras.size());
  sl::BodyTrackingFusionParameters body_fusion_init_params;
  body_fusion_init_params.enable_tracking = true;
  body_fusion_init_params.enable_body_fitting = true;
  fusion.enableBodyTracking(body_fusion_init_params);

  sl::BodyTrackingFusionRuntimeParameters body_tracking_runtime_parameters;
  body_tracking_runtime_parameters.skeleton_minimum_allowed_keypoints = 7;
  body_tracking_runtime_parameters.skeleton_minimum_allowed_camera =
      cameras.size() / 2.;

  sl::Bodies fused_bodies; // reuse to avoid copy / double free
  std::map<sl::CameraIdentifier, sl::Bodies> camera_raw_data;
  sl::FusionMetrics metrics;
  std::map<sl::CameraIdentifier, sl::Mat> views;
  std::map<sl::CameraIdentifier, sl::Mat> pointClouds;

  // Prepare the fixed rotation camera -> SMPL as both matrix and quaternion.
  // Camera: X right, Y down, Z forward
  // SMPL:   X left,  Y up,   Z forward
  // This is equivalent to a 180deg rotation around Z (diag(-1,-1,1)).

  while (rclcpp::ok()) {
    trigger.notifyZED();
    if (fusion.process() != sl::FUSION_ERROR_CODE::SUCCESS)
      continue;
    if (fusion.retrieveBodies(fused_bodies, body_tracking_runtime_parameters) !=
        sl::FUSION_ERROR_CODE::SUCCESS)
      continue;
    if (fused_bodies.body_list.empty())
      continue;

    auto msg = build_smpl_msg(fused_bodies, 0, T_SMPL_TO_ROS, SMPL_TO_ZED);
    publisher->publish(msg);
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
