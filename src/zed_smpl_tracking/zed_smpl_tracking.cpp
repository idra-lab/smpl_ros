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

#include "tf2_ros/static_transform_broadcaster.h"
#include "yolov8_seg.h"
#include "zed_smpl_tracking/ClientPublisher.hpp"
#include "zed_smpl_tracking/GLViewer.hpp"
#include "zed_smpl_tracking/fuseSkeletons.hpp"
#include "zed_smpl_tracking/zed_utils.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("smpl_publisher_node");

  // ------------------ Declare ROS2 Parameters ------------------
  node->declare_parameter<std::string>("calibration_file",
                                       "/home/nardi/smpl_ros/zed_calib3.json");
  node->declare_parameter<std::string>("yolo_model_path",
                                       "/home/nardi/smpl_ros/yolov8x-seg.onnx");
  auto tf_static_broadcaster_ =
      std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);

  node->declare_parameter<int>("max_width", 1280);
  node->declare_parameter<int>("max_height", 720);
  node->declare_parameter<bool>("publish_point_cloud", true);

  std::string calib_file = node->get_parameter("calibration_file").as_string();
  std::string yolo_model_path =
      node->get_parameter("yolo_model_path").as_string();
  int max_width = node->get_parameter("max_width").as_int();
  int max_height = node->get_parameter("max_height").as_int();
  bool publish_point_cloud =
      node->get_parameter("publish_point_cloud").as_bool();

  auto smpl_pub =
      node->create_publisher<smpl_msgs::msg::Smpl>("/smpl_params", 10);
  auto cloud_pub =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("/human_cloud", 10);

  // --- ROS spinning in background thread ---
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  std::atomic<bool> exec_running{true};
  std::thread ros_spin_thread([&]() {
    exec.spin();
    exec_running = false;
  });
  RCLCPP_INFO(node->get_logger(), "ROS spinning thread started.");

  // ------------------ ZED + SMPL Setup ------------------
  // calibration must be done in IMAGE frame but we want data in ROS frame ->
  // ZED automatically converts the read JSON extrinsics to ROS frame
  constexpr sl::COORDINATE_SYSTEM ROS_COORDINATE_SYSTEM =
      sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  constexpr sl::UNIT UNIT = sl::UNIT::METER;
  Eigen::Matrix4d T_SMPL_TO_ROS = smpl_to_ros_transform();

  auto configurations =
      sl::readFusionConfigurationFile(calib_file, ROS_COORDINATE_SYSTEM, UNIT);

  if (configurations.empty()) {
    RCLCPP_ERROR(node->get_logger(), "No ZED configurations found.");
    return EXIT_FAILURE;
  }

  RCLCPP_INFO(node->get_logger(), "Starting ZED SMPL tracking...");
  Trigger trigger;
  std::vector<ClientPublisher> clients(configurations.size());
  int id_ = 0, gpu_id = 0, nb_gpu = 0;
  cudaGetDeviceCount(&nb_gpu);

  for (auto conf : configurations) {
    if (conf.communication_parameters.getType() ==
        sl::CommunicationParameters::COMM_TYPE::INTRA_PROCESS) {
      gpu_id = id_ % nb_gpu;
      if (!clients[id_].open(conf.input_type, &trigger, gpu_id))
        continue;
      id_++;
    }
  }

  GLViewer viewer;
  viewer.init(argc, argv);

  for (auto &client : clients)
    client.start();

  // Fusion initialization
  sl::InitFusionParameters init_params;
  init_params.coordinate_units = UNIT;
  init_params.coordinate_system = ROS_COORDINATE_SYSTEM;
  init_params.verbose = true;
  sl::Resolution resolution(max_width, max_height);
  init_params.maximum_working_resolution = resolution;

  sl::Fusion fusion;
  fusion.init(init_params);

  // Subscribe to cameras
  std::vector<Eigen::Matrix4d> T_cams_extrinsics;

  std::vector<sl::CameraIdentifier> cameras;
  std::vector<int> cam_ids;
  for (auto &conf : configurations) {
    auto T = slTransformToEigen(conf.pose);
    T_cams_extrinsics.push_back(T);
    sl::CameraIdentifier uuid(conf.serial_number);
    fusion.updatePose(uuid, conf.pose);
    if (fusion.subscribe(uuid, conf.communication_parameters, conf.pose,
                         conf.override_gravity) ==
        sl::FUSION_ERROR_CODE::SUCCESS)
      cameras.push_back(uuid);
    cam_ids.push_back(conf.serial_number);
  }
  broadcastStaticCameras(tf_static_broadcaster_, T_cams_extrinsics, cam_ids,
                         "map");

  // Ensure that fusion poses are set

  if (cameras.empty()) {
    RCLCPP_ERROR(node->get_logger(), "No cameras connected!");
    return EXIT_FAILURE;
  }
  RCLCPP_INFO(node->get_logger(), "%ld ZED cameras connected.", cameras.size());

  // Enable body tracking and fitting
  sl::BodyTrackingFusionParameters body_fusion_init_params;
  body_fusion_init_params.enable_tracking = true;
  body_fusion_init_params.enable_body_fitting = true;
  fusion.enableBodyTracking(body_fusion_init_params);

  sl::BodyTrackingFusionRuntimeParameters body_tracking_runtime_parameters;
  body_tracking_runtime_parameters.skeleton_minimum_allowed_keypoints = 7;
  body_tracking_runtime_parameters.skeleton_minimum_allowed_camera =
      cameras.size() / 2.0;

  //   --- Safely retrieve per-camera data ---
  std::map<sl::CameraIdentifier, sl::Bodies> camera_raw_data;
  sl::FusionMetrics metrics;
  std::map<sl::CameraIdentifier, sl::Mat> views;
  std::map<sl::CameraIdentifier, sl::Mat> pointClouds;

  sl::Bodies fused_bodies;
  std::vector<sl::BodyData> raw_bodies_vector;

  Yolov8Seg yolov8Seg;
  cv::dnn::Net yolo_net;
  if (publish_point_cloud) {
    yolo_net = LoadYOLOModel(yolov8Seg, yolo_model_path);
  }

  // ------------------ Main loop ------------------
  while (rclcpp::ok() && viewer.isAvailable()) {
    trigger.notifyZED();

    std::vector<std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>> pcs(
        clients.size());
    if (publish_point_cloud) {
      for (int i = 0; i < cameras.size(); i++) {
        pcs[i] = clients[i].getFilteredPointCloud(T_cams_extrinsics[i],
                                                  yolo_net, yolov8Seg);
      }
      auto merged_cloud = mergePointClouds(pcs);
      publishMergedPointCloud(cloud_pub, merged_cloud, "map");
    }

    // SMPL processing
    // if (fusion.process() != sl::FUSION_ERROR_CODE::SUCCESS) {
    //   RCLCPP_WARN(node->get_logger(), "Could not process fusion step");
    //   continue;
    // }

    // This produces a wrong result even though singular cameras have correct
    // bodies
    // if (fusion.retrieveBodies(fused_bodies, body_tracking_runtime_parameters)
    // !=
    //     sl::FUSION_ERROR_CODE::SUCCESS) {
    //   RCLCPP_WARN(node->get_logger(), "Could not retrieve bodies");
    //   continue;
    // }
    // if (fused_bodies.body_list.empty()) {
    //   RCLCPP_WARN(node->get_logger(), "No bodies found");
    //   continue;
    // }
    // Prepare per-camera BodyData vector
    for (size_t i = 0; i < cameras.size(); i++) {
      sl::Bodies detected_bodies;
      clients[i].zed.retrieveBodies(detected_bodies);
      if (detected_bodies.body_list.empty()) {
        continue;
      }
      // extract only the first body (TODO: get most centered?)
      raw_bodies_vector.push_back(detected_bodies.body_list[0]);
      RCLCPP_INFO(node->get_logger(), "Body rot %f %f %f %f",
                  detected_bodies.body_list[0].global_root_orientation.w,
                  detected_bodies.body_list[0].global_root_orientation.x,
                  detected_bodies.body_list[0].global_root_orientation.y,
                  detected_bodies.body_list[0].global_root_orientation.z);

    }
    // Extract vector of Body converting from sl::Bodies to custom Body struct
    std::vector<Body> bodies = extractBodyData(raw_bodies_vector, SMPL_TO_ZED);

    // Merge the bodies into a single fused BodyData
    if (!raw_bodies_vector.empty()) {
      Body fusedBody = mergeBodiesWithExtrinsics(bodies, T_cams_extrinsics);

      // Build and publish SMPL message
      auto msg = buildSMPLMessage(fusedBody, T_SMPL_TO_ROS);
      smpl_pub->publish(msg);
      raw_bodies_vector.clear();
    }
  }

  // ------------------ Shutdown ------------------
  trigger.running = false;
  trigger.notifyZED();
  for (auto &client : clients)
    client.stop();
  fusion.close();

  exec.cancel();
  if (ros_spin_thread.joinable())
    ros_spin_thread.join();

  rclcpp::shutdown();
  return 0;
}
