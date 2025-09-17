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
#include "utils/json.hpp"
#include "yolov8_seg.h"
#include "zed_smpl_tracking/ClientPublisher.hpp"
#include "zed_smpl_tracking/fuseSkeletons.hpp"
#include "zed_smpl_tracking/utils.hpp"

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
  // used only with fusion API
  node->declare_parameter<int>("max_width", 1280);
  node->declare_parameter<int>("max_height", 720);
  node->declare_parameter<bool>("publish_point_cloud", true);
  node->declare_parameter<std::string>("point_cloud_output_file", "human_cloud.ply");
  node->declare_parameter<std::string>("smpl_params_file", "");

  std::string calib_file = node->get_parameter("calibration_file").as_string();
  RCLCPP_INFO(node->get_logger(), "Using calibration file: %s",
              calib_file.c_str());
  std::string yolo_model_path =
      node->get_parameter("yolo_model_path").as_string();
  RCLCPP_INFO(node->get_logger(), "Using YOLO model file: %s",
              yolo_model_path.c_str());
  int max_width = node->get_parameter("max_width").as_int();
  int max_height = node->get_parameter("max_height").as_int();
  bool publish_point_cloud =
      node->get_parameter("publish_point_cloud").as_bool();
  std::string smpl_params_path =
      node->get_parameter("smpl_params_file").as_string();
  std::vector<double> betas(10, 0.0);
  if (smpl_params_path == "") {
    RCLCPP_INFO(node->get_logger(),
                "No .json params file specified: SMPL betas set to zero");
  } else {
    betas = load_smpl_betas(smpl_params_path);
  }
  std::string pc_output_file =
      node->get_parameter("point_cloud_output_file").as_string();
  RCLCPP_INFO(node->get_logger(), "Point cloud will be saved to: %s",
              pc_output_file.c_str());

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

  std::vector<sl::Bodies> detected_bodies;
  detected_bodies.resize(cameras.size());

  auto time_now = std::chrono::high_resolution_clock::now();
  bool already_saved = false;
  // ------------------ Main loop ------------------
  while (rclcpp::ok()) {
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
      // dump point cloud after 5 seconds
      if (!already_saved) {
        auto time_after = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                            time_after - time_now)
                            .count();
        if (duration > 5.0) {
          // create folder named as time
          save_ply(pc_output_file, merged_cloud);
          already_saved = true;
          RCLCPP_INFO(node->get_logger(), "Saved point cloud to %s",
                      pc_output_file.c_str());
        }
      }
    }
    Body fusedBody;
    bool useFusionAPI = false;
    if (useFusionAPI) {
      // This produces a wrong result even though singular cameras have
      // correct bodies
      if (fusion.process() != sl::FUSION_ERROR_CODE::SUCCESS) {
        RCLCPP_WARN(node->get_logger(), "Fusion process failed");
        continue;
      }
      if (fusion.retrieveBodies(fused_bodies,
                                body_tracking_runtime_parameters) !=
          sl::FUSION_ERROR_CODE::SUCCESS) {
        RCLCPP_WARN(node->get_logger(), "Could not retrieve bodies");
        continue;
      }
      if (fused_bodies.body_list.empty()) {
        RCLCPP_WARN(node->get_logger(), "No bodies found");
        continue;
      }
      raw_bodies_vector.push_back(fused_bodies.body_list[0]);
      std::vector<Body> bodies =
          extractBodyData(raw_bodies_vector, SMPL_TO_ZED);
      // bodies are already merged by the Fusion API
      if (!raw_bodies_vector.empty()) {
        fusedBody = bodies[0];
      }
    } else {

      // Prepare per-camera BodyData vector
      for (size_t i = 0; i < cameras.size(); i++) {

        clients[i].zed.retrieveBodies(detected_bodies[i]);
        if (detected_bodies[i].body_list.empty()) {
          continue;
        }
        // extract only the first body (TODO: get most centered?)
        raw_bodies_vector.push_back(detected_bodies[i].body_list[0]);
      }
      if (raw_bodies_vector.empty()) {
        continue;
      }
      // Extract vector of Body converting from sl::Bodies to custom Body
      // struct
      std::vector<Body> bodies =
          extractBodyData(raw_bodies_vector, SMPL_TO_ZED);
      // Merge the bodies into a single fused BodyData
      if (!raw_bodies_vector.empty()) {
        fusedBody = mergeBodiesWithExtrinsics(bodies, T_cams_extrinsics);
      }
    }
    raw_bodies_vector.clear();

    // Build and publish SMPL message
    auto msg = buildSMPLMessage(fusedBody, T_SMPL_TO_ROS, betas);
    smpl_pub->publish(msg);
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
