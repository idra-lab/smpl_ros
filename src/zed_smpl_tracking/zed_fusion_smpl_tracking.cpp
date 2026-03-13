#include <cv_bridge/cv_bridge.h>

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thread>
#include <vector>

#include "smpl_msgs/msg/smpl.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "utils/json.hpp"
#include "utils/voxel_filter.h"
#include "yolov8_seg.h"
#include "zed_smpl_tracking/ClientPublisher.hpp"
#include "zed_smpl_tracking/bodyConverter.hpp"
#include "zed_smpl_tracking/fuseSkeletons.hpp"
#include "zed_smpl_tracking/utils.hpp"

#include "utils/constants.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("smpl_publisher_node");

  // ------------------ Declare ROS2 Parameters ------------------
  node->declare_parameter<std::string>("calibration_file", "");
  node->declare_parameter<std::string>("yolo_model_path", "");
  auto tf_static_broadcaster_ =
      std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
  // used only with fusion API

  // Parameters declaration
  node->declare_parameter<std::string>("resolution", "1920x1080");
  node->declare_parameter<bool>("publish_merged_point_cloud", false);
  node->declare_parameter<bool>("publish_separate_point_clouds", false);
  node->declare_parameter<bool>("publish_human_depth_map", false);
  node->declare_parameter<bool>("publish_image", false);
  node->declare_parameter<bool>("publish_body", false);
  node->declare_parameter<std::string>("point_cloud_output_file",
                                       "human_cloud.ply");
  node->declare_parameter<double>("time_before_saving_pc", 5.0);
  node->declare_parameter<std::string>("smpl_params_file", "");
  node->declare_parameter<bool>("visualize_image", false);
  node->declare_parameter<bool>("overlay_yolo_mask", false);
  node->declare_parameter<int>("erode_body_mask_kernel_size", 5);
  node->declare_parameter<double>("published_body_filter_voxel_size", 0.02);

  // Print all declared parameters value
  // RCLCPP_INFO(node->get_logger(), "Node parameters:");
  // auto parameters = node->list_parameters({}, 10);
  // for (const auto &param_name : parameters.names) {
  //   auto param_value = node->get_parameter(param_name).as_string();
  //   RCLCPP_INFO(node->get_logger(), "  %s: %s", param_name.c_str(),
  //               param_value.c_str());
  // }

  // Parameters retrieval
  std::string calib_file = node->get_parameter("calibration_file").as_string();
  RCLCPP_INFO(node->get_logger(), "Using calibration file: %s",
              calib_file.c_str());
  std::string yolo_model_path =
      node->get_parameter("yolo_model_path").as_string();
  RCLCPP_INFO(node->get_logger(), "Using YOLO model file: %s",
              yolo_model_path.c_str());
  std::string resolution_str = node->get_parameter("resolution").as_string();
  bool publish_merged_point_cloud =
      node->get_parameter("publish_merged_point_cloud").as_bool();
  bool publish_human_depth_map =
      node->get_parameter("publish_human_depth_map").as_bool();
  bool publish_image = node->get_parameter("publish_image").as_bool();
  std::string smpl_params_path =
      node->get_parameter("smpl_params_file").as_string();
  bool publish_body = node->get_parameter("publish_body").as_bool();
  bool visualize_image = node->get_parameter("visualize_image").as_bool();
  bool overlay_yolo_mask = node->get_parameter("overlay_yolo_mask").as_bool();
  int erode_body_mask_kernel_size =
      node->get_parameter("erode_body_mask_kernel_size").as_int();
  double published_body_filter_voxel_size =
      node->get_parameter("published_body_filter_voxel_size").as_double();
  bool publish_separate_point_clouds =
      node->get_parameter("publish_separate_point_clouds").as_bool();

  std::vector<double> betas(300, 0.0);
  if (smpl_params_path == "") {
    RCLCPP_INFO(node->get_logger(),
                "No .json params file specified: SMPL betas set to zero");
  } else {
    betas = load_smpl_betas(smpl_params_path);
  }
  std::string pc_output_file =
      node->get_parameter("point_cloud_output_file").as_string();
  double time_before_saving_pc =
      node->get_parameter("time_before_saving_pc").as_double();
  RCLCPP_INFO(node->get_logger(),
              "Point cloud will be saved to: %s after %.2f seconds",
              pc_output_file.c_str(), time_before_saving_pc);

  // ------------------ ROS Publishers ------------------
  auto smpl_pub =
      node->create_publisher<smpl_msgs::msg::Smpl>("/smpl_params", 10);
  auto cloud_pub =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("/human_cloud", 10);
  // Image publisher (for visualization/debugging)
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> image_pubs;
  if (publish_image) {
    RCLCPP_INFO(node->get_logger(), "Image publishing enabled.");
    // create one publisher per camera
    for (int i = 0; i < 4; i++) {
      auto pub = node->create_publisher<sensor_msgs::msg::Image>(
          "/camera" + std::to_string(i + 1) + "/image", 10);
      image_pubs.push_back(pub);
    }
  }

  // ------------------ ZED + SMPL Setup ------------------
  // calibration must be done in IMAGE frame but we want data in ROS frame ->
  // ZED automatically converts the read JSON extrinsics to ROS frame
  int width = std::stoi(resolution_str.substr(0, resolution_str.find("x")));
  int height = std::stoi(resolution_str.substr(resolution_str.find("x") + 1));
  sl::RESOLUTION resolution;
  switch (height) {
  case 1242:
    resolution = sl::RESOLUTION::HD2K;
    break;
  case 1536:
    resolution = sl::RESOLUTION::HD1536;
    break;
  case 1080:
    resolution = sl::RESOLUTION::HD1080;
    break;
  case 1200:
    resolution = sl::RESOLUTION::HD1200;
    break;
  case 720:
    resolution = sl::RESOLUTION::HD720;
    break;
  case 376:
    resolution = sl::RESOLUTION::VGA;
    break;
  default:
    RCLCPP_ERROR(node->get_logger(), "Unsupported resolution height: %d",
                 height);
    return EXIT_FAILURE;
  }
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
      if (!clients[id_].open(conf.input_type,
                             sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD,
                             resolution, &trigger, gpu_id))
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
  init_params.maximum_working_resolution =
      sl::Resolution(std::max(1280, width), std::max(720, height));

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
  std::vector<std::string> cam_frames;

  for (int i = 0; i < clients.size(); i++) {
    auto cam_info = clients[i]
                        .zed.getCameraInformation()
                        .camera_configuration.calibration_parameters;
    auto conf = configurations[i];
    // print fx fy cx cy
    RCLCPP_INFO(
        node->get_logger(),
        "Camera SN %d: Intrinsics (fx, fy, cx, cy) = %.2f x %.2f x %.2f x %.2f",
        conf.serial_number, cam_info.left_cam.fx, cam_info.left_cam.fy,
        cam_info.left_cam.cx, cam_info.left_cam.cy);
    RCLCPP_INFO_STREAM(node->get_logger(), "Camera SN "
                                               << conf.serial_number
                                               << ": Extrinsics matrix:\n"
                                               << T_cams_extrinsics[i]);
    std::string frame_name = "cam" + std::to_string(i + 1) + "_" +
                             std::to_string(conf.serial_number);
    cam_frames.push_back(frame_name);
  }
  // ------------------  Per-camera publishers ------------------
  rclcpp::QoS camera_info_qos(1);
  camera_info_qos.reliable();
  camera_info_qos.transient_local();
  std::vector<rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr>
      cam_info_pubs;

  for (int i = 0; i < clients.size(); i++) {
    auto pub = node->create_publisher<sensor_msgs::msg::CameraInfo>(
        cam_frames[i] + "/camera_info", camera_info_qos);

    cam_info_pubs.push_back(pub);

    publishCameraInfo(pub, clients[i].zed, node->now(), cam_frames[i], width,
                      height);
  }

  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr>
      per_cam_cloud_pubs;

  if (publish_separate_point_clouds) {
    RCLCPP_INFO(node->get_logger(),
                "Per-camera point cloud publishing enabled.");
    // Create one publisher per camera
    for (size_t i = 0; i < cam_ids.size(); ++i) {
      std::string topic = cam_frames[i] + "/point_cloud";
      auto pub =
          node->create_publisher<sensor_msgs::msg::PointCloud2>(topic, 10);
      per_cam_cloud_pubs.push_back(pub);
      RCLCPP_INFO(node->get_logger(), "PointCloud topic: %s", topic.c_str());
    }
  }

  // --- ROS spinning in background thread ---
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  std::atomic<bool> exec_running{true};
  std::thread ros_spin_thread([&]() {
    exec.spin();
    exec_running = false;
  });
  RCLCPP_INFO(node->get_logger(), "ROS spinning thread started.");

  broadcastStaticCameras(tf_static_broadcaster_, T_cams_extrinsics, cam_frames,
                         cam_frames[0]);

  RCLCPP_INFO(node->get_logger(),
              "Published static TFs for cameras with frames: , parent frame: ");

  // Create depthmap publishers using cam ids in the topic name
  std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> depth_pubs;
  if (publish_human_depth_map) {
    RCLCPP_INFO(node->get_logger(), "Depth map publishing enabled.");
    for (size_t i = 0; i < cam_frames.size(); ++i) {
      std::string topic = cam_frames[i] + "/depth";
      auto pub = node->create_publisher<sensor_msgs::msg::Image>(topic, 10);
      depth_pubs.push_back(pub);

      RCLCPP_INFO(node->get_logger(), "Depth topic: %s", topic.c_str());
    }
  }

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
  if (publish_merged_point_cloud || overlay_yolo_mask) {
    if (yolo_model_path.empty()) {
      RCLCPP_WARN(
          node->get_logger(),
          "overlay_yolo_mask or publish_merged_point_cloud is enabled but "
          "yolo_model_path is empty.");
    }
    yolo_net = LoadYOLOModel(yolov8Seg, yolo_model_path);
  }

  std::vector<sl::Bodies> detected_bodies;
  detected_bodies.resize(cameras.size());

  auto time_now = std::chrono::high_resolution_clock::now();
  bool already_saved = false;
  bool include_normals = true;
  std::vector<cv::Mat> human_masks;
  human_masks.resize(clients.size());
  std::vector<cv::Rect> human_bboxes;
  human_bboxes.resize(clients.size());
  auto identity = Eigen::Matrix4d::Identity();

  SimpleTimer timer;
  // ------------------ Main loop ------------------
  while (rclcpp::ok()) {
    trigger.notifyZED();
    std::cout << "------------------ New Frame ------------------" << std::endl;
    // points, colors, normals
    std::vector<std::vector<
        std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>>>
        pcs(clients.size());
    if (publish_merged_point_cloud) {
      for (int i = 0; i < cameras.size(); i++) {
        // get points, colors and normals
        if (clients[i].getYoloPredictionMask(yolo_net, yolov8Seg, human_masks[i],
                                             human_bboxes[i],
                                             erode_body_mask_kernel_size)) {
          auto pcn = clients[i].getFilteredPointCloud(
              identity, human_masks[i], human_bboxes[i], include_normals);
          pcn = voxelDownsample(pcn, published_body_filter_voxel_size);
          pcs[i] = pcn;
        }
      }
      auto merged_cloud = mergePointClouds(pcs);
      publishPointCloud(cloud_pub, merged_cloud, cam_frames[0],
                        include_normals);
      // dump point cloud after 5 seconds
      if (!already_saved) {
        auto time_after = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                            time_after - time_now)
                            .count();
        if (duration > time_before_saving_pc) {
          // create folder named as time
          if (pc_output_file != "") {
            save_ply(pc_output_file, merged_cloud, include_normals);
            already_saved = true;
            RCLCPP_INFO(node->get_logger(), "Saved point cloud to %s",
                        pc_output_file.c_str());
          }
        }
      }
    }
    if (publish_separate_point_clouds) {
      // publish pc on separate topics
      auto identity = Eigen::Matrix4d::Identity();
      for (int i = 0; i < cameras.size(); i++) {
        auto pcn = clients[i].getFilteredPointCloud(
            identity, human_masks[i], human_bboxes[i], include_normals);
        publishPointCloud(per_cam_cloud_pubs[i], pcn, cam_frames[i],
                          include_normals);
      }
    }
    if (publish_human_depth_map) {
      for (size_t i = 0; i < clients.size(); ++i) {

        cv::Mat depth_map =
            clients[i].getFilteredDepthMap(human_masks[i], human_bboxes[i]);
        if (depth_map.empty()) {
          RCLCPP_WARN(node->get_logger(),
                      "Empty depth map for camera %d, skipping depth "
                      "publishing for this frame.",
                      cam_ids[i]);
          continue;
        }
        // continue;

        publish_depth_msg(depth_pubs[i], depth_map, cam_frames[i]);
      }
    }
    // Publish RGB image if requested
    if (publish_image || visualize_image) {
      for (size_t i = 0; i < clients.size(); ++i) {

        sl::Mat zed_image;
        if (clients[i].zed.retrieveImage(zed_image, sl::VIEW::LEFT) ==
            sl::ERROR_CODE::SUCCESS) {

          cv::Mat cvImage(zed_image.getHeight(), zed_image.getWidth(), CV_8UC4,
                          zed_image.getPtr<sl::uchar1>(sl::MEM::CPU));

          cv::cvtColor(cvImage, cvImage, cv::COLOR_BGRA2BGR);

          cv::Mat displayed_image = cvImage;
          if (overlay_yolo_mask && !yolo_model_path.empty()) {
            displayed_image =
                clients[i].overlayPersonMask(cvImage, human_masks[i], human_bboxes[i]);
          }

          if (publish_image) {
            publish_image_msg(image_pubs[i], displayed_image, cam_frames[i]);
          }

          if (visualize_image) {
            cv::imshow(std::string("Camera ") + std::to_string(cam_ids[i]),
                       displayed_image);
          }
        }
      }

      if (visualize_image) {
        cv::waitKey(1); // Call once per loop, not per camera
      }
    }
    Body fusedBody;
    // bool useFusionAPI = false;
    // if (useFusionAPI) {
    //   // This produces a wrong result even though singular cameras have
    //   // correct bodies
    //   if (fusion.process() != sl::FUSION_ERROR_CODE::SUCCESS) {
    //     RCLCPP_WARN(node->get_logger(), "Fusion process failed");
    //     continue;
    //   }
    //   if (fusion.retrieveBodies(fused_bodies,
    //                             body_tracking_runtime_parameters) !=
    //       sl::FUSION_ERROR_CODE::SUCCESS) {
    //     RCLCPP_WARN(node->get_logger(), "Could not retrieve bodies");
    //     continue;
    //   }
    //   if (fused_bodies.body_list.empty()) {
    //     RCLCPP_WARN(node->get_logger(), "No bodies found");
    //     continue;
    //   }
    //   raw_bodies_vector.push_back(fused_bodies.body_list[0]);
    //   std::vector<Body> bodies =
    //       extractBodyData(raw_bodies_vector, SMPL_TO_ZED);
    //   // bodies are already merged by the Fusion API
    //   if (!raw_bodies_vector.empty()) {
    //     fusedBody = bodies[0];
    //   }
    // } else {
    // timer.tik();
    if (publish_body) {
      // Prepare per-camera BodyData vector
      for (size_t i = 0; i < cameras.size(); i++) {
        clients[i].zed.retrieveBodies(detected_bodies[i]);
        // timer.tok((std::string("Bodies retrieval time for camera ") +
        //            std::to_string(cam_ids[i]))
        //               .c_str());
        if (detected_bodies[i].body_list.empty()) {
          continue;
        }
        // extract only the first body (TODO: get most centered?)
        raw_bodies_vector.push_back(detected_bodies[i].body_list[0]);
      }
      if (raw_bodies_vector.size() < cameras.size()) {
        RCLCPP_WARN(node->get_logger(),
                    "Not all cameras detected bodies (%ld/%ld)",
                    raw_bodies_vector.size(), cameras.size());
      }
      // Extract vector of Body converting from sl::Bodies to custom Body
      // struct
      std::vector<Body> bodies =
          extractBodyData(raw_bodies_vector, SMPL_TO_ZED);
      // timer.tok("Bodies extraction time");
      // Merge the bodies into a single fused BodyData
      if (!raw_bodies_vector.empty()) {
        fusedBody = mergeBodiesWithExtrinsics(bodies, T_cams_extrinsics);
      }
      // timer.tok("Bodies merging time");
      // }
      raw_bodies_vector.clear();

      // Build and publish SMPL message
      auto msg = buildSMPLMessage(fusedBody, T_SMPL_TO_ROS, betas);
      // timer.tok("SMPL message building time");
      smpl_pub->publish(msg);
      // timer.tok("SMPL message publishing time");
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
