#include <cv_bridge/cv_bridge.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thread>

#include "smpl_msgs/msg/fixed_size_image.hpp"
#include "smpl_msgs/msg/smpl.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "utils/json.hpp"
#include "yolo_seg.h"
#include "zed_smpl_tracking/ClientPublisher.hpp"
#include "zed_smpl_tracking/bodyConverter.hpp"
#include "zed_smpl_tracking/utils.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("smpl_single_camera_node");

  // ------------------ Parameters ------------------
  node->declare_parameter<std::string>("yolo_model_path", "");
  node->declare_parameter<bool>("publish_point_cloud", true);
  node->declare_parameter<bool>("publish_human_point_cloud", true);
  node->declare_parameter<bool>("publish_human_depth_map", false);
  node->declare_parameter<bool>("publish_image", true);
  node->declare_parameter<bool>("publish_human", true);
  node->declare_parameter<std::string>("point_cloud_output_file",
                                       "human_cloud.ply");
  node->declare_parameter<std::string>("smpl_params_file", "");

  std::string yolo_model_path =
      node->get_parameter("yolo_model_path").as_string();
  bool publish_point_cloud =
      node->get_parameter("publish_point_cloud").as_bool();
  bool publish_human_point_cloud =
      node->get_parameter("publish_human_point_cloud").as_bool();
  bool publish_human_depth_map =
      node->get_parameter("publish_human_depth_map").as_bool();
  bool publish_image = node->get_parameter("publish_image").as_bool();
  bool publish_human = node->get_parameter("publish_human").as_bool();
  std::string pc_output_file =
      node->get_parameter("point_cloud_output_file").as_string();
  std::string smpl_params_path =
      node->get_parameter("smpl_params_file").as_string();

  std::vector<double> betas(10, 0.0);
  if (!smpl_params_path.empty()) {
    betas = load_smpl_betas(smpl_params_path);
  }

  // ------------------ Publishers ------------------
  auto smpl_pub =
      node->create_publisher<smpl_msgs::msg::Smpl>("/smpl_params", 10);
  auto cloud_pub =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("/human_cloud", 10);
  auto image_pub =
      node->create_publisher<smpl_msgs::msg::FixedSizeImage>("/zed/image", 10);
  auto depth_map_pub = node->create_publisher<smpl_msgs::msg::FixedSizeImage>(
      "/human_depth_map", 10);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  std::atomic<bool> exec_running{true};
  std::thread ros_spin_thread([&]() {
    exec.spin();
    exec_running = false;
  });

  // ------------------ ClientPublisher ------------------
  RCLCPP_INFO(node->get_logger(), "Starting ZED SMPL tracking...");
  ClientPublisher client;
  Trigger trigger;
  if (!client.open(sl::InputType(), sl::COORDINATE_SYSTEM::IMAGE,
                   sl::RESOLUTION::HD2K, &trigger, 0)) {
    RCLCPP_ERROR(node->get_logger(), "Failed to open ZED camera");
    return 1;
  }
  client.start();
  RCLCPP_INFO(node->get_logger(), "ZED camera opened successfully");
  // Load YOLOE model if requested
  std::unique_ptr<YoloeSegDetector> yoloe_detector;
  if (publish_human_point_cloud && !yolo_model_path.empty()) {
    RCLCPP_INFO(node->get_logger(), "Loading YOLOE model from %s",
                yolo_model_path.c_str());
    yoloe_detector = LoadYOLOModel(yolo_model_path);
  }

  Eigen::Matrix4d T_SMPL_TO_ROS = smpl_to_ros_transform();
  std::string frame_id = "zed_camera_frame";
  auto start_time = std::chrono::high_resolution_clock::now();
  bool cloud_saved = false;
  cv::Mat human_mask;
  cv::Rect human_bbox;
  auto identity = Eigen::Matrix4d::Identity();
  bool include_normals = true;
  cv::Mat rgb;
  // ------------------ Main Loop ------------------
  while (rclcpp::ok()) {
    trigger.notifyZED();
    // Grab filtered point cloud (human only)
    std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>>
        pc_data;
    if (publish_human_point_cloud) {
      // clear mask and bbox for each frame
      human_mask = cv::Mat();
      human_bbox = cv::Rect();
      if (yoloe_detector &&
          client.getYoloPredictionMask(*yoloe_detector, rgb, human_mask,
                                       human_bbox, 0.15f,0.45f, 0, 0)) {
        auto pc_data = client.getFilteredPointCloud(
            identity, human_mask, human_bbox, include_normals);
      }
      if (!pc_data.empty()) {
        publishPointCloud(cloud_pub, pc_data, frame_id);

        // Save point cloud after 5 seconds
        auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_time)
                .count();
        if (!cloud_saved && elapsed > 5) {
          save_ply(pc_output_file, pc_data, false);
          RCLCPP_INFO(node->get_logger(), "Saved human point cloud to %s",
                      pc_output_file.c_str());
          cloud_saved = true;
        }
      }
    }
    if (publish_human_depth_map) {
      cv::Mat depth_map = client.getFilteredDepthMap(human_mask, human_bbox);
      if (!depth_map.empty()) {
        publish_depth_msg(depth_map_pub, depth_map, rgb, frame_id);
      }
    }
    // Publish RGB image if requested
    // if (publish_image) {
    //   publish_image_msg(image_pub, rgb, frame_id);
    // }
  }

  // Retrieve body and publish SMPL
  if (publish_human) {
    sl::Bodies bodies;
    sl::BodyTrackingRuntimeParameters body_runtime;
    body_runtime.detection_confidence_threshold = 40;
    if (client.zed.retrieveBodies(bodies, body_runtime) ==
            sl::ERROR_CODE::SUCCESS &&
        !bodies.body_list.empty()) {
      std::vector<sl::BodyData> body_vec = {bodies.body_list[0]};
      std::vector<Body> bodies_out = extractBodyData(body_vec, SMPL_TO_ZED);
      Body fusedBody = bodies_out[0];
      auto msg =
          buildSMPLMessage(fusedBody, T_SMPL_TO_ROS, betas, node->get_clock());
      smpl_pub->publish(msg);
    }
    // TODO fix
    // if (publish_point_cloud) {
    //   auto points = client.extractPointCloudFast(false);
    //   publishMergedPointCloud(cloud_pub, points, frame_id);
    // }
  }
  trigger.running = false;
  trigger.notifyZED();
  // ------------------ Cleanup ------------------
  client.stop();
  exec.cancel();
  if (ros_spin_thread.joinable())
    ros_spin_thread.join();
  rclcpp::shutdown();
  return 0;
}
