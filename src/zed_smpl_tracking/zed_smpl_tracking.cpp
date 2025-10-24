#include <cv_bridge/cv_bridge.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <thread>

#include "smpl_msgs/msg/smpl.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "utils/json.hpp"
#include "yolov8_seg.h"
#include "zed_smpl_tracking/ClientPublisher.hpp"
#include "zed_smpl_tracking/utils.hpp"
// Helper to publish OpenCV images
void publish_image_msg(
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub,
    const cv::Mat &image, const std::string &frame_id) {
  std_msgs::msg::Header header;
  header.stamp = rclcpp::Clock().now();
  header.frame_id = frame_id;
  auto image_msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
  image_pub->publish(*image_msg);
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("smpl_single_camera_node");

  // ------------------ Parameters ------------------
  node->declare_parameter<std::string>("yolo_model_path", "");
  node->declare_parameter<bool>("publish_point_cloud", false);
  node->declare_parameter<bool>("publish_human_point_cloud", false);
  node->declare_parameter<bool>("publish_image", false);
  node->declare_parameter<bool>("publish_human", false);
  node->declare_parameter<std::string>("point_cloud_output_file",
                                       "human_cloud.ply");
  node->declare_parameter<std::string>("smpl_params_file", "");

  std::string yolo_model_path =
      node->get_parameter("yolo_model_path").as_string();
  bool publish_point_cloud =
      node->get_parameter("publish_point_cloud").as_bool();
  bool publish_human_point_cloud =
      node->get_parameter("publish_human_point_cloud").as_bool();
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
      node->create_publisher<sensor_msgs::msg::Image>("/zed/image", 10);

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
  if (!client.open(sl::InputType(), sl::COORDINATE_SYSTEM::IMAGE, &trigger,
                   0)) {
    RCLCPP_ERROR(node->get_logger(), "Failed to open ZED camera");
    return 1;
  }
  client.start();
  RCLCPP_INFO(node->get_logger(), "ZED camera opened successfully");
  // Load YOLO model if requested
  Yolov8Seg yolov8Seg;
  cv::dnn::Net yolo_net;
  if (publish_human_point_cloud && !yolo_model_path.empty()) {
    RCLCPP_INFO(node->get_logger(), "Loading YOLO model from %s",
                yolo_model_path.c_str());
    yolo_net = LoadYOLOModel(yolov8Seg, yolo_model_path);
  }

  Eigen::Matrix4d T_SMPL_TO_ROS = smpl_to_ros_transform();
  std::string frame_id = "zed_camera_frame";
  auto start_time = std::chrono::high_resolution_clock::now();
  bool cloud_saved = false;

  // ------------------ Main Loop ------------------
  while (rclcpp::ok()) {
    trigger.notifyZED();
    // Grab filtered point cloud (human only)
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> pc_data;
    if (publish_human_point_cloud) {
      pc_data = client.getFilteredPointCloud(Eigen::Matrix4d::Identity(),
                                             yolo_net, yolov8Seg);
      if (!pc_data.empty()) {
        publishMergedPointCloud(cloud_pub, pc_data, frame_id);

        // Save point cloud after 5 seconds
        auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - start_time)
                .count();
        if (!cloud_saved && elapsed > 5) {
          save_ply(pc_output_file, pc_data);
          RCLCPP_INFO(node->get_logger(), "Saved human point cloud to %s",
                      pc_output_file.c_str());
          cloud_saved = true;
        }
      }
    }

    // Publish RGB image if requested
    if (publish_image) {
      sl::Mat zed_image;
      if (client.zed.retrieveImage(zed_image, sl::VIEW::LEFT) ==
          sl::ERROR_CODE::SUCCESS) {
        cv::Mat cvImage(zed_image.getHeight(), zed_image.getWidth(), CV_8UC4,
                        zed_image.getPtr<sl::uchar1>(sl::MEM::CPU));
        cv::cvtColor(cvImage, cvImage, cv::COLOR_BGRA2BGR);
        publish_image_msg(image_pub, cvImage, frame_id);
      }
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
        auto msg = buildSMPLMessage(fusedBody, T_SMPL_TO_ROS, betas);
        smpl_pub->publish(msg);
      }
    }
    if (publish_point_cloud) {
      auto points = client.extractPointCloudFast();
      publishMergedPointCloud(cloud_pub, points, frame_id);
    }
  }
  trigger.running = false;
  trigger.notifyZED();
  // ------------------ Cleanup ------------------
  client.stop();
  exec.cancel();
  if (ros_spin_thread.joinable()) ros_spin_thread.join();
  rclcpp::shutdown();
  return 0;
}
