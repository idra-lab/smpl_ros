#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <torch/torch.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

static const int SMPL_PARENTS[24] = {-1, 0,  0,  0,  1,  2,  3,  4,
                                     5,  6,  7,  8,  9,  9,  9,  12,
                                     13, 14, 16, 17, 18, 19, 20, 21};

class SMPLRviz {
public:
  SMPLRviz(rclcpp::Node::SharedPtr node, std::string frame_id = "map")
      : node_(node), frame_id_(frame_id) {
    marker_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/smpl_markers", 10);
    cloud_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/target_cloud", 10);
    next_marker_id_ = 0;
  }

  void add_mesh(const torch::Tensor &vertices, const torch::Tensor &faces) {
    visualization_msgs::msg::Marker mesh;
    mesh.header.frame_id = frame_id_;
    mesh.ns = "smpl_mesh";
    mesh.id = next_marker_id_++;
    mesh.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    mesh.action = visualization_msgs::msg::Marker::ADD;
    mesh.scale.x = mesh.scale.y = mesh.scale.z = 1.0;
    mesh.color.a = 1.0;
    mesh.color.r = 1.0;
    mesh.color.g = 0.5;
    mesh.color.b = 0.0;

    auto verts = vertices.squeeze(0).to(torch::kCPU).contiguous();
    auto verts_acc = verts.accessor<double, 2>();
    auto faces_cpu = faces.to(torch::kCPU).contiguous();
    auto faces_acc = faces_cpu.accessor<uint32_t, 2>();

    mesh.points.reserve(faces_cpu.size(0) * 3);
    geometry_msgs::msg::Point p0, p1, p2;
    for (int64_t i = 0; i < faces_cpu.size(0); ++i) {
      p0.x = verts_acc[faces_acc[i][0]][0];
      p0.y = verts_acc[faces_acc[i][0]][1];
      p0.z = verts_acc[faces_acc[i][0]][2];
      p1.x = verts_acc[faces_acc[i][1]][0];
      p1.y = verts_acc[faces_acc[i][1]][1];
      p1.z = verts_acc[faces_acc[i][1]][2];
      p2.x = verts_acc[faces_acc[i][2]][0];
      p2.y = verts_acc[faces_acc[i][2]][1];
      p2.z = verts_acc[faces_acc[i][2]][2];

      mesh.points.push_back(p0);
      mesh.points.push_back(p1);
      mesh.points.push_back(p2);
    }

    markers_.markers.push_back(mesh);
  }

  void add_keypoints(const torch::Tensor &keypoints) {
    visualization_msgs::msg::Marker kp;
    kp.header.frame_id = frame_id_;
    kp.ns = "keypoints";
    kp.id = next_marker_id_++;
    kp.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    kp.action = visualization_msgs::msg::Marker::ADD;
    kp.scale.x = kp.scale.y = kp.scale.z = 0.04;
    kp.color.a = 1.0;
    kp.color.r = 1.0;
    kp.color.g = 0.0;
    kp.color.b = 0.0;

    geometry_msgs::msg::Point p;
    for (int i = 0; i < keypoints.size(0); ++i) {
      p.x = keypoints.index({i, 0}).item<double>();
      p.y = keypoints.index({i, 1}).item<double>();
      p.z = keypoints.index({i, 2}).item<double>();
      kp.points.push_back(p);
    }

    markers_.markers.push_back(kp);
  }
  void add_arrow(const torch::Tensor &start, const torch::Tensor &end,
                 const std::string &ns = "arrows", float r = 0.0, float g = 0.0,
                 float b = 1.0) {
    if (start.sizes() != end.sizes() || start.sizes().size() != 1 ||
        start.size(0) != 3) {
      RCLCPP_ERROR(rclcpp::get_logger("SMPLRviz"),
                   "add_arrow: start and end must be 1D tensors of size 3");
      return;
    }

    visualization_msgs::msg::Marker arrow;
    arrow.header.frame_id = frame_id_;
    arrow.ns = ns;
    arrow.id = next_marker_id_++;
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    arrow.scale.x = 0.02; // shaft diameter
    arrow.scale.y = 0.04; // head diameter
    arrow.scale.z = 0.1;  // head length
    arrow.color.a = 1.0;
    arrow.color.r = r;
    arrow.color.g = g;
    arrow.color.b = b;

    geometry_msgs::msg::Point p_start, p_end;
    p_start.x = start.index({0}).item<double>();
    p_start.y = start.index({1}).item<double>();
    p_start.z = start.index({2}).item<double>();
    p_end.x = end.index({0}).item<double>();
    p_end.y = end.index({1}).item<double>();
    p_end.z = end.index({2}).item<double>();

    arrow.points.push_back(p_start);
    arrow.points.push_back(p_end);

    markers_.markers.push_back(arrow);
  }

  void add_skeleton(const torch::Tensor &keypoints) {
    visualization_msgs::msg::Marker skel;
    skel.header.frame_id = frame_id_;
    skel.ns = "skeleton";
    skel.id = next_marker_id_++;
    skel.type = visualization_msgs::msg::Marker::LINE_LIST;
    skel.action = visualization_msgs::msg::Marker::ADD;
    skel.scale.x = 0.02; // line thickness
    skel.color.a = 1.0;
    skel.color.r = 0.0;
    skel.color.g = 1.0;
    skel.color.b = 0.0;

    geometry_msgs::msg::Point p, q;
    for (int i = 0; i < 24; ++i) {
      int parent = SMPL_PARENTS[i];
      if (parent >= 0) {
        p.x = keypoints.index({parent, 0}).item<double>();
        p.y = keypoints.index({parent, 1}).item<double>();
        p.z = keypoints.index({parent, 2}).item<double>();

        q.x = keypoints.index({i, 0}).item<double>();
        q.y = keypoints.index({i, 1}).item<double>();
        q.z = keypoints.index({i, 2}).item<double>();

        skel.points.push_back(p);
        skel.points.push_back(q);
      }
    }

    markers_.markers.push_back(skel);
  }

  void update_point_cloud(const torch::Tensor &points,
                          const torch::Tensor &colors,
                          const std::string &frame_id = "map") {
    auto points_cpu = points.to(torch::kCPU).contiguous();
    auto colors_cpu = colors.to(torch::kCPU).contiguous();
    int64_t num_points = points_cpu.size(0);

    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.height = 1;
    cloud_msg.width = static_cast<uint32_t>(num_points);
    cloud_msg.is_dense = true;
    cloud_msg.point_step = 16;
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
    cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);

    cloud_msg.fields.clear();
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

    auto points_acc = points_cpu.accessor<double, 2>();
    auto colors_acc = colors_cpu.accessor<double, 2>();

    for (int64_t i = 0; i < num_points;
         ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_rgb) {
      *iter_x = static_cast<float>(points_acc[i][0]);
      *iter_y = static_cast<float>(points_acc[i][1]);
      *iter_z = static_cast<float>(points_acc[i][2]);

      uint8_t r = static_cast<uint8_t>(colors_acc[i][0] * 255.0);
      uint8_t g = static_cast<uint8_t>(colors_acc[i][1] * 255.0);
      uint8_t b = static_cast<uint8_t>(colors_acc[i][2] * 255.0);
      uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                      static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
      *iter_rgb = *reinterpret_cast<float *>(&rgb);
    }

    cloud_msg.header.stamp = node_->now();
    cloud_msg.header.frame_id = frame_id;
    cloud_pub_->publish(cloud_msg);
  }

  void update_visualization() {
    for (auto &m : markers_.markers) {
      m.header.stamp = node_->now();
    }
    marker_pub_->publish(markers_);
    markers_.markers.clear();
    next_marker_id_ = 0;
  }

  void clear_markers() {
    visualization_msgs::msg::Marker delete_all;
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    visualization_msgs::msg::MarkerArray arr;
    arr.markers.push_back(delete_all);
    marker_pub_->publish(arr);
    markers_.markers.clear();
    next_marker_id_ = 0;
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  visualization_msgs::msg::MarkerArray markers_;
  std::string frame_id_;
  int next_marker_id_;
};
