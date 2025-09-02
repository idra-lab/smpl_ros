#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <torch/torch.h>
#include <visualization_msgs/msg/marker.hpp>

class SMPLRviz {
public:
  SMPLRviz(rclcpp::Node::SharedPtr node, std::string frame_id = "map")
      : node_(node) {
    mesh_pub_ = node_->create_publisher<visualization_msgs::msg::Marker>(
        "/smpl_mesh", 10);
    cloud_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/target_cloud", 10);

    mesh_marker_.header.frame_id = frame_id;
    mesh_marker_.ns = "smpl";
    mesh_marker_.id = 0;
    mesh_marker_.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    mesh_marker_.action = visualization_msgs::msg::Marker::ADD;
    mesh_marker_.scale.x = 1.0;
    mesh_marker_.scale.y = 1.0;
    mesh_marker_.scale.z = 1.0;
    mesh_marker_.color.a = 1.0;
    mesh_marker_.color.r = 0.6;
    mesh_marker_.color.g = 0.6;
    mesh_marker_.color.b = 0.8;
  }

  void update_mesh(const torch::Tensor &vertices, const torch::Tensor &faces) {
    auto verts = vertices.squeeze(0).to(torch::kCPU).contiguous();
    auto verts_acc = verts.accessor<double, 2>();
    auto faces_cpu = faces.to(torch::kCPU).contiguous();
    auto faces_acc = faces_cpu.accessor<uint32_t, 2>();

    mesh_marker_.points.clear();
    mesh_marker_.points.reserve(faces_cpu.size(0) * 3);
    geometry_msgs::msg::Point p0, p1, p2;
    int64_t f0, f1, f2;
    for (int64_t i = 0; i < faces_cpu.size(0); ++i) {
      f0 = faces_acc[i][0];
      f1 = faces_acc[i][1];
      f2 = faces_acc[i][2];
      p0.x = verts_acc[f0][0];
      p0.y = verts_acc[f0][1];
      p0.z = verts_acc[f0][2];
      p1.x = verts_acc[f1][0];
      p1.y = verts_acc[f1][1];
      p1.z = verts_acc[f1][2];
      p2.x = verts_acc[f2][0];
      p2.y = verts_acc[f2][1];
      p2.z = verts_acc[f2][2];

      mesh_marker_.points.push_back(p0);
      mesh_marker_.points.push_back(p1);
      mesh_marker_.points.push_back(p2);
    }

    mesh_marker_.header.stamp = node_->now();
    mesh_pub_->publish(mesh_marker_);
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

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr mesh_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  visualization_msgs::msg::Marker mesh_marker_;
};
