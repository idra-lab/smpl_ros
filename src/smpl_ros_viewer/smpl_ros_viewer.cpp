#include <iostream>
#include <memory>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "smpl_msgs/msg/smpl.hpp"
#include "smpl_ros_viewer/smpl_rviz.hpp"
#include "smplx.hpp"

using namespace torch::indexing;

class SMPLVisualizerNode : public rclcpp::Node {
public:
  SMPLVisualizerNode()
      : Node("smpl_visualizer"),
        device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {

    RCLCPP_INFO(this->get_logger(), "Using device: %s",
                device_ == torch::kCUDA ? "CUDA" : "CPU");

    // Declare ROS2 parameters
    this->declare_parameter<std::string>("model_path", "");

    // Read parameters
    model_path_ = this->get_parameter("model_path").as_string();
    RCLCPP_INFO(this->get_logger(), "Using SMPL model path: %s",
                model_path_.c_str());
    if (model_path_.empty()) {
      RCLCPP_FATAL(this->get_logger(), "SMPL model path not provided!");
      throw std::runtime_error("SMPL model path missing");
    }
    SMPL_TO_ROS_ = torch::tensor(
        {{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}},
        torch::TensorOptions().dtype(torch::kFloat64).device(device_));
  }

  void initialize() {
    // Load SMPL model
    smpl_ = std::make_unique<smplx::SMPL>(model_path_.c_str(), device_);
    smpl_->eval();
    faces_ = smpl_->faces();

    // Allocate body pose tensor (69 DOF) and translation
    const int batch_size = 1;
    body_pose_ = torch::zeros({batch_size, 69}, torch::kFloat64).to(device_);
    transl_ = torch::zeros({batch_size, 3}, torch::kFloat64).to(device_);
    global_orient_ = torch::zeros({batch_size, 3}, torch::kFloat64).to(device_);
    betas_ = torch::zeros({batch_size, smpl_->num_betas()}, torch::kFloat64)
                 .to(device_);

    // Initialize RViz visualization
    vis_ = std::make_shared<SMPLRviz>(shared_from_this());

    // Subscribe to custom SMPL message
    subscriber_ = this->create_subscription<smpl_msgs::msg::Smpl>(
        "/smpl_params", 10,
        std::bind(&SMPLVisualizerNode::smplCallback, this,
                  std::placeholders::_1));
  }

private:
  void smplCallback(const smpl_msgs::msg::Smpl::SharedPtr msg) {
    // Copy body_pose (69 elements)
    for (int i = 0; i < 69; ++i) {
      body_pose_.index_put_({0, i}, static_cast<double>(msg->body_pose[i]));
    }

    // Copy translation
    for (int i = 0; i < 3; ++i) {
      transl_.index_put_({0, i}, static_cast<double>(msg->transl[i]));
    }

    // Copy global orientation
    for (int i = 0; i < 3; ++i) {
      global_orient_.index_put_({0, i},
                                static_cast<double>(msg->global_orient[i]));
    }

    // --- 1) SMPL output using the parameters directly ---
    auto output = smpl_->forward(smplx::body_pose(body_pose_),
                                 smplx::global_orient(global_orient_),
                                 smplx::betas(betas_), smplx::transl(transl_),
                                 smplx::return_verts(true));
    auto vertices_param = output.vertices.value().squeeze(0); // (6890, 3)

    // --- 2) SMPL output with zero global orientation ---
    auto output_zero = smpl_->forward(
        smplx::body_pose(body_pose_),
        smplx::global_orient(torch::zeros({1, 3}, torch::kFloat64).to(device_)),
        smplx::betas(betas_),
        smplx::transl(torch::zeros({1, 3}, torch::kFloat64).to(device_)),
        smplx::return_verts(true));
    auto vertices_zero = output_zero.vertices.value().squeeze(0); // (6890, 3)

    // --- Apply global_orient + transl manually to vertices_zero ---
    // auto batch_rodrigues = [](const torch::Tensor &rot_vecs) {
    //   auto device = rot_vecs.device();
    //   auto dtype = rot_vecs.dtype();

    //   auto theta =
    //       torch::norm(rot_vecs, 2, /*dim=*/1, /*keepdim=*/true); // (B,1)
    //   auto k = rot_vecs / (theta + 1e-8); // normalize axis

    //   auto kx = k.index({torch::indexing::Slice(), 0});
    //   auto ky = k.index({torch::indexing::Slice(), 1});
    //   auto kz = k.index({torch::indexing::Slice(), 2});

    //   auto zero = torch::zeros_like(kx);
    //   auto K = torch::stack({torch::stack({zero, -kz, ky}, 1),
    //                          torch::stack({kz, zero, -kx}, 1),
    //                          torch::stack({-ky, kx, zero}, 1)},
    //                         1); // (B,3,3)

    //   auto I = torch::eye(3, dtype).to(device).unsqueeze(0); // (1,3,3)
    //   auto theta_expanded = theta.view({-1, 1, 1});          // (B,1,1)

    //   return I + torch::sin(theta_expanded) * K +
    //          (1 - torch::cos(theta_expanded)) * torch::matmul(K, K);
    // };
    // torch::Tensor rot_vec = global_orient_; // (1,3)
    // torch::Tensor rot_mat = batch_rodrigues(
        // rot_vec); // implement batch_rodrigues to convert 1x3 -> 3x3
    // torch::Tensor vertices_manual =
    //     torch::matmul(vertices_zero, rot_mat.squeeze(0).transpose(0, 1));
    // vertices_manual += transl_;

    // --- Transform to ROS axes ---
    vertices_param = torch::matmul(vertices_param, SMPL_TO_ROS_);
    // vertices_manual = torch::matmul(vertices_manual, SMPL_TO_ROS_);

    // --- Update RViz ---
    vis_->add_mesh(vertices_param.unsqueeze(0), faces_);
    // vis_->add_mesh(vertices_manual.unsqueeze(0), faces_);

    // --- Keypoints ---
    torch::Tensor keypoints =
        torch::zeros({24, 3}, torch::kFloat64).to(device_);
    for (int i = 0; i < 24; ++i) {
      keypoints.index_put_({i, 0},
                           static_cast<double>(msg->keypoints[i * 3 + 0]));
      keypoints.index_put_({i, 1},
                           static_cast<double>(msg->keypoints[i * 3 + 1]));
      keypoints.index_put_({i, 2},
                           static_cast<double>(msg->keypoints[i * 3 + 2]));
    }
    keypoints = torch::matmul(keypoints, SMPL_TO_ROS_);
    vis_->add_keypoints(keypoints);
    vis_->add_skeleton(keypoints);

    vis_->update_visualization();
  }

  // ---------------- ROS ----------------
  rclcpp::Subscription<smpl_msgs::msg::Smpl>::SharedPtr subscriber_;

  // ---------------- SMPL ----------------
  std::unique_ptr<smplx::SMPL> smpl_;
  torch::Tensor body_pose_;
  torch::Tensor transl_;
  torch::Tensor global_orient_;
  torch::Tensor betas_;
  torch::Tensor faces_;
  torch::Device device_;
  torch::Tensor SMPL_TO_ROS_;
  // ---------------- Visualization ----------------
  std::shared_ptr<SMPLRviz> vis_;

  // ---------------- Parameters ----------------
  std::string model_path_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SMPLVisualizerNode>();
  node->initialize();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
