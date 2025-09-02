#include <filesystem>
#include <fstream>
#include <iostream>
#include <open3d/Open3D.h>
#include <rclcpp/rclcpp.hpp>

#include "chamfer.h"
#include "json.hpp"
#include "o3d_converter.h"
#include "smpl_rviz.hpp"
#include "smplx.hpp"
#include "tqdm/tqdm.hpp"

#define VISUALIZATION true
#define VISUALIZATION_UPDATE_EVERY 1
using namespace torch::indexing;

class SMPLOptimizerNode : public rclcpp::Node {
public:
  SMPLOptimizerNode()
      : Node("smpl_optimizer"),
        device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
        chamfer_() {
    RCLCPP_INFO(this->get_logger(), "Using device: %s",
                device_ == torch::kCUDA ? "CUDA" : "CPU");

    // Declare parameters
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::string>("config_path", "");
    this->declare_parameter<std::string>("point_cloud_path", "");
  }

  void initialize() {
    // Load parameters
    model_path_ = this->get_parameter("model_path").as_string();
    config_path_ = this->get_parameter("config_path").as_string();
    point_cloud_path_ = this->get_parameter("point_cloud_path").as_string();
    if (VISUALIZATION) {
      vis_ = std::make_shared<SMPLRviz>(shared_from_this());
    }
    if (!std::filesystem::exists(model_path_) ||
        !std::filesystem::exists(config_path_) ||
        !std::filesystem::exists(point_cloud_path_)) {
      RCLCPP_FATAL(this->get_logger(), "Input file(s) missing!");
      throw std::runtime_error("Missing init files");
    }

    // Load JSON config
    std::ifstream file(config_path_);
    file >> config_;

    // Load SMPL model
    smpl_ = std::make_unique<smplx::SMPL>(model_path_.c_str(), device_);
    smpl_->eval();

    // Load target point cloud using Open3D
    auto cloud_ptr =
        open3d::io::CreatePointCloudFromFile(point_cloud_path_, "auto", true);
    if (cloud_ptr->IsEmpty()) {
      throw std::runtime_error("Failed to load point cloud");
    }

    cloud_ptr = cloud_ptr->VoxelDownSample(0.005);
    auto [vertices, colors] = open3d_pointcloud_to_tensor(*cloud_ptr);

    // Publish point cloud via RViz helper
    if (VISUALIZATION) {
      vis_->update_point_cloud(vertices, colors);
    }

    vertices_target_ = vertices.to(device_).unsqueeze(0); // (1, N, 3)
    RCLCPP_INFO(this->get_logger(), "Loaded target point cloud with %ld points",
                vertices_target_.size(1));

    // Initialize optimization parameters
    const int batch_size = 1;
    betas_ = torch::zeros({batch_size, smpl_->num_betas()}, torch::kFloat64)
                 .to(device_)
                 .set_requires_grad(true);
    body_pose_ = torch::zeros({batch_size, 69}, torch::kFloat64)
                     .to(device_)
                     .set_requires_grad(true);
    global_orient_ = torch::zeros({batch_size, 3}, torch::kFloat64)
                         .to(device_)
                         .set_requires_grad(true);
    transl_ = torch::zeros({batch_size, 3}, torch::kFloat64)
                  .to(device_)
                  .set_requires_grad(true);
    {
      torch::NoGradGuard no_grad;

      auto deg_to_rad = [](double degrees) { return degrees * M_PI / 180.0; };
      transl_.index_put_(
          {0, Slice()},
          torch::tensor({static_cast<double>(config_["initial_transl"][0]),
                         static_cast<double>(config_["initial_transl"][1]),
                         static_cast<double>(config_["initial_transl"][2])}));

      // Global orientation
      double r = static_cast<double>(config_["initial_global_orient_rpy"][0]);
      double p = static_cast<double>(config_["initial_global_orient_rpy"][1]);
      double y = static_cast<double>(config_["initial_global_orient_rpy"][2]);
      // Convert RPY to rodrigues
      Eigen::Matrix3d R;
      R = Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX()) *
          Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ());
      Eigen::AngleAxisd aa(R);
      Eigen::Vector3d rod = aa.axis() * aa.angle();
      global_orient_.index_put_({0, 0}, rod[0]);
      global_orient_.index_put_({0, 1}, rod[1]);
      global_orient_.index_put_({0, 2}, rod[2]);

      double rot_x_r = deg_to_rad(config_["shoulder_r_start"][0]);
      double rot_y_r = deg_to_rad(config_["shoulder_r_start"][1]);
      double rot_z_r = deg_to_rad(config_["shoulder_r_start"][2]);
      torch::Tensor shoulder_rot_r =
          torch::tensor({rot_x_r, rot_y_r, rot_z_r}, torch::kFloat64);
      double rot_x_l = deg_to_rad(config_["shoulder_l_start"][0]);
      double rot_y_l = deg_to_rad(config_["shoulder_l_start"][1]);
      double rot_z_l = deg_to_rad(config_["shoulder_l_start"][2]);
      torch::Tensor shoulder_rot_l =
          torch::tensor({rot_x_l, rot_y_l, rot_z_l}, torch::kFloat64);
      const int left_shoulder = 15;
      const int right_shoulder = 16;
      body_pose_.narrow(1, 3 * right_shoulder, 3).copy_(shoulder_rot_r);
      body_pose_.narrow(1, 3 * left_shoulder, 3).copy_(shoulder_rot_l);
      double y_offset = config_["pos"]["y_offset"];
      transl_.index_put_({0, 1}, y_offset);
    }

    faces_ = smpl_->faces();
  }

  void run_optimization() {
    std::vector<torch::Tensor> params;

    // Position optimization
    params = {transl_};
    optimize(params, config_["pos"]["lr"], config_["pos"]["it"]);

    // Rotation optimization
    params = {global_orient_};
    optimize(params, config_["rot"]["lr"], config_["rot"]["it"]);

    // Position + rotation optimization
    params = {global_orient_, transl_};
    optimize(params, config_["pos-rot"]["lr"], config_["pos-rot"]["it"]);

    // Body pose optimization
    params = {body_pose_};
    optimize(params, config_["body_pose"]["lr"], config_["body_pose"]["it"]);

    // Shape optimization
    params = {betas_};
    optimize(params, config_["shape"]["lr"], config_["shape"]["it"]);

    RCLCPP_INFO(this->get_logger(), "Optimization complete.");
    rclcpp::shutdown();
    exit(0);
  }

private:
  void optimize(const std::vector<torch::Tensor> &params_to_optimize,
                double learning_rate, int steps, double eps_thresh = 1e-6) {
    torch::optim::Adam optimizer(params_to_optimize,
                                 torch::optim::AdamOptions(learning_rate));
    double loss_old = 0.0;
    auto bar = tqdm::tqdm(tqdm::range(steps));
    bar.set_prefix("Iterating over A: ");
    for (int step : bar) {

      if (!rclcpp::ok()) {
        RCLCPP_INFO(this->get_logger(),
                    "ROS shutdown detected, stopping optimization");
        break;
      }
      optimizer.zero_grad();

      auto output = smpl_->forward(
          smplx::betas(betas_), smplx::global_orient(global_orient_),
          smplx::body_pose(body_pose_), smplx::transl(transl_),
          smplx::return_verts(true));

      auto vertices_pred = output.vertices.value();
      auto loss = chamfer_.forward(vertices_pred.to(torch::kFloat64),
                                   vertices_target_.to(torch::kFloat64), true);
      // if (betas_.defined()) {
      //   loss += 0.001 * betas_.pow(2).sum();
      // }
      bar << "loss = " << loss.item<float>();
      loss.backward();
      optimizer.step();
      // Check for convergence
      if (std::abs(loss.item<double>() - loss_old) < eps_thresh) {
        RCLCPP_INFO(this->get_logger(),
                    "Loss converged, stopping optimization at step %d", step);
        break;
      }
      loss_old = loss.item<double>();
      // Update visualization
      if (VISUALIZATION &&
          (step % VISUALIZATION_UPDATE_EVERY == 0 || step == steps - 1)) {
        vis_->update_mesh(vertices_pred, faces_);
      }
    }
  }

  std::unique_ptr<smplx::SMPL> smpl_;
  torch::Tensor vertices_target_;
  torch::Tensor faces_;
  torch::Tensor betas_, body_pose_, global_orient_, transl_;
  torch::Device device_;
  ChamferDistance chamfer_;
  nlohmann::json config_;

  std::shared_ptr<SMPLRviz> vis_;

  std::string model_path_, config_path_, point_cloud_path_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SMPLOptimizerNode>();
  node->initialize();
  node->run_optimization();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
