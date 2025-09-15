#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <open3d/Open3D.h>
#include <rclcpp/rclcpp.hpp>

#include "chamfer.h"
#include "smpl_ros_viewer/smpl_rviz.hpp"
#include "smplx.hpp"
#include "utils/json.hpp"
#include "utils/o3d_converter.h"
#include "utils/ros_pbar.hpp"

#define VISUALIZATION true
#define VISUALIZATION_UPDATE_EVERY 1

using namespace torch::indexing;

/// SMPLOptimizerNode
/// ROS2 Node for optimizing SMPL model parameters to fit a target point cloud.
class SMPLOptimizerNode : public rclcpp::Node {
public:
  SMPLOptimizerNode()
      : Node("smpl_optimizer"),
        device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
        chamfer_() {
    RCLCPP_INFO(this->get_logger(), "Using device: %s",
                device_ == torch::kCUDA ? "CUDA" : "CPU");

    // Declare ROS2 parameters
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::string>("config_path", "");
    this->declare_parameter<std::string>("point_cloud_path", "");
  }

  /// Initialize the node: load model, config, point cloud, and setup
  /// optimization parameters
  void initialize() {
    load_parameters();
    load_config();
    load_smpl_model();
    load_target_point_cloud();
    initialize_optimization_params();
  }

  /// Run the full optimization pipeline
  void run_optimization() {
    dump_parameters_json(); // dump initial parameters
    RCLCPP_INFO(this->get_logger(), "Starting optimization...");

    // Optimize translation
    optimize({transl_}, config_["pos"]["lr"], config_["pos"]["it"], "Position");

    // Optimize global orientation
    optimize({global_orient_}, config_["rot"]["lr"], config_["rot"]["it"],
             "Rotation");

    // Optimize position + rotation together
    optimize({global_orient_, transl_}, config_["pos-rot"]["lr"],
             config_["pos-rot"]["it"], "Position + Rotation");

    // Optimize body pose
    optimize({body_pose_}, config_["body_pose"]["lr"],
             config_["body_pose"]["it"], "Body Pose");

    // Optimize shape (betas)
    optimize({betas_}, config_["shape"]["lr"], config_["shape"]["it"], "Shape");

    // Dump final parameters to JSON
    dump_parameters_json();

    RCLCPP_INFO(this->get_logger(), "Optimization complete.");
    rclcpp::shutdown();
    exit(0);
  }

private:
  // -------------------- Initialization Helpers --------------------
  void load_parameters() {
    model_path_ = this->get_parameter("model_path").as_string();
    config_path_ = this->get_parameter("config_path").as_string();
    point_cloud_path_ = this->get_parameter("point_cloud_path").as_string();

    if (!std::filesystem::exists(model_path_) ||
        !std::filesystem::exists(config_path_) ||
        !std::filesystem::exists(point_cloud_path_)) {
      RCLCPP_FATAL(this->get_logger(), "One or more input files are missing!");
      throw std::runtime_error("Missing input files");
    }

    if (VISUALIZATION) {
      vis_ = std::make_shared<SMPLRviz>(shared_from_this());
    }
  }

  void load_config() {
    std::ifstream file(config_path_);
    file >> config_;
  }

  void load_smpl_model() {
    smpl_ = std::make_unique<smplx::SMPL>(model_path_.c_str(), device_);
    smpl_->eval();
    faces_ = smpl_->faces();
  }

  void load_target_point_cloud() {
    auto cloud_ptr =
        open3d::io::CreatePointCloudFromFile(point_cloud_path_, "auto", true);
    if (cloud_ptr->IsEmpty()) {
      throw std::runtime_error("Failed to load target point cloud");
    }

    cloud_ptr = cloud_ptr->VoxelDownSample(0.02);
    auto [vertices, colors] = open3d_pointcloud_to_tensor(*cloud_ptr);

    vertices_target_ = vertices.to(device_).unsqueeze(0); // (1, N, 3)
    RCLCPP_INFO(this->get_logger(), "Loaded target point cloud with %ld points",
                vertices_target_.size(1));

    if (VISUALIZATION) {
      vis_->update_point_cloud(vertices, colors);
    }
  }

  void initialize_optimization_params() {
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

    torch::NoGradGuard no_grad;
    auto deg_to_rad = [](double degrees) { return degrees * M_PI / 180.0; };

    // Initialize translation
    transl_.index_put_(
        {0, Slice()},
        torch::tensor({static_cast<double>(config_["initial_transl"][0]),
                       static_cast<double>(config_["initial_transl"][1]),
                       static_cast<double>(config_["initial_transl"][2])}));

    // Initialize global orientation (RPY to Rodrigues)
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(config_["initial_global_orient_rpy"][0],
                          Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(config_["initial_global_orient_rpy"][1],
                          Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(config_["initial_global_orient_rpy"][2],
                          Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd aa(R);
    Eigen::Vector3d rod = aa.axis() * aa.angle();
    global_orient_.index_put_({0, 0}, rod[0]);
    global_orient_.index_put_({0, 1}, rod[1]);
    global_orient_.index_put_({0, 2}, rod[2]);

    // Initialize shoulders
    const int left_shoulder = 15;
    const int right_shoulder = 16;
    torch::Tensor shoulder_rot_r =
        torch::tensor({deg_to_rad(config_["shoulder_r_start"][0]),
                       deg_to_rad(config_["shoulder_r_start"][1]),
                       deg_to_rad(config_["shoulder_r_start"][2])},
                      torch::kFloat64);
    torch::Tensor shoulder_rot_l =
        torch::tensor({deg_to_rad(config_["shoulder_l_start"][0]),
                       deg_to_rad(config_["shoulder_l_start"][1]),
                       deg_to_rad(config_["shoulder_l_start"][2])},
                      torch::kFloat64);
    body_pose_.narrow(1, 3 * right_shoulder, 3).copy_(shoulder_rot_r);
    body_pose_.narrow(1, 3 * left_shoulder, 3).copy_(shoulder_rot_l);
  }

  // -------------------- Optimization Loop --------------------
  void optimize(const std::vector<torch::Tensor> &params_to_optimize,
                double learning_rate, int steps, const std::string &stage_name,
                double eps_thresh = 1e-6) {
    torch::optim::Adam optimizer(params_to_optimize,
                                 torch::optim::AdamOptions(learning_rate));

    double loss_old = 0.0;
    std::vector<int> steps_vec(steps);
    std::iota(steps_vec.begin(), steps_vec.end(), 0);

    auto bar = ros_pbar::ros_pbar(steps_vec, this->get_logger());
    bar.set_prefix("Optimizing " + stage_name);

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
      loss.backward();
      optimizer.step();

      // Compute gradient magnitude for convergence
      double loss_grad = std::abs(loss.item<double>() - loss_old);
      bar.set_suffix("Loss: " + std::to_string(loss.item<double>()) +
                     " | grad: " + std::to_string(loss_grad));

      if (loss_grad < eps_thresh) {
        RCLCPP_INFO(this->get_logger(), "Loss converged at step %d", step);
        break;
      }
      loss_old = loss.item<double>();

      // Update visualization
      if (VISUALIZATION &&
          (step % VISUALIZATION_UPDATE_EVERY == 0 || step == steps - 1)) {
        vis_->add_mesh(vertices_pred, faces_);
        vis_->update_visualization();
      }
    }

    RCLCPP_INFO(this->get_logger(), "------------------------------");
  }

  // -------------------- Dump final parameters --------------------
  void dump_parameters_json() {
    auto torch_to_std_vector = [](const torch::Tensor &tensor) {
      std::vector<double> vec(tensor.numel());
      for(int i=0; i<tensor[0].numel(); i++) {
        if (std::isnan(tensor[0][i].item<double>())) {
          RCLCPP_WARN(rclcpp::get_logger("dump_parameters_json"),
                      "NaN detected in parameters, setting to zero.");
          tensor[0][i] = 0.0;
        }
        vec[i] = tensor[0][i].item<double>();
      }
      return vec;
    };
    RCLCPP_INFO_STREAM(this->get_logger(), "Dumping betas with sizes: "
                                              << betas_.sizes());
    RCLCPP_INFO_STREAM(this->get_logger(), "Dumping body_pose with sizes: "
                                              << body_pose_.sizes());
    RCLCPP_INFO_STREAM(this->get_logger(), "Dumping global_orient with sizes: "
                                              << global_orient_.sizes());
    RCLCPP_INFO_STREAM(this->get_logger(), "Dumping transl with sizes: "
                                              << transl_.sizes());
    nlohmann::json j;
    j["betas"] = torch_to_std_vector(betas_);
    j["body_pose"] = torch_to_std_vector(body_pose_);
    j["global_orient"] = torch_to_std_vector(global_orient_);
    j["transl"] = torch_to_std_vector(transl_);

    std::ofstream out("optimized_params.json");
    out << std::setw(4) << j << std::endl;
    RCLCPP_INFO(this->get_logger(),
                "Dumped optimized parameters to optimized_params.json");
    out.close();
  }

  // -------------------- Member Variables --------------------
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
  std::cout << std::unitbuf; // Auto-flush every output
  rclcpp::init(argc, argv);

  auto node = std::make_shared<SMPLOptimizerNode>();
  node->initialize();
  node->run_optimization();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
