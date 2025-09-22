#include <open3d/Open3D.h>

#include <filesystem>
#include <fstream>
#include <iostream>
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

class SMPLOptimizerNode : public rclcpp::Node {
 public:
  SMPLOptimizerNode()
      : Node("smpl_optimizer_node"),
        device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
        chamfer_() {
    RCLCPP_INFO(this->get_logger(), "Using device: %s",
                device_ == torch::kCUDA ? "CUDA" : "CPU");

    // Declare parameters
    this->declare_parameter<std::string>("point_cloud_folder", "");
    this->declare_parameter<std::string>("smpl_model_path", "");
    this->declare_parameter<std::string>("config_path", "");

    // Get parameters
    input_folder_ = this->get_parameter("point_cloud_folder").as_string();
    model_path_ = this->get_parameter("smpl_model_path").as_string();
    config_path_ = this->get_parameter("config_path").as_string();

    if (!std::filesystem::exists(input_folder_) ||
        !std::filesystem::exists(model_path_) ||
        !std::filesystem::exists(config_path_)) {
      RCLCPP_FATAL(this->get_logger(), "One or more input paths do not exist!");
      throw std::runtime_error("Missing input files");
    }

    fitting_folder_ = input_folder_ + "_fitting";
    std::filesystem::create_directories(fitting_folder_);
    SMPL_TO_ROS_ = torch::tensor(
        {{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}},
        torch::TensorOptions().dtype(torch::kFloat64).device(device_));
    ROS_TO_SMPL_ = torch::inverse(SMPL_TO_ROS_);
    load_config();
    load_smpl_model();
  }

  void run_folder() {
    // Collect all cloud files first
    std::vector<std::filesystem::path> cloud_files;
    for (auto &p : std::filesystem::directory_iterator(input_folder_)) {
      if (!p.is_regular_file()) continue;
      std::string ext = p.path().extension().string();
      if (ext != ".ply" && ext != ".pcd") continue;
      cloud_files.push_back(p.path());
    }

    // Sort by filename, clouds must be in the form cloud_XXXX.ply
    std::sort(
        cloud_files.begin(), cloud_files.end(),
        [](const std::filesystem::path &a, const std::filesystem::path &b) {
          return a.filename().string() < b.filename().string();
        });

    // Process in order
    for (auto &path : cloud_files) {
      if (rclcpp::ok() == false) break;
      std::string filename = path.string();
      RCLCPP_INFO(this->get_logger(),
                  "--------------------\n--------------------\n Processing "
                  "cloud: %s\n\n",
                  filename.c_str());

      load_target_point_cloud(filename);
      RCLCPP_INFO(this->get_logger(), "[INFO] Point cloud loaded.");
      run_optimization(filename);
    }

    RCLCPP_INFO(this->get_logger(), "All clouds processed. Results in: %s",
                fitting_folder_.c_str());
  }

  void init_visualization() {
    vis_ = std::make_shared<SMPLRviz>(shared_from_this(), "map");
  }

 private:
  void load_config() {
    std::ifstream file(config_path_);
    file >> config_;
  }

  void load_smpl_model() {
    smpl_ = std::make_unique<smplx::SMPL>(model_path_.c_str(), device_);
    smpl_->eval();
    RCLCPP_INFO(this->get_logger(), "SMPL model loaded from: %s",
                model_path_.c_str());
    faces_ = smpl_->faces();
    RCLCPP_INFO(this->get_logger(), "num_betas = %d", smpl_->num_betas());
  }

  void load_target_point_cloud(const std::string &cloud_path) {
    auto cloud_ptr =
        open3d::io::CreatePointCloudFromFile(cloud_path, "auto", true);
    if (cloud_ptr->IsEmpty()) {
      throw std::runtime_error("Failed to load target point cloud: " +
                               cloud_path);
    }
    cloud_ptr = cloud_ptr->VoxelDownSample(0.01);
    cloud_ptr->RemoveStatisticalOutliers(40, 2.0);

    auto [vertices, colors] = open3d_pointcloud_to_tensor(*cloud_ptr);
    vertices_target_ = vertices.to(device_).unsqueeze(0);
    RCLCPP_INFO_STREAM(this->get_logger(),
                       "Point tensor size: " << vertices_target_.sizes());
    if (VISUALIZATION) {
      vis_->update_point_cloud(vertices, colors);
      RCLCPP_INFO(this->get_logger(), "Added target point cloud to RViz.");
    }
  }

  void run_optimization(const std::string &cloud_file) {
    RCLCPP_INFO(this->get_logger(), "Running optimization on: %s",
                cloud_file.c_str());

    // Load initial pose JSON
    std::string base_name = std::filesystem::path(cloud_file).stem().string();
    std::string params_json_file =
        std::filesystem::path(cloud_file).parent_path().string() + "/" +
        base_name + "_params.json";

    if (!std::filesystem::exists(params_json_file)) {
      throw std::runtime_error("Missing init pose JSON: " + params_json_file);
    }

    nlohmann::json j_init;
    {
      std::ifstream in(params_json_file);
      in >> j_init;
    }

    {
      torch::NoGradGuard no_grad;
      betas_ = torch::zeros({1, smpl_->num_betas()}, torch::kFloat64)
                   .to(device_)
                   .set_requires_grad(true);

      // body_pose
      auto body_pose_vec = j_init["smpl_body_pose"].get<std::vector<double>>();
      body_pose_ =
          torch::from_blob(body_pose_vec.data(),
                           {(int64_t)1, (int64_t)body_pose_vec.size()},
                           torch::TensorOptions().dtype(torch::kFloat64))
              .clone()
              .to(device_)
              .set_requires_grad(true);

      // transl
      auto transl_vec = j_init["smpl_transl"].get<std::vector<double>>();
      transl_ = torch::from_blob(transl_vec.data(),
                                 {(int64_t)1, (int64_t)transl_vec.size()},
                                 torch::TensorOptions().dtype(torch::kFloat64))
                    .clone()
                    .to(device_)
                    .set_requires_grad(true);
      // global_orient
      auto orient_vec = j_init["smpl_global_orient"].get<std::vector<double>>();
      global_orient_ =
          torch::from_blob(orient_vec.data(),
                           {(int64_t)1, (int64_t)orient_vec.size()},
                           torch::TensorOptions().dtype(torch::kFloat64))
              .clone()
              .to(device_)
              .set_requires_grad(true);
    }

    // Optimization stages
    optimize_stage({transl_}, config_["pos"]["lr"], config_["pos"]["it"],
                   "Position");
    optimize_stage({global_orient_}, config_["rot"]["lr"], config_["rot"]["it"],
                   "Rotation");
    optimize_stage({global_orient_, transl_}, config_["pos-rot"]["lr"],
                   config_["pos-rot"]["it"], "Position + Rotation");
    optimize_stage({body_pose_}, config_["body_pose"]["lr"],
                   config_["body_pose"]["it"], "Body Pose");
    if (is_first_frame_) {
      // optimize shape only for the first frame
      is_first_frame_ = false;
      optimize_stage({betas_}, config_["shape"]["lr"], config_["shape"]["it"],
                     "Shape");
    }
    // Dump JSON results
    std::string json_file = fitting_folder_ + "/" + base_name + "_smpl.json";
    dump_parameters_json(json_file);

    // Save mesh and keypoints
    auto output = smpl_->forward(
        smplx::betas(betas_), smplx::global_orient(global_orient_),
        smplx::body_pose(body_pose_), smplx::transl(transl_),
        smplx::return_verts(true));

    auto vertices_pred = output.vertices.value();
    save_mesh(vertices_pred, faces_,
              fitting_folder_ + "/" + base_name + "_mesh.ply");

    auto keypoints = output.joints.value();
    save_keypoints_as_pointcloud(
        keypoints, fitting_folder_ + "/" + base_name + "_keypoints.ply");
  }

  void optimize_stage(const std::vector<torch::Tensor> &params_to_optimize,
                      double lr, int steps, const std::string &stage_name,
                      double eps_thresh = 1e-4) {
    torch::optim::Adam optimizer(params_to_optimize,
                                 torch::optim::AdamOptions(lr));

    double loss_old = 0.0;
    std::vector<int> steps_vec(steps);
    std::iota(steps_vec.begin(), steps_vec.end(), 0);

    auto bar = ros_pbar::ros_pbar(steps_vec, this->get_logger());
    bar.set_prefix("Optimizing " + stage_name);

    for (int step : bar) {
      if (!rclcpp::ok()) break;

      optimizer.zero_grad();

      auto output = smpl_->forward(
          smplx::betas(betas_), smplx::global_orient(global_orient_),
          smplx::body_pose(body_pose_), smplx::transl(transl_),
          smplx::return_verts(true));
      auto vertices_pred = output.vertices.value();
      // --- Transform to ROS axes ---
      vertices_pred = torch::matmul(vertices_pred, SMPL_TO_ROS_);
      auto loss = chamfer_.forward(vertices_pred.to(torch::kFloat64),
                                   vertices_target_.to(torch::kFloat64), true);
      // loss += torch::mean(betas_ * betas_).sum() * 1e-1;
      loss.backward();
      optimizer.step();

      double loss_grad = std::abs(loss.item<double>() - loss_old);
      bar.set_suffix("Loss: " + std::to_string(loss.item<double>()) +
                     " | grad: " + std::to_string(loss_grad));

      if (loss_grad < eps_thresh) {
        RCLCPP_INFO(this->get_logger(), "Loss converged at step %d", step);
        break;
      }
      loss_old = loss.item<double>();

      if (VISUALIZATION &&
          (step % VISUALIZATION_UPDATE_EVERY == 0 || step == steps - 1)) {
        vis_->add_mesh(vertices_pred, faces_);
        vis_->update_visualization();
      }
    }

    RCLCPP_INFO(this->get_logger(), "------------------------------");
  }

  void dump_parameters_json(const std::string &filename) {
    auto torch_to_std_vector = [](const torch::Tensor &tensor) {
      std::vector<double> vec(tensor.numel());
      for (int i = 0; i < tensor[0].numel(); i++)
        vec[i] = tensor[0][i].item<double>();
      return vec;
    };
    nlohmann::json j;
    j["betas"] = torch_to_std_vector(betas_);
    j["body_pose"] = torch_to_std_vector(body_pose_);
    j["global_orient"] = torch_to_std_vector(global_orient_);
    j["transl"] = torch_to_std_vector(transl_);
    std::ofstream out(filename);
    out << std::setw(4) << j << std::endl;
    out.close();
  }

  void save_mesh(torch::Tensor vertices, torch::Tensor faces,
                 const std::string &filename) {
    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3i> fs;
    verts.reserve(vertices.size(1));
    for (int i = 0; i < vertices.size(1); ++i) {
      verts.emplace_back(vertices[0][i][0].item<double>(),
                         vertices[0][i][1].item<double>(),
                         vertices[0][i][2].item<double>());
    }
    for (int i = 0; i < faces.size(0); ++i) {
      fs.emplace_back(faces[i][0].item<int>(), faces[i][1].item<int>(),
                      faces[i][2].item<int>());
    }
    auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
    mesh->vertices_ = verts;
    mesh->triangles_ = fs;
    open3d::io::WriteTriangleMesh(filename, *mesh);
  }

  void save_keypoints_as_pointcloud(torch::Tensor keypoints,
                                    const std::string &filename) {
    keypoints = keypoints.cpu().squeeze(0);
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    for (int i = 0; i < keypoints.size(0); ++i) {
      cloud->points_.push_back(Eigen::Vector3d(keypoints[i][0].item<double>(),
                                               keypoints[i][1].item<double>(),
                                               keypoints[i][2].item<double>()));
    }
    open3d::io::WritePointCloud(filename, *cloud);
  }

  // Members
  std::unique_ptr<smplx::SMPL> smpl_;
  torch::Tensor vertices_target_, faces_;
  torch::Tensor betas_, body_pose_, global_orient_, transl_;
  ChamferDistance chamfer_;
  nlohmann::json config_;
  std::shared_ptr<SMPLRviz> vis_;
  torch::Device device_;

  std::string input_folder_, model_path_, config_path_;
  std::string fitting_folder_;
  torch::Tensor SMPL_TO_ROS_, ROS_TO_SMPL_;
  bool is_first_frame_ = true;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SMPLOptimizerNode>();
  if (VISUALIZATION) node->init_visualization();
  node->run_folder();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
