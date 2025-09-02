#include <open3d/Open3D.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "chamfer.h"
#include "json.hpp"
#include "o3d_converter.h"
#include "smplx.hpp"

using namespace torch::indexing;

class SMPLOptimizer {
public:
  SMPLOptimizer(smplx::SMPL &smpl_model, const torch::Tensor &target_vertices,
                const torch::Tensor &faces, torch::Device device)
      : smpl(smpl_model), faces_(faces), device_(device), chamfer_() {
    // Store target vertices on device with batch dimension
    vertices_target_ = target_vertices.unsqueeze(0).to(device_);

    // Create visualizer window once
    vis_.CreateVisualizerWindow("SMPL Fitting", 1920, 1080);

    // Add target point cloud geometry once
    auto target_cloud = tensor_to_open3d_pointcloud(target_vertices.squeeze(0));
    vis_.AddGeometry(target_cloud);

    // Initialize mesh from initial SMPL vertices (zeros)
    auto output_pred = smpl.forward(
        smplx::betas(
            torch::zeros({1, smpl.num_betas()}, torch::kFloat64).to(device_)),
        smplx::global_orient(torch::zeros({1, 3}, torch::kFloat64).to(device_)),
        smplx::body_pose(torch::zeros({1, 69}, torch::kFloat64).to(device_)),
        smplx::transl(torch::zeros({1, 3}, torch::kFloat64).to(device_)),
        smplx::return_verts(true));
    auto vertices_pred = output_pred.vertices.value();

    mesh_ptr_ = tensor_to_open3d_mesh(vertices_pred, faces_);
    ref_frame_ptr = open3d::geometry::TriangleMesh::CreateCoordinateFrame(
        0.5, Eigen::Vector3d(0, 0, 0));
    mesh_ptr_->ComputeVertexNormals();
    vis_.AddGeometry(mesh_ptr_);
    vis_.AddGeometry(ref_frame_ptr);

    vis_.UpdateGeometry();
    vis_.PollEvents();
    vis_.UpdateRender();
  }

  ~SMPLOptimizer() { vis_.DestroyVisualizerWindow(); }

  // Optimize the parameters in the given vector
  void optimize(const std::vector<torch::Tensor> &params_to_optimize,
                torch::Tensor &betas, torch::Tensor &body_pose,
                torch::Tensor &global_orient, torch::Tensor &transl,
                double learning_rate = 0.1, int steps = 200) {
    torch::optim::Adam optimizer(params_to_optimize,
                                 torch::optim::AdamOptions(learning_rate));

    for (int i = 0; i < steps; ++i) {
      optimizer.zero_grad();

      auto output =
          smpl.forward(smplx::betas(betas), smplx::global_orient(global_orient),
                       smplx::body_pose(body_pose), smplx::transl(transl));

      auto vertices_pred = output.vertices.value();

      auto loss = chamfer_.forward(vertices_pred.to(torch::kFloat64),
                                   vertices_target_.to(torch::kFloat64), true);
      // add beta squared regularization
      if (betas.defined()) {
        loss += 0.01 * betas.pow(2).sum();
      }
      loss.backward();
      optimizer.step();

      if (i % 10 == 0 || i == steps - 1) {
        std::cout << "Step " << i << ", Chamfer Loss: " << loss.item<float>()
                  << std::endl;
      }

      if (i % 1 == 0 || i == steps - 1) {
        update_mesh(vertices_pred);
      }
    }

    std::cout << "Optimization done. Close the window to continue."
              << std::endl;
    vis_.Run();
  }

private:
  void update_mesh(const torch::Tensor &vertices_pred) {
    auto verts = vertices_pred.squeeze(0).to(torch::kCPU).contiguous();

    mesh_ptr_->vertices_.clear();
    for (int64_t j = 0; j < verts.size(0); ++j) {
      auto v = verts[j];
      mesh_ptr_->vertices_.emplace_back(
          v[0].item<double>(), v[1].item<double>(), v[2].item<double>());
    }
    mesh_ptr_->ComputeVertexNormals();

    vis_.UpdateGeometry(mesh_ptr_);
    vis_.PollEvents();
    vis_.UpdateRender();
  }

  smplx::SMPL &smpl;
  torch::Tensor vertices_target_;
  torch::Tensor faces_;
  torch::Device device_;
  ChamferDistance chamfer_;
  open3d::visualization::Visualizer vis_;
  std::shared_ptr<open3d::geometry::TriangleMesh> ref_frame_ptr;
  std::shared_ptr<open3d::geometry::TriangleMesh> mesh_ptr_;
};

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model_path> <json_config_path> <point_cloud_path>"
              << std::endl;
    return 1;
  }

  std::string path = argv[1];
  if (!std::filesystem::exists(path)) {
    std::cerr << "Model path does not exist: " << path << std::endl;
    return 1;
  }
  std::string config_path = argv[2];
  if (!std::filesystem::exists(config_path)) {
    std::cerr << "Config path does not exist: " << config_path << std::endl;
    return 1;
  }
  std::string point_cloud_path = argv[3];
  if (!std::filesystem::exists(point_cloud_path)) {
    std::cerr << "Point cloud path does not exist: " << point_cloud_path
              << std::endl;
    return 1;
  }
  nlohmann::json config;
  std::ifstream file(config_path);
  file >> config;
  std::cout << "Using config: " << config.dump(4) << std::endl;

  torch::Device device =
      torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "Using device: " << device << "\n";

  smplx::SMPL smpl(path.c_str(), device);
  smpl.eval();

  const int batch_size = 1;

  // Load target point cloud
  auto cloud_ptr =
      open3d::io::CreatePointCloudFromFile(point_cloud_path, "auto", true);
  if (cloud_ptr->IsEmpty()) {
    std::cerr << "Failed to load point cloud from target.ply" << std::endl;
    return 1;
  }

  // downsample the point cloud
  cloud_ptr = cloud_ptr->VoxelDownSample(0.01);

  auto [vertices_target, colors_target] =
      open3d_pointcloud_to_tensor(*cloud_ptr);

  // Initialize parameters to optimize
  torch::Tensor betas =
      torch::zeros({batch_size, smpl.num_betas()}, torch::kFloat64)
          .to(device)
          .set_requires_grad(true);
  torch::Tensor body_pose = torch::zeros({batch_size, 69}, torch::kFloat64)
                                .to(device)
                                .set_requires_grad(true);
  torch::Tensor global_orient = torch::zeros({batch_size, 3}, torch::kFloat64)
                                    .to(device)
                                    .set_requires_grad(true);
  torch::Tensor transl =
      torch::zeros({batch_size, 3}, torch::kFloat64).to(device);
  double y_offset = config["pos"]["y_offset"];
  transl.index_put_({0, 1}, y_offset);
  transl.set_requires_grad(true);

  // Create optimizer class instance
  SMPLOptimizer optimizer(smpl, vertices_target, smpl.faces(), device);

  auto deg_to_rad = [](double degrees) { return degrees * M_PI / 180.0; };
  // prepare the model in A-pose
  // pose[3 * 17 : 3 * 18] = rot_x
  // Joint indices
  {
    // Init values
    torch::NoGradGuard no_grad;

    transl.index_put_({0, 0}, static_cast<double>(config["initial_transl"][0]));
    transl.index_put_({0, 1}, static_cast<double>(config["initial_transl"][1]));
    transl.index_put_({0, 2}, static_cast<double>(config["initial_transl"][2]));
    std::cout << "Loaded Transl: " << transl << std::endl;

    // Global orientation
    double r = static_cast<double>(config["initial_global_orient_rpy"][0]);
    double p = static_cast<double>(config["initial_global_orient_rpy"][1]);
    double y = static_cast<double>(config["initial_global_orient_rpy"][2]);
    // Convert RPY to rodrigues
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd aa(R);
    Eigen::Vector3d rod = aa.axis() * aa.angle();
    global_orient.index_put_({0, 0}, rod[0]);
    global_orient.index_put_({0, 1}, rod[1]);
    global_orient.index_put_({0, 2}, rod[2]);
    std::cout << "Loaded Global orient: " << global_orient << std::endl;

    double rot_x_r = deg_to_rad(config["shoulder_r_start"][0]);
    double rot_y_r = deg_to_rad(config["shoulder_r_start"][1]);
    double rot_z_r = deg_to_rad(config["shoulder_r_start"][2]);
    torch::Tensor shoulder_rot_r =
        torch::tensor({rot_x_r, rot_y_r, rot_z_r}, torch::kFloat64);
    double rot_x_l = deg_to_rad(config["shoulder_l_start"][0]);
    double rot_y_l = deg_to_rad(config["shoulder_l_start"][1]);
    double rot_z_l = deg_to_rad(config["shoulder_l_start"][2]);
    torch::Tensor shoulder_rot_l =
        torch::tensor({rot_x_l, rot_y_l, rot_z_l}, torch::kFloat64);
    const int left_shoulder = 15;
    const int right_shoulder = 16;
    body_pose.narrow(1, 3 * right_shoulder, 3).copy_(shoulder_rot_r);
    body_pose.narrow(1, 3 * left_shoulder, 3).copy_(shoulder_rot_l);
  }
  std::vector<torch::Tensor> params_to_optimize = {transl};
  optimizer.optimize(params_to_optimize, betas, body_pose, global_orient,
                     transl, config["pos"]["lr"], config["pos"]["it"]);
  std::cout << "Transl after optimization: " << transl << std::endl;

  params_to_optimize = {global_orient};
  optimizer.optimize(params_to_optimize, betas, body_pose, global_orient,
                     transl, config["rot"]["lr"], config["rot"]["it"]);
  std::cout << "Global orient after optimization: " << global_orient
            << std::endl;

  params_to_optimize = {global_orient, transl};
  optimizer.optimize(params_to_optimize, betas, body_pose, global_orient,
                     transl, config["pos-rot"]["lr"], config["pos-rot"]["it"]);

  std::cout << "Global orient and transl after optimization: " << global_orient
            << ", " << transl << std::endl;

  params_to_optimize = {body_pose};
  optimizer.optimize(params_to_optimize, betas, body_pose, global_orient,
                     transl, config["body_pose"]["lr"],
                     config["body_pose"]["it"]);

  std::cout << "Body pose after optimization: " << body_pose << std::endl;
  params_to_optimize = {betas};
  optimizer.optimize(params_to_optimize, betas, body_pose, global_orient,
                     transl, config["shape"]["lr"], config["shape"]["it"]);
  std::cout << "Betas after optimization: " << betas << std::endl;

  std::cout << "Final parameters:" << std::endl;
  // Save final predicted mesh
  auto final_output = smpl.forward(
      smplx::betas(betas.detach()), smplx::global_orient(global_orient),
      smplx::body_pose(body_pose), smplx::transl(transl));
  auto final_vertices = final_output.vertices.value();

  save_obj("predicted.obj", final_vertices.squeeze(0).cpu(),
           smpl.faces().cpu());

  return 0;
}
