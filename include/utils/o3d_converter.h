
#ifndef O3D_CONVERTER_H
#define O3D_CONVERTER_H

#include <open3d/Open3D.h>
#include <torch/torch.h>

#include <memory>

std::tuple<torch::Tensor, torch::Tensor> open3d_mesh_to_tensor(
    const open3d::geometry::TriangleMesh &mesh) {
  // Convert Open3D mesh to PyTorch tensors
  torch::Tensor vertices =
      torch::empty({static_cast<int64_t>(mesh.vertices_.size()), 3}, torch::kFloat64);
  torch::Tensor faces =
      torch::empty({static_cast<int64_t>(mesh.triangles_.size()), 3}, torch::kInt64);

  for (size_t i = 0; i < mesh.vertices_.size(); ++i) {
    const auto &v = mesh.vertices_[i];
    vertices[i] = torch::tensor({v[0], v[1], v[2]}, torch::kFloat64);
  }

  for (size_t i = 0; i < mesh.triangles_.size(); ++i) {
    const auto &f = mesh.triangles_[i];
    faces[i] = torch::tensor({f[0], f[1], f[2]}, torch::kInt64);
  }

  return {vertices, faces};
}
std::tuple<torch::Tensor, torch::Tensor> open3d_pointcloud_to_tensor(
    const open3d::geometry::PointCloud &cloud) {
  // Convert Open3D point cloud with colors to PyTorch tensors
  torch::Tensor points =
      torch::empty({static_cast<int64_t>(cloud.points_.size()), 3}, torch::kFloat64);
  torch::Tensor colors =
      torch::empty({static_cast<int64_t>(cloud.colors_.size()), 3}, torch::kFloat64);

  for (size_t i = 0; i < cloud.points_.size(); ++i) {
    const auto &p = cloud.points_[i];
    points[i] = torch::tensor({p[0], p[1], p[2]}, torch::kFloat64);
  }

  for (size_t i = 0; i < cloud.colors_.size(); ++i) {
    const auto &c = cloud.colors_[i];
    colors[i] = torch::tensor({c[0], c[1], c[2]}, torch::kFloat64);
  }
  return {points, colors};
}

std::shared_ptr<open3d::geometry::TriangleMesh> tensor_to_open3d_mesh(
    const torch::Tensor &vertices, const torch::Tensor &faces) {
  auto mesh = std::make_shared<open3d::geometry::TriangleMesh>();
  auto verts = vertices.squeeze(0).to(torch::kCPU).contiguous();
  auto tris = faces.to(torch::kCPU).contiguous();

  for (int64_t i = 0; i < verts.size(0); ++i) {
    auto v = verts[i];
    mesh->vertices_.emplace_back(v[0].item<double>(), v[1].item<double>(),
                                 v[2].item<double>());
  }

  for (int64_t i = 0; i < tris.size(0); ++i) {
    auto f = tris[i];
    mesh->triangles_.emplace_back(
        Eigen::Vector3i(f[0].item<int>(), f[1].item<int>(), f[2].item<int>()));
  }

  mesh->ComputeVertexNormals();
  return mesh;
}
std::shared_ptr<open3d::geometry::PointCloud> tensor_to_open3d_pointcloud(
    const torch::Tensor &vertices) {
  auto cloud = std::make_shared<open3d::geometry::PointCloud>();
  auto verts = vertices.squeeze(0).to(torch::kCPU).contiguous();

  cloud->points_.resize(verts.size(0));
  for (int64_t i = 0; i < verts.size(0); ++i) {
    auto v = verts[i];
    cloud->points_[i] = Eigen::Vector3d(
        v[0].item<double>(), v[1].item<double>(), v[2].item<double>());
  }
  cloud->PaintUniformColor(Eigen::Vector3d(0.1, 0.1, 0.8));
  return cloud;
}

void save_obj(const std::string &filename, const torch::Tensor &vertices,
              const torch::Tensor &faces) {
  std::ofstream obj_file(filename);
  if (!obj_file.is_open()) {
    throw std::runtime_error("Could not open OBJ file for writing.");
  }
  obj_file << std::fixed << std::setprecision(8);

  for (int64_t i = 0; i < vertices.size(0); ++i) {
    auto v = vertices[i];
    obj_file << "v " << v[0].item<float>() << " " << v[1].item<float>() << " "
             << v[2].item<float>() << "\n";
  }

  for (int64_t i = 0; i < faces.size(0); ++i) {
    auto f = faces[i];
    obj_file << "f " << f[0].item<int64_t>() + 1 << " "
             << f[1].item<int64_t>() + 1 << " " << f[2].item<int64_t>() + 1
             << "\n";
  }
}

#endif  // O3D_CONVERTER_H