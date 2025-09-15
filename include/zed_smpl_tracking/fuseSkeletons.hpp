#include "bodyStruct.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <sl/Camera.hpp>
#include <vector>

// --- Quaternion averaging using SVD ---
Eigen::Quaterniond
averageQuaternionsSVD(const std::vector<Eigen::Quaterniond> &quats) {
  if (quats.empty())
    return Eigen::Quaterniond::Identity();

  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  for (auto &q : quats) {
    Eigen::Vector4d v(q.w(), q.x(), q.y(), q.z());
    A += v * v.transpose();
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(A);
  Eigen::Vector4d avg =
      es.eigenvectors().col(3); // eigenvector with largest eigenvalue
  return Eigen::Quaterniond(avg(0), avg(1), avg(2), avg(3)).normalized();
}

// --- Transform 3D point ---
inline Eigen::Vector3d transformPoint(const Eigen::Vector3d &p,
                                      const Eigen::Matrix4d &T) {
  Eigen::Vector4d pt(p.x(), p.y(), p.z(), 1.0);
  Eigen::Vector4d pt_world = T * pt;
  return pt_world.head<3>();
}

// --- Transform quaternion by extrinsic rotation ---
inline Eigen::Quaterniond transformQuaternion(const Eigen::Quaterniond &q,
                                              const Eigen::Matrix4d &T) {
  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  return Eigen::Quaterniond(R) * q;
}

// --- Merge multiple Body objects with camera extrinsics ---
Body mergeBodiesWithExtrinsics(
    const std::vector<Body> &bodies,
    const std::vector<Eigen::Matrix4d> &T_cams_extrinsics) {
  Body merged;

  if (bodies.empty() || bodies.size() != T_cams_extrinsics.size())
    return merged;

  std::vector<Body> valid_bodies;
  valid_bodies.reserve(bodies.size());

  // --- Transform each body to world coordinates ---
  for (size_t i = 0; i < bodies.size(); i++) {
    const auto &T = T_cams_extrinsics[i];
    const auto &b = bodies[i];

    if (b.keypoints.empty())
      continue; // skip invalid

    Body body = b; // copy

    size_t num_joints = body.keypoints.size();

    // Transform keypoints and orientations
    for (size_t j = 0; j < num_joints; j++) {
      body.keypoints[j] = transformPoint(body.keypoints[j], T);
      body.local_orient[j] = body.local_orient[j]; 
    }

    // Transform root
    body.root_position = transformPoint(body.root_position, T);
    body.global_orientation = transformQuaternion(body.global_orientation, T);

    valid_bodies.push_back(std::move(body));
  }

  if (valid_bodies.empty())
    return merged;

  size_t num_joints = valid_bodies[0].keypoints.size();
  size_t num_bodies = valid_bodies.size();

  // --- Average per-joint positions and orientations ---
  for (size_t j = 0; j < num_joints; j++) {
    std::vector<Eigen::Vector3d> keypoint;
    std::vector<Eigen::Quaterniond> quats;

    for (auto &b : valid_bodies) {
      // check for NaN keypoints
      if (!std::isnan(b.keypoints[j].x()) && !std::isnan(b.keypoints[j].y()) &&
          !std::isnan(b.keypoints[j].z())) {
        keypoint.push_back(b.keypoints[j]);
      }
      // check for NaN quaternions
      if (!std::isnan(b.local_orient[j].x()) &&
          !std::isnan(b.local_orient[j].y()) &&
          !std::isnan(b.local_orient[j].z()) &&
          !std::isnan(b.local_orient[j].w())) {
        quats.push_back(b.local_orient[j]);
      }
    }

    for (const auto &kp : keypoint) {
      merged.keypoints[j] += kp;
    }
    if (!keypoint.empty()) {
      merged.keypoints[j] /= static_cast<double>(keypoint.size());
    } else {
      merged.keypoints[j] = Eigen::Vector3d::Zero();
    }
    merged.local_orient[j] = averageQuaternionsSVD(quats);
  }

  // --- Average root ---
  Eigen::Vector3d sum_root = Eigen::Vector3d::Zero();
  std::vector<Eigen::Quaterniond> root_quats;

  for (auto &b : valid_bodies) {
    sum_root += b.root_position;
    root_quats.push_back(b.global_orientation);
  }

  merged.root_position = sum_root / static_cast<double>(num_bodies);
  merged.global_orientation = averageQuaternionsSVD(root_quats);

  return merged;
}
