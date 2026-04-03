#include "bodyStruct.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <sl/Camera.hpp>
#include <vector>
#include "utils/geom_utils.hpp"

// --- Merge multiple Body detections using per-camera extrinsic matrices ---
//
// Each body is first transformed into a common world frame using its
// corresponding extrinsic matrix. Joint positions and orientations are then
// averaged across all valid observations.
//
// Robustness measures applied throughout:
//   - Extrinsic matrices with non-finite entries are skipped.
//   - Bodies with empty keypoint arrays are skipped.
//   - Per-joint NaN/Inf keypoints and quaternions are excluded from averaging.
//   - merged.keypoints is zero-initialised before accumulation.
//   - Root position and orientation use the same NaN-filtered averaging path.
Body mergeBodiesWithExtrinsics(
    const std::vector<Body> &bodies,
    const std::vector<Eigen::Matrix4d> &T_cams_extrinsics) {

  Body merged;

  if (bodies.empty()) {
    std::cerr << "Warning: no bodies provided to mergeBodiesWithExtrinsics, "
                 "returning empty Body.\n";
    return merged;
  }

  std::vector<Body> valid_bodies;
  valid_bodies.reserve(bodies.size());

  // --- Transform each body into the world frame ---
  for (size_t i = 0; i < bodies.size(); ++i) {
    if (i >= T_cams_extrinsics.size()) {
      std::cerr << "Warning: body index " << i
                << " has no matching extrinsic matrix, skipping.\n";
      continue;
    }

    const Eigen::Matrix4d &T = T_cams_extrinsics[i];

    if (!T.allFinite()) {
      std::cerr << "Warning: extrinsic matrix " << i
                << " contains non-finite values, skipping.\n";
      continue;
    }

    const Body &b = bodies[i];
    if (b.keypoints.empty())
      continue; // no usable data

    Body body = b; // working copy
    const size_t num_joints = body.keypoints.size();

    for (size_t j = 0; j < num_joints; ++j) {
      body.keypoints[j] = transformPoint(body.keypoints[j], T);
      // local_orient is defined in the local joint frame and is intentionally
      // left untransformed here. Uncomment the line below if your pipeline
      // stores local_orient in camera space and needs world-space conversion:
      // body.local_orient[j] = transformQuaternion(body.local_orient[j], T);
    }

    body.root_position = transformPoint(body.root_position, T);
    body.global_orientation = transformQuaternion(body.global_orientation, T);

    valid_bodies.push_back(std::move(body));
  }

  if (valid_bodies.empty())
    return merged;

  const size_t num_joints = valid_bodies[0].keypoints.size();

  // Explicitly zero-initialise merged arrays before accumulation.
  // Without this, operator+= below would accumulate into garbage memory.
  // merged.keypoints.assign(num_joints, Eigen::Vector3d::Zero());
  // merged.local_orient.assign(num_joints, Eigen::Quaterniond::Identity());
  // --- Inizializzazione degli array fissi ---
  // std::array non ha .resize(), usiamo .fill() per resettare i valori
  merged.keypoints.fill(Eigen::Vector3d::Zero());
  merged.local_orient.fill(Eigen::Quaterniond::Identity());

  // Se vuoi essere extra sicuro che num_joints non superi 24:
  const size_t max_joints = merged.keypoints.size(); // che è 24
  const size_t joints_to_process = std::min(num_joints, max_joints);

  // --- Average per-joint positions and orientations ---
  for (size_t j = 0; j < joints_to_process; ++j) {
    std::vector<Eigen::Vector3d> valid_kps;
    std::vector<Eigen::Quaterniond> valid_quats;

    for (const auto &b : valid_bodies) {
      if (j < b.keypoints.size() && isValidPoint(b.keypoints[j]))
        valid_kps.push_back(b.keypoints[j]);

      if (j < b.local_orient.size() && isValidQuaternion(b.local_orient[j]))
        valid_quats.push_back(b.local_orient[j].normalized());
    }

    if (!valid_kps.empty()) {
      Eigen::Vector3d sum = Eigen::Vector3d::Zero();
      for (const auto &kp : valid_kps)
        sum += kp;
      merged.keypoints[j] = sum / static_cast<double>(valid_kps.size());
    }
    // Falls back to Identity if valid_quats is empty
    merged.local_orient[j] = averageQuaternionsSVD(valid_quats);
  }

  // --- Average root position and global orientation ---
  std::vector<Eigen::Vector3d> valid_roots;
  std::vector<Eigen::Quaterniond> root_quats;

  for (const auto &b : valid_bodies) {
    if (isValidPoint(b.root_position))
      valid_roots.push_back(b.root_position);

    if (isValidQuaternion(b.global_orientation))
      root_quats.push_back(b.global_orientation.normalized());
  }

  if (!valid_roots.empty()) {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto &r : valid_roots)
      sum += r;
    merged.root_position = sum / static_cast<double>(valid_roots.size());
  }

  // Falls back to Identity if no valid global orientations were found
  merged.global_orientation = averageQuaternionsSVD(root_quats);

  return merged;
}