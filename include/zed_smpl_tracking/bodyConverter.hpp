#include "sl/Camera.hpp"
#include "bodyStruct.hpp"

std::vector<Body> extractBodyData(const std::vector<sl::BodyData> &zed_bodies,
                                  const std::array<int, 24> &SMPL_TO_ZED) {
  std::vector<Body> bodies;
  bodies.reserve(zed_bodies.size());

  for (const auto &zed_body : zed_bodies) {
    Body body;

    // Root
    const auto &root = zed_body.keypoint[2];
    body.root_position << root.x, root.y, root.z;

    const auto &gro = zed_body.global_root_orientation;
    body.global_orientation = Eigen::Quaterniond(gro.w, gro.x, gro.y, gro.z);

    // Local orientations
    const auto &q = zed_body.local_orientation_per_joint;
    for (int j = 1; j < 24; ++j) {
      int zed_idx = SMPL_TO_ZED[j];
      body.local_orient[j] = Eigen::Quaterniond(q[zed_idx].w, q[zed_idx].x,
                                                q[zed_idx].y, q[zed_idx].z);
    }

    // Keypoints
    for (int j = 0; j < 24; ++j) {
      int zed_idx = SMPL_TO_ZED[j];
      const auto &kp = zed_body.keypoint[zed_idx];
      body.keypoints[j] << kp.x, kp.y, kp.z;
    }

    bodies.emplace_back(std::move(body));
  }

  return bodies;
}