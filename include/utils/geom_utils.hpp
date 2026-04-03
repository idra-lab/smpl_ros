#pragma once
#include <Eigen/Dense>

inline bool isValidQuaternion(const Eigen::Quaterniond &q) {
  if (std::isnan(q.x()) || std::isnan(q.y()) ||
      std::isnan(q.z()) || std::isnan(q.w()))
    return false;
  if (std::isinf(q.x()) || std::isinf(q.y()) ||
      std::isinf(q.z()) || std::isinf(q.w()))
    return false;
  // Cannot normalize a quaternion with near-zero norm
  return q.norm() > 1e-6;
}
inline bool isValidPoint(const Eigen::Vector3d &p) { return p.allFinite(); }

// --- Check if a 3x3 matrix is a valid rotation (element of SO(3)) ---
// Verifies that all entries are finite, det ≈ +1, and R^T * R ≈ I.
inline bool isValidRotationMatrix(const Eigen::Matrix3d &R) {
  if (!R.allFinite())
    return false;
  if (std::abs(R.determinant() - 1.0) > 1e-4)
    return false;
  Eigen::Matrix3d err = R.transpose() * R - Eigen::Matrix3d::Identity();
  return err.norm() < 1e-4;
}
// --- Robust quaternion averaging via eigendecomposition (SVD method) ---
//
// The average rotation is the eigenvector corresponding to the largest
// eigenvalue of the 4x4 accumulation matrix A = sum(v_i * v_i^T).
//
// CRITICAL FIX — sign alignment:
//   q and -q represent the same rotation but are antipodal in 4D space.
//   If they are mixed without alignment, they cancel in the sum and the
//   result is garbage (this was the root cause of the random torso spinning).
//   We fix this by flipping any quaternion whose dot product with the
//   first quaternion (used as reference) is negative.

Eigen::Quaterniond
averageQuaternionsSVD(const std::vector<Eigen::Quaterniond> &quats) {
  if (quats.empty())
    return Eigen::Quaterniond::Identity();
  if (quats.size() == 1)
    return quats[0].normalized();

  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

  // Use the first valid quaternion as the sign reference
  Eigen::Vector4d ref(quats[0].w(), quats[0].x(), quats[0].y(), quats[0].z());
  ref.normalize();

  for (const auto &q : quats) {
    Eigen::Vector4d v(q.w(), q.x(), q.y(), q.z());
    if (v.norm() < 1e-6)
      continue; // skip degenerate quaternions
    v.normalize();

    // Align sign: flip if this quaternion points away from the reference
    // hemisphere — q and -q are the same rotation, but opposite in 4D.
    if (v.dot(ref) < 0.0)
      v = -v;

    A += v * v.transpose();
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(A);
  if (es.info() != Eigen::Success)
    return Eigen::Quaterniond::Identity();

  // Eigenvectors are sorted in ascending order; the last column corresponds
  // to the largest eigenvalue, which is the optimal average direction.
  Eigen::Vector4d avg = es.eigenvectors().col(3);
  Eigen::Quaterniond result(avg(0), avg(1), avg(2), avg(3));

  if (!isValidQuaternion(result))
    return Eigen::Quaterniond::Identity();

  return result.normalized();
}

// --- Transform a 3D point by a 4x4 homogeneous matrix ---
// Divides by the homogeneous w component to handle any projective component,
// though for standard affine extrinsics w will always be 1.
inline Eigen::Vector3d transformPoint(const Eigen::Vector3d &p,
                                      const Eigen::Matrix4d &T) {
  if (!isValidPoint(p) || !T.allFinite())
    return Eigen::Vector3d::Zero();

  Eigen::Vector4d pt(p.x(), p.y(), p.z(), 1.0);
  Eigen::Vector4d pt_world = T * pt;

  // Guard against degenerate projective division
  if (std::abs(pt_world.w()) < 1e-9)
    return Eigen::Vector3d::Zero();

  return pt_world.head<3>() / pt_world.w();
}

// --- Rotate a quaternion by the rotational part of a 4x4 extrinsic matrix ---
// Extracts the upper-left 3x3 block and projects it onto SO(3) via SVD if it
// is not already a valid rotation matrix (e.g. due to floating-point drift).
inline Eigen::Quaterniond transformQuaternion(const Eigen::Quaterniond &q,
                                              const Eigen::Matrix4d &T) {
  if (!isValidQuaternion(q) || !T.allFinite())
    return Eigen::Quaterniond::Identity();

  Eigen::Matrix3d R = T.block<3, 3>(0, 0);

  if (!isValidRotationMatrix(R)) {
    // Project onto the nearest rotation matrix in SO(3) using SVD:
    //   R = U * V^T  (drop singular values, keep the orthonormal frame)
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU |
                                                 Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();

    // If det = -1 the SVD returned a reflection; flip the last singular vector
    if (R.determinant() < 0)
      R = svd.matrixU() * Eigen::DiagonalMatrix<double, 3>(1, 1, -1) *
          svd.matrixV().transpose();
  }

  return (Eigen::Quaterniond(R) * q).normalized();
}