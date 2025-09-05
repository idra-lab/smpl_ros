#include <sl/Camera.hpp>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <rclcpp/rclcpp.hpp>

// --- Quaternion averaging using SVD ---
Eigen::Quaternionf averageQuaternionsSVD(const std::vector<Eigen::Quaternionf> &quats) {
    if (quats.empty()) return Eigen::Quaternionf::Identity();

    Eigen::Matrix4f A = Eigen::Matrix4f::Zero();
    for (auto &q : quats) {
        Eigen::Vector4f v(q.w(), q.x(), q.y(), q.z());
        A += v * v.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4f> es(A);
    Eigen::Vector4f avg = es.eigenvectors().col(3); // largest eigenvalue
    return Eigen::Quaternionf(avg(0), avg(1), avg(2), avg(3)).normalized();
}

// --- Transform 3D point ---
inline sl::float3 transformPoint(const sl::float3 &p, const Eigen::Matrix4d &T) {
    Eigen::Vector4d pt(p.x, p.y, p.z, 1.0);
    Eigen::Vector4d pt_world = T * pt;
    return sl::float3{static_cast<float>(pt_world.x()),
                      static_cast<float>(pt_world.y()),
                      static_cast<float>(pt_world.z())};
}

// --- Transform quaternion by rotation from extrinsic ---
inline sl::float4 transformQuaternion(const sl::float4 &q, const Eigen::Matrix4d &T) {
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Quaterniond q_orig(q.w, q.x, q.y, q.z);
    Eigen::Quaterniond q_new = Eigen::Quaterniond(R) * q_orig;
    return sl::float4{static_cast<float>(q_new.x()),
                      static_cast<float>(q_new.y()),
                      static_cast<float>(q_new.z()),
                      static_cast<float>(q_new.w())};
}

// --- Merge multiple BodyData objects with camera extrinsics ---
sl::BodyData mergeBodiesWithExtrinsics(const std::vector<sl::BodyData> &bodies,
                                       const std::vector<Eigen::Matrix4d> &T_cams_extrinsics) {
    sl::BodyData merged;

    if (bodies.empty() || bodies.size() != T_cams_extrinsics.size()) return merged;

    std::vector<sl::BodyData> valid_bodies;

    // --- Transform each body to world coordinates and filter invalid ---
    for (size_t i = 0; i < bodies.size(); i++) {
        const auto &T = T_cams_extrinsics[i];
        const auto &b = bodies[i];

        if (b.keypoint.empty()) continue; // skip bodies with no keypoints

        sl::BodyData body = b; // copy

        size_t num_joints = body.keypoint.size();

        // Ensure per-joint vectors are correctly sized
        if (body.local_position_per_joint.size() != num_joints)
            body.local_position_per_joint.resize(num_joints, sl::float3{0,0,0});
        if (body.local_orientation_per_joint.size() != num_joints)
            body.local_orientation_per_joint.resize(num_joints, sl::float4{0,0,0,1});

        // Transform keypoints and per-joint data
        for (size_t j = 0; j < num_joints; j++) {
            body.keypoint[j] = transformPoint(body.keypoint[j], T);
            body.local_position_per_joint[j] = transformPoint(body.local_position_per_joint[j], T);
            body.local_orientation_per_joint[j] = transformQuaternion(body.local_orientation_per_joint[j], T);
        }

        // Transform root
        body.position = transformPoint(body.position, T);
        body.global_root_orientation = transformQuaternion(body.global_root_orientation, T);

        valid_bodies.push_back(body);
    }

    if (valid_bodies.empty()) return merged; // nothing to merge

    size_t num_joints = valid_bodies[0].keypoint.size();
    size_t num_bodies = valid_bodies.size();

    // --- Resize merged BodyData ---
    merged.keypoint.resize(num_joints);
    merged.local_position_per_joint.resize(num_joints);
    merged.local_orientation_per_joint.resize(num_joints);

    // --- Average per-joint positions and orientations ---
    for (size_t j = 0; j < num_joints; j++) {
        Eigen::Vector3f sum_pos(0,0,0);
        Eigen::Vector3f sum_local(0,0,0);
        std::vector<Eigen::Quaternionf> quats;

        for (auto &b : valid_bodies) {
            sum_pos += Eigen::Vector3f(b.keypoint[j].x, b.keypoint[j].y, b.keypoint[j].z);
            sum_local += Eigen::Vector3f(b.local_position_per_joint[j].x,
                                         b.local_position_per_joint[j].y,
                                         b.local_position_per_joint[j].z);
            quats.emplace_back(b.local_orientation_per_joint[j].w,
                               b.local_orientation_per_joint[j].x,
                               b.local_orientation_per_joint[j].y,
                               b.local_orientation_per_joint[j].z);
        }

        // Average
        merged.keypoint[j].x = sum_pos.x() / num_bodies;
        merged.keypoint[j].y = sum_pos.y() / num_bodies;
        merged.keypoint[j].z = sum_pos.z() / num_bodies;

        merged.local_position_per_joint[j].x = sum_local.x() / num_bodies;
        merged.local_position_per_joint[j].y = sum_local.y() / num_bodies;
        merged.local_position_per_joint[j].z = sum_local.z() / num_bodies;

        Eigen::Quaternionf avg_q = averageQuaternionsSVD(quats);
        merged.local_orientation_per_joint[j].x = avg_q.x();
        merged.local_orientation_per_joint[j].y = avg_q.y();
        merged.local_orientation_per_joint[j].z = avg_q.z();
        merged.local_orientation_per_joint[j].w = avg_q.w();
    }

    // --- Average root ---
    Eigen::Vector3f root_pos(0,0,0);
    std::vector<Eigen::Quaternionf> root_quats;
    for (auto &b : valid_bodies) {
        root_pos += Eigen::Vector3f(b.position.x, b.position.y, b.position.z);
        root_quats.emplace_back(b.global_root_orientation.w,
                                b.global_root_orientation.x,
                                b.global_root_orientation.y,
                                b.global_root_orientation.z);
    }

    merged.position.x = root_pos.x() / num_bodies;
    merged.position.y = root_pos.y() / num_bodies;
    merged.position.z = root_pos.z() / num_bodies;

    Eigen::Quaternionf avg_root = averageQuaternionsSVD(root_quats);
    merged.global_root_orientation.x = avg_root.x();
    merged.global_root_orientation.y = avg_root.y();
    merged.global_root_orientation.z = avg_root.z();
    merged.global_root_orientation.w = avg_root.w();

    RCLCPP_INFO(rclcpp::get_logger("mergeBodiesWithExtrinsics"), "Merged %zu bodies into one with %zu keypoints.", num_bodies, merged.keypoint.size());

    return merged;
}
