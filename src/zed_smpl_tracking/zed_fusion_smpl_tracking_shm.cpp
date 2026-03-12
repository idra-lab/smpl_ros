/**
 * smpl_shm_publisher — ZED SMPL tracking con shared memory invece di ROS2.
 *
 * Pubblica su /dev/shm:
 *   /zed_pointclouds  → point cloud per camera  (vedi SHMPCHeader/SHMPCSlot)
 *   /zed_smpl         → parametri SMPL          (vedi
 * SHMSMPLHeader/SHMSMPLSlot)
 */

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "utils/json.hpp"
#include "utils/shm_transport.h"
#include "utils/voxel_filter.h"
#include "yolov8_seg.h"
#include "zed_smpl_tracking/ClientPublisher.hpp"
#include "zed_smpl_tracking/bodyConverter.hpp"
#include "zed_smpl_tracking/fuseSkeletons.hpp"
// #include "zed_smpl_tracking/utils.hpp"
#include "utils/constants.hpp"
// ─── helpers ─────────────────────────────────────────────────────────────────

static std::atomic<bool> g_running{true};

static void sigint_handler(int) { g_running = false; }

static uint64_t now_ns() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(
          high_resolution_clock::now().time_since_epoch())
          .count());
}

// Open/create a SHM segment and ftruncate to size. Returns mmap'd pointer.
static void *shm_create(const char *name, size_t size) {
  shm_unlink(name); // remove stale segment
  int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
  if (fd < 0) {
    perror("shm_open");
    exit(1);
  }
  if (ftruncate(fd, static_cast<off_t>(size)) < 0) {
    perror("ftruncate");
    exit(1);
  }
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
  close(fd);
  memset(ptr, 0, size);
  return ptr;
}

Eigen::Matrix4d slTransformToEigen(const sl::Transform &T) {
  Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();

  // Rotation part
  sl::Matrix3f r = T.getRotationMatrix(); // returns sl::Matrix3f
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      mat(i, j) = static_cast<double>(r(i, j));

  // Translation part
  sl::Translation t = T.getTranslation();
  mat(0, 3) = static_cast<double>(t.x);
  mat(1, 3) = static_cast<double>(t.y);
  mat(2, 3) = static_cast<double>(t.z);

  return mat;
}

// ─── SHM writers ─────────────────────────────────────────────────────────────

/**
 * Write a point cloud (vector of (xyz, rgb, normal) tuples) into the ring
 * buffer for camera `cam_idx`.  Uses a simple state machine on SHMPCSlot::state
 * (0=free → 1=writing → 2=ready) so the Python reader never sees partial data.
 */
static void write_pointcloud(
    void *shm_pc, int cam_idx,
    const std::vector<
        std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>> &cloud,
    uint32_t frame_id) {
  auto *hdr = shm_pc_header(shm_pc);

  // Pick next slot (round-robin by write_seq)
  uint64_t seq = hdr->write_seq[cam_idx].load(std::memory_order_relaxed);
  int slot = static_cast<int>(seq % SHM_PC_SLOTS);

  SHMPCSlot *s = shm_pc_slot(shm_pc, cam_idx, slot);
  SHMFloat *data = shm_pc_data(shm_pc, cam_idx, slot);

  // Mark slot as being written
  s->state.store(1, std::memory_order_release);

  uint32_t n = static_cast<uint32_t>(
      std::min(cloud.size(), static_cast<size_t>(SHM_MAX_POINTS)));
  s->num_points = n;
  s->timestamp_ns = now_ns();
  s->cam_idx = static_cast<uint32_t>(cam_idx);
  s->frame_id = frame_id;

  // Interleaved: x y z r g b nx ny nz
  for (uint32_t i = 0; i < n; ++i) {
    const auto &[xyz, rgb, nrm] = cloud[i];
    float *p = data + i * SHM_POINT_FLOATS;
    p[0] = static_cast<float>(xyz.x());
    p[1] = static_cast<float>(xyz.y());
    p[2] = static_cast<float>(xyz.z());
    p[3] = static_cast<float>(rgb.x());
    p[4] = static_cast<float>(rgb.y());
    p[5] = static_cast<float>(rgb.z());
    p[6] = static_cast<float>(nrm.x());
    p[7] = static_cast<float>(nrm.y());
    p[8] = static_cast<float>(nrm.z());
  }

  // Mark ready, then bump sequence
  s->state.store(2, std::memory_order_release);
  hdr->write_seq[cam_idx].fetch_add(1, std::memory_order_release);

  // Wake futex listeners (Linux only – optional, comment out if not needed)
  // syscall(SYS_futex, &hdr->write_seq[cam_idx], FUTEX_WAKE, INT_MAX, ...);
}

/**
 * Write SMPL parameters into the ring buffer.
 * The message layout mirrors smpl_msgs/Smpl but as raw floats.
 */
static void write_smpl(void *shm_smpl, const Body &fused_body,
                       const Eigen::Matrix4d &T_SMPL_TO_ROS,
                       const std::vector<double> &betas, uint32_t frame_id,
                       bool valid) {
  auto *hdr = shm_smpl_header(shm_smpl);
  uint64_t seq = hdr->write_seq.load(std::memory_order_relaxed);
  int slot = static_cast<int>(seq % SHM_SMPL_SLOTS);

  SHMSMPLSlot *s = shm_smpl_slot(shm_smpl, slot);
  s->state.store(1, std::memory_order_release);

  s->valid = valid ? 1u : 0u;
  s->timestamp_ns = now_ns();
  s->frame_id = frame_id;

  if (valid) {
    // Betas (shape)
    for (int i = 0; i < SHM_SMPL_SHAPE_PARAMS; ++i)
      s->betas[i] = (i < static_cast<int>(betas.size()))
                        ? static_cast<float>(betas[i])
                        : 0.f;

    // Translation: root_position transformed to ROS frame
    Eigen::Vector4d t_body(fused_body.root_position.x(),
                           fused_body.root_position.y(),
                           fused_body.root_position.z(), 1.0);
    Eigen::Vector4d t_ros = T_SMPL_TO_ROS * t_body;
    s->trans[0] = static_cast<float>(t_ros.x());
    s->trans[1] = static_cast<float>(t_ros.y());
    s->trans[2] = static_cast<float>(t_ros.z());

    // Global orientation: quaternion → axis-angle
    // global_orientation is already in world frame after
    // mergeBodiesWithExtrinsics
    Eigen::Quaterniond q_ros =
        Eigen::Quaterniond(T_SMPL_TO_ROS.block<3, 3>(0, 0)) *
        fused_body.global_orientation;
    q_ros.normalize();
    Eigen::AngleAxisd aa(q_ros);
    Eigen::Vector3d axis_angle = aa.axis() * aa.angle();
    s->global_orient[0] = static_cast<float>(axis_angle.x());
    s->global_orient[1] = static_cast<float>(axis_angle.y());
    s->global_orient[2] = static_cast<float>(axis_angle.z());

    // Per-joint local orientations (24 joints × 3 axis-angle = 72 floats)
    // local_orient is std::vector<Eigen::Quaterniond> or similar — adapt if
    // needed
    for (int j = 0;
         j < 24 && j < static_cast<int>(fused_body.local_orient.size()); ++j) {
      Eigen::AngleAxisd aa_j(fused_body.local_orient[j].normalized());
      Eigen::Vector3d v = aa_j.axis() * aa_j.angle();
      s->pose[j * 3 + 0] = static_cast<float>(v.x());
      s->pose[j * 3 + 1] = static_cast<float>(v.y());
      s->pose[j * 3 + 2] = static_cast<float>(v.z());
    }
    // Zero-fill remaining slots if fewer than 24 joints
    for (int j = static_cast<int>(fused_body.local_orient.size()); j < 24;
         ++j) {
      s->pose[j * 3 + 0] = 0.f;
      s->pose[j * 3 + 1] = 0.f;
      s->pose[j * 3 + 2] = 0.f;
    }
  }

  s->state.store(2, std::memory_order_release);
  hdr->write_seq.fetch_add(1, std::memory_order_release);
}

// ─── main
// ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  signal(SIGINT, sigint_handler);
  signal(SIGTERM, sigint_handler);

  // ── Config (previously ROS parameters, now CLI args or hardcoded) ─────────
  // Override via env vars for flexibility:
  //   ZED_CALIB_FILE, ZED_YOLO_PATH, ZED_RESOLUTION,
  //   ZED_VOXEL_SIZE, ZED_ERODE_KERNEL, ZED_OVERLAY_YOLO, ZED_VISUALIZE
  auto getenv_str = [](const char *k, const char *def) -> std::string {
    const char *v = std::getenv(k);
    return v ? std::string(v) : std::string(def);
  };
  auto getenv_bool = [](const char *k, bool def) -> bool {
    const char *v = std::getenv(k);
    if (!v)
      return def;
    return std::string(v) == "1" || std::string(v) == "true";
  };
  auto getenv_double = [](const char *k, double def) -> double {
    const char *v = std::getenv(k);
    return v ? std::stod(v) : def;
  };
  auto getenv_int = [](const char *k, int def) -> int {
    const char *v = std::getenv(k);
    return v ? std::stoi(v) : def;
  };

  std::string calib_file =
      getenv_str("ZED_CALIB_FILE", "two_cams_fusion_config.json");
  std::string yolo_model_path = getenv_str("ZED_YOLO_PATH", "yolov8s-seg.onnx");
  std::string resolution_str = getenv_str("ZED_RESOLUTION", "1280x720");
  double voxel_size = getenv_double("ZED_VOXEL_SIZE", 0.015);
  int erode_kernel = getenv_int("ZED_ERODE_KERNEL", 18);
  bool overlay_yolo_mask = getenv_bool("ZED_OVERLAY_YOLO", true);
  bool visualize_image = getenv_bool("ZED_VISUALIZE", true);
  bool publish_body = getenv_bool("ZED_PUBLISH_BODY", true);

  std::cout << "[SHM] calib:       " << calib_file << "\n"
            << "[SHM] yolo:        " << yolo_model_path << "\n"
            << "[SHM] resolution:  " << resolution_str << "\n"
            << "[SHM] voxel_size:  " << voxel_size << "\n"
            << "[SHM] erode:       " << erode_kernel << "\n";

  // ── ZED resolution ────────────────────────────────────────────────────────
  int width = std::stoi(resolution_str.substr(0, resolution_str.find('x')));
  int height = std::stoi(resolution_str.substr(resolution_str.find('x') + 1));
  sl::RESOLUTION resolution;
  switch (height) {
  case 1242:
    resolution = sl::RESOLUTION::HD2K;
    break;
  case 1536:
    resolution = sl::RESOLUTION::HD1536;
    break;
  case 1080:
    resolution = sl::RESOLUTION::HD1080;
    break;
  case 1200:
    resolution = sl::RESOLUTION::HD1200;
    break;
  case 720:
    resolution = sl::RESOLUTION::HD720;
    break;
  case 376:
    resolution = sl::RESOLUTION::VGA;
    break;
  default:
    std::cerr << "[SHM] Unsupported resolution height: " << height << "\n";
    return EXIT_FAILURE;
  }

  constexpr sl::COORDINATE_SYSTEM ROS_COORDINATE_SYSTEM =
      sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  constexpr sl::UNIT UNIT = sl::UNIT::METER;

  Eigen::Matrix4d T_SMPL_TO_ROS = smpl_to_ros_transform();

  // ── ZED cameras setup ─────────────────────────────────────────────────────
  auto configurations =
      sl::readFusionConfigurationFile(calib_file, ROS_COORDINATE_SYSTEM, UNIT);
  if (configurations.empty()) {
    std::cerr << "[SHM] No ZED configurations found.\n";
    return EXIT_FAILURE;
  }

  Trigger trigger;
  std::vector<ClientPublisher> clients(configurations.size());
  int id_ = 0, gpu_id = 0, nb_gpu = 0;
  cudaGetDeviceCount(&nb_gpu);

  for (auto &conf : configurations) {
    if (conf.communication_parameters.getType() ==
        sl::CommunicationParameters::COMM_TYPE::INTRA_PROCESS) {
      gpu_id = id_ % nb_gpu;
      if (!clients[id_].open(conf.input_type, ROS_COORDINATE_SYSTEM, resolution,
                             &trigger, gpu_id))
        continue;
      id_++;
    }
  }
  for (auto &c : clients)
    c.start();

  // ── Fusion ────────────────────────────────────────────────────────────────
  sl::InitFusionParameters init_params;
  init_params.coordinate_units = UNIT;
  init_params.coordinate_system = ROS_COORDINATE_SYSTEM;
  init_params.verbose = true;
  init_params.maximum_working_resolution =
      sl::Resolution(std::max(1280, width), std::max(720, height));

  sl::Fusion fusion;
  fusion.init(init_params);

  std::vector<Eigen::Matrix4d> T_cams_extrinsics;
  std::vector<sl::CameraIdentifier> cameras;

  for (auto &conf : configurations) {
    T_cams_extrinsics.push_back(slTransformToEigen(conf.pose));
    sl::CameraIdentifier uuid(conf.serial_number);
    fusion.updatePose(uuid, conf.pose);
    if (fusion.subscribe(uuid, conf.communication_parameters, conf.pose,
                         conf.override_gravity) ==
        sl::FUSION_ERROR_CODE::SUCCESS)
      cameras.push_back(uuid);
  }

  if (cameras.empty()) {
    std::cerr << "[SHM] No cameras connected!\n";
    return EXIT_FAILURE;
  }
  std::cout << "[SHM] " << cameras.size() << " ZED cameras connected.\n";

  if (publish_body) {
    sl::BodyTrackingFusionParameters bfp;
    bfp.enable_tracking = true;
    bfp.enable_body_fitting = true;
    fusion.enableBodyTracking(bfp);
  }

  sl::BodyTrackingFusionRuntimeParameters bt_rt;
  bt_rt.skeleton_minimum_allowed_keypoints = 7;
  bt_rt.skeleton_minimum_allowed_camera = cameras.size() / 2.0;

  // ── YOLO ─────────────────────────────────────────────────────────────────
  Yolov8Seg yolov8Seg;
  cv::dnn::Net yolo_net;
  if (!yolo_model_path.empty())
    yolo_net = LoadYOLOModel(yolov8Seg, yolo_model_path);

  // ── Shared memory init ────────────────────────────────────────────────────
  void *shm_pc = shm_create(SHM_PC_NAME, shm_pc_total_size());
  void *shm_smpl = shm_create(SHM_SMPL_NAME, shm_smpl_total_size());

  // Initialise headers
  {
    auto *h = shm_pc_header(shm_pc);
    h->num_cams = static_cast<uint32_t>(clients.size());
    h->max_points = SHM_MAX_POINTS;
    h->point_floats = SHM_POINT_FLOATS;
    h->num_slots = SHM_PC_SLOTS;
    for (int i = 0; i < SHM_MAX_CAMS; ++i)
      h->write_seq[i].store(0);
  }
  {
    auto *h = shm_smpl_header(shm_smpl);
    h->num_slots = SHM_SMPL_SLOTS;
    h->write_seq.store(0);
  }

  std::cout << "[SHM] PC   segment: " << SHM_PC_NAME << "  ("
            << shm_pc_total_size() / (1024 * 1024) << " MB)\n"
            << "[SHM] SMPL segment: " << SHM_SMPL_NAME << "  ("
            << shm_smpl_total_size() << " B)\n";

  // ── Per-frame state ───────────────────────────────────────────────────────
  std::vector<sl::Bodies> detected_bodies(cameras.size());
  std::vector<sl::BodyData> raw_bodies_vector;
  constexpr bool include_normals = true;
  uint32_t frame_id = 0;

  // ── Main loop ─────────────────────────────────────────────────────────────
  while (g_running) {
    trigger.notifyZED();
    std::cout << "──── Frame " << frame_id << " ────\n";

    // ── Point clouds ──────────────────────────────────────────────────────
    auto identity = Eigen::Matrix4d::Identity();
    for (int i = 0; i < static_cast<int>(clients.size()); ++i) {
      auto pcn = clients[i].getFilteredPointCloud(
          identity, yolo_net, yolov8Seg, include_normals, erode_kernel);
      pcn = voxelDownsample(pcn, voxel_size);
      write_pointcloud(shm_pc, i, pcn, frame_id);
    }

    // ── SMPL body ─────────────────────────────────────────────────────────
    if (publish_body) {
      for (size_t i = 0; i < cameras.size(); ++i) {
        clients[i].zed.retrieveBodies(detected_bodies[i]);
        if (!detected_bodies[i].body_list.empty())
          raw_bodies_vector.push_back(detected_bodies[i].body_list[0]);
      }

      bool body_valid = !raw_bodies_vector.empty();
      Body fused_body;
      if (body_valid) {
        std::vector<Body> bodies =
            extractBodyData(raw_bodies_vector, SMPL_TO_ZED);
        fused_body = mergeBodiesWithExtrinsics(bodies, T_cams_extrinsics);
      }

      // betas = zeros (no smpl_params_file in this config)
      static const std::vector<double> zero_betas(300, 0.0);
      write_smpl(shm_smpl, fused_body, T_SMPL_TO_ROS, zero_betas, frame_id,
                 body_valid);

      raw_bodies_vector.clear();
    }

    // ── Visualisation ─────────────────────────────────────────────────────
    if (visualize_image) {
      for (size_t i = 0; i < clients.size(); ++i) {
        sl::Mat zed_image;
        if (clients[i].zed.retrieveImage(zed_image, sl::VIEW::LEFT) ==
            sl::ERROR_CODE::SUCCESS) {
          cv::Mat cv_img(zed_image.getHeight(), zed_image.getWidth(), CV_8UC4,
                         zed_image.getPtr<sl::uchar1>(sl::MEM::CPU));
          cv::cvtColor(cv_img, cv_img, cv::COLOR_BGRA2BGR);
          cv::Mat displayed = cv_img;
          if (overlay_yolo_mask && !yolo_model_path.empty())
            displayed = clients[i].overlayBestPersonMask(cv_img, yolo_net, yolov8Seg);
          cv::imshow("Camera " + std::to_string(i), displayed);
        }
      }
      cv::waitKey(1);
    }

    ++frame_id;
  }

  // ── Shutdown ──────────────────────────────────────────────────────────────
  std::cout << "[SHM] Shutting down...\n";
  trigger.running = false;
  trigger.notifyZED();
  for (auto &c : clients)
    c.stop();
  fusion.close();

  munmap(shm_pc, shm_pc_total_size());
  munmap(shm_smpl, shm_smpl_total_size());
  shm_unlink(SHM_PC_NAME);
  shm_unlink(SHM_SMPL_NAME);

  return 0;
}