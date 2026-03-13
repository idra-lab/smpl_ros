/**
 * smpl_shm_publisher — ZED SMPL tracking con shared memory invece di ROS2.
 *
 * Pubblica su /dev/shm:
 *   /zed_pointclouds  → point cloud per camera
 *   /zed_smpl         → parametri SMPL
 *   /zed_caminfo      → intrinseci + estrinseci per camera (scritto una volta)
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

#include "utils/constants.hpp"
#include "utils/json.hpp"
#include "utils/shm_transport.h"
#include "utils/voxel_filter.h"
#include "yolov8_seg.h"
#include "zed_smpl_tracking/ClientPublisher.hpp"
#include "zed_smpl_tracking/bodyConverter.hpp"
#include "zed_smpl_tracking/fuseSkeletons.hpp"

// ─── minimal ROS2-params YAML parser ─────────────────────────────────────────
//
// Parses files shaped like:
//   node_name:
//     ros__parameters:
//       key: value
//
// Usage:  ./smpl_shm_publisher params.yaml
//         (falls back to env vars / defaults if no file given)

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

struct Params {
  std::unordered_map<std::string, std::string> data;

  // ── loader ────────────────────────────────────────────────────────────────
  static Params from_yaml(const std::string &path) {
    Params p;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open params file: " + path);

    bool in_ros_params = false;
    std::string line;
    while (std::getline(f, line)) {
      // strip inline comments
      auto cpos = line.find('#');
      if (cpos != std::string::npos) line = line.substr(0, cpos);

      // detect "ros__parameters:" block (any indentation)
      if (line.find("ros__parameters:") != std::string::npos) {
        in_ros_params = true;
        continue;
      }
      if (!in_ros_params) continue;

      // stop at next top-level key (no leading spaces = new node section)
      if (!line.empty() && line[0] != ' ' && line[0] != '\t') {
        in_ros_params = false;
        continue;
      }

      auto colon = line.find(':');
      if (colon == std::string::npos) continue;

      std::string key   = line.substr(0, colon);
      std::string value = line.substr(colon + 1);

      // trim whitespace
      auto trim = [](std::string &s) {
        size_t b = s.find_first_not_of(" \t");
        size_t e = s.find_last_not_of(" \t\r\n");
        s = (b == std::string::npos) ? "" : s.substr(b, e - b + 1);
      };
      trim(key); trim(value);

      if (key.empty() || value.empty()) continue;

      // strip surrounding quotes
      if (value.size() >= 2 &&
          ((value.front() == '"' && value.back() == '"') ||
           (value.front() == '\'' && value.back() == '\'')))
        value = value.substr(1, value.size() - 2);

      p.data[key] = value;
    }
    return p;
  }

  // ── accessors with env-var override then default ──────────────────────────
  std::string get_str(const std::string &key, const std::string &env_var,
                      const std::string &def) const {
    if (const char *v = std::getenv(env_var.c_str())) return v;
    auto it = data.find(key);
    return (it != data.end()) ? it->second : def;
  }
  double get_double(const std::string &key, const std::string &env_var,
                    double def) const {
    if (const char *v = std::getenv(env_var.c_str())) return std::stod(v);
    auto it = data.find(key);
    return (it != data.end()) ? std::stod(it->second) : def;
  }
  int get_int(const std::string &key, const std::string &env_var,
              int def) const {
    if (const char *v = std::getenv(env_var.c_str())) return std::stoi(v);
    auto it = data.find(key);
    return (it != data.end()) ? std::stoi(it->second) : def;
  }
  bool get_bool(const std::string &key, const std::string &env_var,
                bool def) const {
    if (const char *v = std::getenv(env_var.c_str())) {
      std::string s(v);
      return s == "1" || s == "true";
    }
    auto it = data.find(key);
    if (it == data.end()) return def;
    return it->second == "true" || it->second == "1";
  }
};

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

static void *shm_create(const char *name, size_t size) {
  shm_unlink(name);
  int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
  if (fd < 0) { perror("shm_open"); exit(1); }
  if (ftruncate(fd, static_cast<off_t>(size)) < 0) { perror("ftruncate"); exit(1); }
  void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) { perror("mmap"); exit(1); }
  close(fd);
  memset(ptr, 0, size);
  return ptr;
}

Eigen::Matrix4d slTransformToEigen(const sl::Transform &T) {
  Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
  sl::Matrix3f r = T.getRotationMatrix();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      mat(i, j) = static_cast<double>(r(i, j));
  sl::Translation t = T.getTranslation();
  mat(0, 3) = static_cast<double>(t.x);
  mat(1, 3) = static_cast<double>(t.y);
  mat(2, 3) = static_cast<double>(t.z);
  return mat;
}

// ─── SHM writers ─────────────────────────────────────────────────────────────

static void write_pointcloud(
    void *shm_pc, int cam_idx,
    const std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d,
                                 Eigen::Vector3d>> &cloud,
    uint32_t frame_id) {
  auto *hdr = shm_pc_header(shm_pc);
  uint64_t seq = hdr->write_seq[cam_idx].load(std::memory_order_relaxed);
  int slot = static_cast<int>(seq % SHM_PC_SLOTS);

  SHMPCSlot *s    = shm_pc_slot(shm_pc, cam_idx, slot);
  SHMFloat  *data = shm_pc_data(shm_pc, cam_idx, slot);

  s->state.store(1, std::memory_order_release);

  uint32_t n = static_cast<uint32_t>(
      std::min(cloud.size(), static_cast<size_t>(SHM_MAX_POINTS)));
  s->num_points   = n;
  s->timestamp_ns = now_ns();
  s->cam_idx      = static_cast<uint32_t>(cam_idx);
  s->frame_id     = frame_id;

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

  s->state.store(2, std::memory_order_release);
  hdr->write_seq[cam_idx].fetch_add(1, std::memory_order_release);
}

static void write_smpl(void *shm_smpl, const Body &fused_body,
                       const Eigen::Matrix4d &T_SMPL_TO_ROS,
                       const std::vector<double> &betas, uint32_t frame_id,
                       bool valid) {
  auto *hdr = shm_smpl_header(shm_smpl);
  uint64_t seq = hdr->write_seq.load(std::memory_order_relaxed);
  int slot = static_cast<int>(seq % SHM_SMPL_SLOTS);

  SHMSMPLSlot *s = shm_smpl_slot(shm_smpl, slot);
  s->state.store(1, std::memory_order_release);

  s->valid        = valid ? 1u : 0u;
  s->timestamp_ns = now_ns();
  s->frame_id     = frame_id;

  if (valid) {
    for (int i = 0; i < SHM_SMPL_SHAPE_PARAMS; ++i)
      s->betas[i] = (i < static_cast<int>(betas.size()))
                        ? static_cast<float>(betas[i]) : 0.f;

    Eigen::Vector4d t_body(fused_body.root_position.x(),
                           fused_body.root_position.y(),
                           fused_body.root_position.z(), 1.0);
    Eigen::Vector4d t_ros = T_SMPL_TO_ROS * t_body;
    s->trans[0] = static_cast<float>(t_ros.x());
    s->trans[1] = static_cast<float>(t_ros.y());
    s->trans[2] = static_cast<float>(t_ros.z());

    Eigen::Quaterniond q_ros =
        Eigen::Quaterniond(T_SMPL_TO_ROS.block<3, 3>(0, 0)) *
        fused_body.global_orientation;
    q_ros.normalize();
    Eigen::AngleAxisd aa(q_ros);
    Eigen::Vector3d axis_angle = aa.axis() * aa.angle();
    s->global_orient[0] = static_cast<float>(axis_angle.x());
    s->global_orient[1] = static_cast<float>(axis_angle.y());
    s->global_orient[2] = static_cast<float>(axis_angle.z());

    for (int j = 0;
         j < 24 && j < static_cast<int>(fused_body.local_orient.size()); ++j) {
      Eigen::AngleAxisd aa_j(fused_body.local_orient[j].normalized());
      Eigen::Vector3d v = aa_j.axis() * aa_j.angle();
      s->pose[j * 3 + 0] = static_cast<float>(v.x());
      s->pose[j * 3 + 1] = static_cast<float>(v.y());
      s->pose[j * 3 + 2] = static_cast<float>(v.z());
    }
    for (int j = static_cast<int>(fused_body.local_orient.size()); j < 24; ++j) {
      s->pose[j * 3 + 0] = 0.f;
      s->pose[j * 3 + 1] = 0.f;
      s->pose[j * 3 + 2] = 0.f;
    }
  }

  s->state.store(2, std::memory_order_release);
  hdr->write_seq.fetch_add(1, std::memory_order_release);
}

/**
 * Scritto una sola volta dopo l'apertura delle camere.
 * Intrinseci dal calibration parameters ZED.
 * Estrinseci = T_cam_to_world letti dal fusion config (conf.pose).
 */
static void write_caminfo(
    void *shm_caminfo,
    const std::vector<ClientPublisher> &clients,
    const std::vector<sl::FusionConfiguration> &configurations,
    const std::vector<Eigen::Matrix4d> &T_cams_extrinsics,
    int width, int height)
{
  auto *hdr = shm_caminfo_header(shm_caminfo);
  hdr->ready.store(0, std::memory_order_release);  // mark not-ready while writing
  hdr->num_cams = static_cast<uint32_t>(clients.size());

  for (int i = 0; i < static_cast<int>(clients.size()); ++i) {
    SHMCamInfoEntry *e = shm_caminfo_entry(shm_caminfo, i);

    // ── intrinsics ──────────────────────────────────────────────────────────
    auto cal = clients[i].zed
                   .getCameraInformation()
                   .camera_configuration
                   .calibration_parameters;
    const auto &left = cal.left_cam;

    e->fx            = static_cast<float>(left.fx);
    e->fy            = static_cast<float>(left.fy);
    e->cx            = static_cast<float>(left.cx);
    e->cy            = static_cast<float>(left.cy);
    e->width         = static_cast<uint32_t>(width);
    e->height        = static_cast<uint32_t>(height);
    e->serial_number = static_cast<uint32_t>(configurations[i].serial_number);

    // distortion: disto[0..4] = k1 k2 p1 p2 k3
    for (int d = 0; d < 5; ++d)
      e->dist[d] = static_cast<float>(left.disto[d]);

    // ── extrinsics: T_cam_to_world row-major ────────────────────────────────
    // T_cams_extrinsics[i] is already in ROS world frame (RIGHT_HANDED_Z_UP_X_FWD)
    const Eigen::Matrix4d &T = T_cams_extrinsics[i];
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        e->T_cam_to_world[r * 4 + c] = T(r, c);

    std::cout << "[SHM] cam " << i
              << " SN=" << e->serial_number
              << " fx=" << e->fx << " fy=" << e->fy
              << " cx=" << e->cx << " cy=" << e->cy << "\n"
              << "      T_cam_to_world(0,3)=" << e->T_cam_to_world[3]
              << " (1,3)=" << e->T_cam_to_world[7]
              << " (2,3)=" << e->T_cam_to_world[11] << "\n";
  }

  // All entries written — mark ready
  hdr->ready.store(1, std::memory_order_release);
  std::cout << "[SHM] caminfo written for " << clients.size() << " camera(s)\n";
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  signal(SIGINT,  sigint_handler);
  signal(SIGTERM, sigint_handler);

  // ── Load params: first arg = path to ROS2 params YAML (optional) ──────────
  //    Priority: env var  >  yaml file  >  hardcoded default
  Params cfg;
  if (argc >= 2) {
    try {
      cfg = Params::from_yaml(argv[1]);
      std::cout << "[SHM] Loaded params from: " << argv[1] << "\n";
    } catch (const std::exception &ex) {
      std::cerr << "[SHM] Warning: " << ex.what() << " — using defaults\n";
    }
  }

  // YAML key                        env var override        default
  std::string calib_file      = cfg.get_str   ("calibration_file",            "ZED_CALIB_FILE",   "two_cams_fusion_config.json");
  std::string yolo_model_path = cfg.get_str   ("yolo_model_path",             "ZED_YOLO_PATH",    "yolov8s-seg.onnx");
  std::string resolution_str  = cfg.get_str   ("resolution",                  "ZED_RESOLUTION",   "1280x720");
  double      voxel_size      = cfg.get_double("published_body_filter_voxel_size", "ZED_VOXEL_SIZE",   0.015);
  int         erode_kernel    = cfg.get_int   ("erode_body_mask_kernel_size",  "ZED_ERODE_KERNEL", 18);
  bool overlay_yolo_mask      = cfg.get_bool  ("overlay_yolo_mask",           "ZED_OVERLAY_YOLO", true);
  bool visualize_image        = cfg.get_bool  ("visualize_image",             "ZED_VISUALIZE",    true);
  bool publish_body           = cfg.get_bool  ("publish_body",                "ZED_PUBLISH_BODY", true);

  std::cout << "[SHM] calib:      " << calib_file      << "\n"
            << "[SHM] yolo:       " << yolo_model_path << "\n"
            << "[SHM] resolution: " << resolution_str  << "\n"
            << "[SHM] voxel_size: " << voxel_size      << "\n"
            << "[SHM] erode:      " << erode_kernel     << "\n";

  int width  = std::stoi(resolution_str.substr(0, resolution_str.find('x')));
  int height = std::stoi(resolution_str.substr(resolution_str.find('x') + 1));
  sl::RESOLUTION resolution;
  switch (height) {
    case 1242: resolution = sl::RESOLUTION::HD2K;   break;
    case 1536: resolution = sl::RESOLUTION::HD1536; break;
    case 1080: resolution = sl::RESOLUTION::HD1080; break;
    case 1200: resolution = sl::RESOLUTION::HD1200; break;
    case 720:  resolution = sl::RESOLUTION::HD720;  break;
    case 376:  resolution = sl::RESOLUTION::VGA;    break;
    default:
      std::cerr << "[SHM] Unsupported resolution height: " << height << "\n";
      return EXIT_FAILURE;
  }

  constexpr sl::COORDINATE_SYSTEM ROS_COORDINATE_SYSTEM =
      sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  constexpr sl::UNIT UNIT = sl::UNIT::METER;

  Eigen::Matrix4d T_SMPL_TO_ROS = smpl_to_ros_transform();

  // ── ZED cameras ──────────────────────────────────────────────────────────
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
      if (!clients[id_].open(conf.input_type, ROS_COORDINATE_SYSTEM,
                             resolution, &trigger, gpu_id))
        continue;
      id_++;
    }
  }
  for (auto &c : clients) c.start();

  // ── Fusion ───────────────────────────────────────────────────────────────
  sl::InitFusionParameters init_params;
  init_params.coordinate_units          = UNIT;
  init_params.coordinate_system         = ROS_COORDINATE_SYSTEM;
  init_params.verbose                   = true;
  init_params.maximum_working_resolution =
      sl::Resolution(std::max(1280, width), std::max(720, height));

  sl::Fusion fusion;
  fusion.init(init_params);

  std::vector<Eigen::Matrix4d>      T_cams_extrinsics;
  std::vector<sl::CameraIdentifier> cameras;

  for (auto &conf : configurations) {
    T_cams_extrinsics.push_back(slTransformToEigen(conf.pose));
    sl::CameraIdentifier uuid(conf.serial_number);
    fusion.updatePose(uuid, conf.pose);
    if (fusion.subscribe(uuid, conf.communication_parameters,
                         conf.pose, conf.override_gravity) ==
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
    bfp.enable_tracking     = true;
    bfp.enable_body_fitting = true;
    fusion.enableBodyTracking(bfp);
  }

  sl::BodyTrackingFusionRuntimeParameters bt_rt;
  bt_rt.skeleton_minimum_allowed_keypoints = 7;
  bt_rt.skeleton_minimum_allowed_camera    = cameras.size() / 2.0;

  // ── YOLO ─────────────────────────────────────────────────────────────────
  Yolov8Seg yolov8Seg;
  cv::dnn::Net yolo_net;
  if (!yolo_model_path.empty())
    yolo_net = LoadYOLOModel(yolov8Seg, yolo_model_path);

  // ── Shared memory init ───────────────────────────────────────────────────
  void *shm_pc      = shm_create(SHM_PC_NAME,      shm_pc_total_size());
  void *shm_smpl    = shm_create(SHM_SMPL_NAME,    shm_smpl_total_size());
  void *shm_caminfo = shm_create(SHM_CAMINFO_NAME, shm_caminfo_total_size());

  {
    auto *h         = shm_pc_header(shm_pc);
    h->num_cams     = static_cast<uint32_t>(clients.size());
    h->max_points   = SHM_MAX_POINTS;
    h->point_floats = SHM_POINT_FLOATS;
    h->num_slots    = SHM_PC_SLOTS;
    for (int i = 0; i < SHM_MAX_CAMS; ++i)
      h->write_seq[i].store(0);
  }
  {
    auto *h      = shm_smpl_header(shm_smpl);
    h->num_slots = SHM_SMPL_SLOTS;
    h->write_seq.store(0);
  }

  // Write camera intrinsics + extrinsics once at startup
  // (configurations holds sl::CameraConfiguration which has serial_number)
  for (auto &conf : configurations)
  write_caminfo(shm_caminfo, clients, configurations, T_cams_extrinsics, width, height);

  std::cout << "[SHM] PC   segment: " << SHM_PC_NAME
            << "  (" << shm_pc_total_size() / (1024 * 1024) << " MB)\n"
            << "[SHM] SMPL segment: " << SHM_SMPL_NAME
            << "  (" << shm_smpl_total_size() << " B)\n"
            << "[SHM] CAM  segment: " << SHM_CAMINFO_NAME
            << "  (" << shm_caminfo_total_size() << " B)\n";

  // ── Per-frame state ──────────────────────────────────────────────────────
  std::vector<sl::Bodies>   detected_bodies(cameras.size());
  std::vector<sl::BodyData> raw_bodies_vector;
  cv::Mat human_mask;
  cv::Rect human_bbox;


  constexpr bool include_normals = true;
  uint32_t frame_id = 0;

  // ── Main loop ────────────────────────────────────────────────────────────
  while (g_running) {
    trigger.notifyZED();
    std::cout << "──── Frame " << frame_id << " ────\n";

    // Point clouds
    auto identity = Eigen::Matrix4d::Identity();
    for (int i = 0; i < static_cast<int>(clients.size()); ++i) {
      if (clients[i].getYoloPredictionMask(yolo_net, yolov8Seg, human_mask, human_bbox, erode_kernel)) {
        auto pcn = clients[i].getFilteredPointCloud(
            identity, human_mask, human_bbox, include_normals);
        pcn = voxelDownsample(pcn, voxel_size);
        write_pointcloud(shm_pc, i, pcn, frame_id);
      }
    }

    // SMPL body
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

      static const std::vector<double> zero_betas(10, 0.0);
      write_smpl(shm_smpl, fused_body, T_SMPL_TO_ROS, zero_betas,
                 frame_id, body_valid);

      raw_bodies_vector.clear();
    }

    // Visualisation
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
            displayed = clients[i].overlayPersonMask(cv_img, human_mask, human_bbox);
          cv::imshow("Camera " + std::to_string(i), displayed);
        }
      }
      cv::waitKey(1);
    }

    ++frame_id;
  }

  // ── Shutdown ─────────────────────────────────────────────────────────────
  std::cout << "[SHM] Shutting down...\n";
  trigger.running = false;
  trigger.notifyZED();
  for (auto &c : clients) c.stop();
  fusion.close();

  munmap(shm_pc,      shm_pc_total_size());
  munmap(shm_smpl,    shm_smpl_total_size());
  munmap(shm_caminfo, shm_caminfo_total_size());
  shm_unlink(SHM_PC_NAME);
  shm_unlink(SHM_SMPL_NAME);
  shm_unlink(SHM_CAMINFO_NAME);

  return 0;
}