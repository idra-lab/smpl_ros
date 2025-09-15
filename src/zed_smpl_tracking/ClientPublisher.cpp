#include "zed_smpl_tracking/ClientPublisher.hpp"

ClientPublisher::ClientPublisher() {}

ClientPublisher::~ClientPublisher() { zed.close(); }

bool ClientPublisher::open(sl::InputType input, Trigger *ref, int sdk_gpu_id) {

  p_trigger = ref;

  sl::InitParameters init_parameters;
  init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL_PLUS;
  init_parameters.input = input;
  init_parameters.coordinate_units = sl::UNIT::METER;
  init_parameters.depth_stabilization = 30;
  init_parameters.sdk_gpu_id = sdk_gpu_id;
  // set max_depth to 10m
  init_parameters.depth_maximum_distance = 4.0;
  // set ROS coordinate system
  init_parameters.coordinate_system =
      sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
  init_parameters.camera_resolution = sl::RESOLUTION::HD720;
//   init_parameters.reference_frame = sl::REFERENCE_FRAME::WORLD;
  auto state = zed.open(init_parameters);
  if (state != sl::ERROR_CODE::SUCCESS) {
    std::cout << "Error: " << state << std::endl;
    return false;
  }

  serial = zed.getCameraInformation().serial_number;
  p_trigger->states[serial] = false;

  // in most cases in body tracking setup, the cameras are static
  sl::PositionalTrackingParameters positional_tracking_parameters;
  // in most cases for body detection application the camera is static:
  positional_tracking_parameters.set_as_static = true;

  state = zed.enablePositionalTracking(positional_tracking_parameters);
  if (state != sl::ERROR_CODE::SUCCESS) {
    std::cout << "Error: " << state << std::endl;
    return false;
  }

  // define the body tracking parameters, as the fusion can does the tracking
  // and fitting you don't need to enable them here, unless you need it for your
  // app
  sl::BodyTrackingParameters body_tracking_parameters;
  body_tracking_parameters.detection_model =
      sl::BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;
  body_tracking_parameters.body_format = sl::BODY_FORMAT::BODY_38;
  body_tracking_parameters.enable_body_fitting = true;
  body_tracking_parameters.enable_tracking = true;
  state = zed.enableBodyTracking(body_tracking_parameters);
  if (state != sl::ERROR_CODE::SUCCESS) {
    std::cout << "Error: " << state << std::endl;
    return false;
  }

  return true;
}

void ClientPublisher::start() {
  if (zed.isOpened()) {
    // the camera should stream its data so the fusion can subscibe to it to
    // gather the detected body and others metadata needed for the process.
    zed.startPublishing();
    // the thread can start to process the camera grab in background
    runner = std::thread(&ClientPublisher::work, this);
  }
}

void ClientPublisher::stop() {
  if (runner.joinable())
    runner.join();
  zed.close();
}

void ClientPublisher::work() {
  sl::Bodies bodies;
  sl::BodyTrackingRuntimeParameters body_runtime_parameters;
  body_runtime_parameters.detection_confidence_threshold = 40;
  zed.setBodyTrackingRuntimeParameters(body_runtime_parameters);

  sl::RuntimeParameters rt;
  rt.confidence_threshold = 50;

  // In this sample we use a dummy thread to process the ZED data.
  // you can replace it by your own application and use the ZED like you use to,
  // retrieve its images, depth, sensors data and so on. As long as you call the
  // grab method, since the camera is subscribed to fusion it will run the
  // detection and the camera will be able to seamlessly transmit the data to
  // the fusion module.
  while (p_trigger->running) {
    std::unique_lock<std::mutex> lk(mtx);
    p_trigger->cv.wait(lk);
    if (p_trigger->running) {
      if (zed.grab(rt) == sl::ERROR_CODE::SUCCESS) {
      }
    }
    p_trigger->states[serial] = true;
  }
}

void ClientPublisher::setStartSVOPosition(unsigned pos) {
  zed.setSVOPosition(pos);
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
ClientPublisher::getFilteredPointCloud(const Eigen::Matrix4d &T,
                                       cv::dnn::Net &net,
                                       Yolov8Seg &yolov8Seg) {
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> points_colors;

  // 1️⃣ Grab RGB image
  sl::Mat sl_image;
  if (zed.retrieveImage(sl_image, sl::VIEW::LEFT) != sl::ERROR_CODE::SUCCESS)
  {
    return points_colors;
  }

  cv::Mat cvImage(sl_image.getHeight(), sl_image.getWidth(), CV_8UC4,
                  sl_image.getPtr<sl::uchar1>(sl::MEM::CPU));
  cv::cvtColor(cvImage, cvImage, cv::COLOR_BGRA2BGR);

  // 2️⃣ YOLO detection
  std::vector<OutputParams> detections;
  if (!yolov8Seg.Detect(cvImage, net, detections))
    return points_colors;

  // 3️⃣ Pick first human detection
  cv::Rect human_bbox;
  cv::Mat human_mask;
  for (auto &det : detections) {
    if (det.id == 0) { // person class
      human_bbox = det.box;
      human_mask = det.boxMask.clone();
      break;
    }
  }

  if (human_mask.empty())
    return points_colors;

  // 4️⃣ Erode mask to clean edges
  cv::erode(human_mask, human_mask,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
  sl::Mat pc_mat_;
  // 5️⃣ Grab ZED point cloud
  if (zed.retrieveMeasure(pc_mat_, sl::MEASURE::XYZRGBA) !=
      sl::ERROR_CODE::SUCCESS)
    return points_colors;

  // 6️⃣ Convert ZED point cloud + mask → points_colors
  int width = pc_mat_.getWidth();
  int height = pc_mat_.getHeight();
  float *ptr = pc_mat_.getPtr<float>(sl::MEM::CPU);

  for (int y = human_bbox.y; y < human_bbox.y + human_bbox.height; y++) {
    for (int x = human_bbox.x; x < human_bbox.x + human_bbox.width; x++) {
      if (y >= height || x >= width)
        continue;

      // Mask is relative to bbox
      if (human_mask.at<uchar>(y - human_bbox.y, x - human_bbox.x) == 0){
        continue;
      }

      int idx = (y * width + x) * 4;
      float X = ptr[idx + 0];
      float Y = ptr[idx + 1];
      float Z = ptr[idx + 2];
      float rgba_f = ptr[idx + 3];

      if (!std::isfinite(X) || !std::isfinite(Y) || !std::isfinite(Z))
      {
        continue;
      }

      Eigen::Vector4d pt(X, Y, Z, 1.0);
      Eigen::Vector3d pt_transformed = (T * pt).head<3>();

      // Extract RGB
      uint32_t rgba = *reinterpret_cast<uint32_t *>(&rgba_f);
      uint8_t r = (rgba >> 0) & 0xFF;
      uint8_t g = (rgba >> 8) & 0xFF;
      uint8_t b = (rgba >> 16) & 0xFF;
      Eigen::Vector3d color(r / 255.0, g / 255.0, b / 255.0);

      points_colors.emplace_back(pt_transformed, color);
    }
  }

  return points_colors;
}
