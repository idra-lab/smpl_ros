#include "zed_smpl_tracking/ClientPublisher.hpp"

ClientPublisher::ClientPublisher() {}

ClientPublisher::~ClientPublisher() { zed.close(); }

bool ClientPublisher::open(sl::InputType input,
                           sl::COORDINATE_SYSTEM coord_system,
                           sl::RESOLUTION resolution, Trigger *ref,
                           int sdk_gpu_id) {
  p_trigger = ref;

  sl::InitParameters init_parameters;
  init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL_PLUS;
  init_parameters.input = input;
  init_parameters.coordinate_units = sl::UNIT::METER;
  init_parameters.depth_stabilization = 1;
  init_parameters.sdk_gpu_id = sdk_gpu_id;
  // set max_depth to 10m
  init_parameters.depth_maximum_distance = 4.0;
  // set ROS coordinate system
  init_parameters.coordinate_system = coord_system;
  init_parameters.camera_resolution = resolution;
  //   init_parameters.reference_frame = sl::REFERENCE_FRAME::WORLD;
  auto state = zed.open(init_parameters);
  if (state != sl::ERROR_CODE::SUCCESS) {
    std::cout << "Error: " << state << std::endl;
    return false;
  }
  serial = zed.getCameraInformation().serial_number;
  p_trigger->states[serial] = false;
  std::cout << "ZED camera serial number: " << serial << std::endl;

  // print calibration matrix
  auto cam_params =
      zed.getCameraInformation().camera_configuration.calibration_parameters;
  std::cout << "Camera SN " << serial
            << ": Intrinsics (fx, fy, cx, cy) = " << cam_params.left_cam.fx
            << " x " << cam_params.left_cam.fy << " x "
            << cam_params.left_cam.cx << " x " << cam_params.left_cam.cy
            << std::endl;
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
  // needed to retrieve local orientations
  body_tracking_parameters.enable_tracking = true;
  state = zed.enableBodyTracking(body_tracking_parameters);
  if (state != sl::ERROR_CODE::SUCCESS) {
    std::cout << "Error: " << state << std::endl;
    return false;
  }
  std::cout << "ZED camera initialized successfully." << std::endl;
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

// Use yolo to predict human mask from RGB image
bool ClientPublisher::getYoloPredictionMask(cv::dnn::Net &net,
                                        Yolov8Seg &yolov8Seg,
                                        cv::Mat &out_mask,
                                        cv::Rect &out_bbox,
                                        int erode_kernel_size = 5) {
  sl::Mat sl_image;
  if (zed.retrieveImage(sl_image, sl::VIEW::LEFT) != sl::ERROR_CODE::SUCCESS)
    return false;

  cv::Mat cvImage(sl_image.getHeight(), sl_image.getWidth(), CV_8UC4,
                  sl_image.getPtr<sl::uchar1>(sl::MEM::CPU));
  cv::cvtColor(cvImage, cvImage, cv::COLOR_BGRA2BGR);

  std::vector<OutputParams> detections;
  if (!yolov8Seg.Detect(cvImage, net, detections))
    return false;

  for (auto &det : detections) {
    if (det.id == 0) {
      out_bbox = det.box;
      out_mask = det.boxMask.clone();
      break;
    }
  }

  if (out_mask.empty())
    return false;

  cv::erode(out_mask, out_mask,
            cv::getStructuringElement(
                cv::MORPH_RECT, cv::Size(erode_kernel_size, erode_kernel_size)));
  return true;
}


std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>>
ClientPublisher::getFilteredPointCloud(const Eigen::Matrix4d &T,
                                       const cv::Mat &human_mask,
                                       const cv::Rect &human_bbox,
                                       bool include_normals = false) {
  std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>>
      points_colors_normals;

  if (human_mask.empty())
    return points_colors_normals;

  sl::Mat pc_mat;
  if (zed.retrieveMeasure(pc_mat, sl::MEASURE::XYZRGBA) !=
      sl::ERROR_CODE::SUCCESS)
    return points_colors_normals;

  float *normal_ptr = nullptr;
  sl::Mat normal_mat;
  if (include_normals) {
    if (zed.retrieveMeasure(normal_mat, sl::MEASURE::NORMALS) !=
        sl::ERROR_CODE::SUCCESS)
      include_normals = false;
    else
      normal_ptr = normal_mat.getPtr<float>(sl::MEM::CPU);
  }

  int width = pc_mat.getWidth();
  int height = pc_mat.getHeight();
  float *pc_ptr = pc_mat.getPtr<float>(sl::MEM::CPU);

  for (int y = human_bbox.y; y < human_bbox.y + human_bbox.height; y++) {
    for (int x = human_bbox.x; x < human_bbox.x + human_bbox.width; x++) {
      if (y >= height || x >= width)
        continue;

      if (human_mask.at<uchar>(y - human_bbox.y, x - human_bbox.x) == 0)
        continue;

      int idx = (y * width + x) * 4;
      float X = pc_ptr[idx + 0];
      float Y = pc_ptr[idx + 1];
      float Z = pc_ptr[idx + 2];
      float rgba_f = pc_ptr[idx + 3];

      if (!std::isfinite(X) || !std::isfinite(Y) || !std::isfinite(Z))
        continue;

      Eigen::Vector4d pt(X, Y, Z, 1.0);
      Eigen::Vector3d pt_transformed = (T * pt).head<3>();

      uint32_t rgba = *reinterpret_cast<uint32_t *>(&rgba_f);
      uint8_t r = (rgba >> 0) & 0xFF;
      uint8_t g = (rgba >> 8) & 0xFF;
      uint8_t b = (rgba >> 16) & 0xFF;
      Eigen::Vector3d color(r / 255.0, g / 255.0, b / 255.0);

      Eigen::Vector3d n_transformed(0.0, 0.0, 0.0);
      if (include_normals && normal_ptr != nullptr) {
        float nx = normal_ptr[idx + 0];
        float ny = normal_ptr[idx + 1];
        float nz = normal_ptr[idx + 2];
        if (std::isfinite(nx) && std::isfinite(ny) && std::isfinite(nz)) {
          Eigen::Vector4d n(nx, ny, nz, 0.0);
          n_transformed = (T * n).head<3>().normalized();
        }
      }

      points_colors_normals.emplace_back(pt_transformed, color, n_transformed);
    }
  }

  return points_colors_normals;
}

cv::Mat ClientPublisher::getFilteredDepthMap(const cv::Mat &human_mask,
                                             const cv::Rect &human_bbox)
{
  sl::Mat depth_mat;
  if (zed.retrieveMeasure(depth_mat, sl::MEASURE::DEPTH) !=
      sl::ERROR_CODE::SUCCESS)
    return cv::Mat();

  int width  = depth_mat.getWidth();
  int height = depth_mat.getHeight();

  cv::Mat depthMap(height, width, CV_32FC1,
                   depth_mat.getPtr<float>(sl::MEM::CPU));
  cv::Mat filteredDepth = cv::Mat::zeros(height, width, CV_32FC1);

  cv::Rect image_bounds(0, 0, width, height);
  cv::Rect clipped_bbox = human_bbox & image_bounds;
  if (clipped_bbox.width <= 0 || clipped_bbox.height <= 0)
    return filteredDepth;

  int mask_offset_x = clipped_bbox.x - human_bbox.x;
  int mask_offset_y = clipped_bbox.y - human_bbox.y;

  for (int y = 0; y < clipped_bbox.height; y++) {
    int img_y  = clipped_bbox.y + y;
    int mask_y = mask_offset_y + y;

    const float *depth_ptr = depthMap.ptr<float>(img_y);
    float *out_ptr         = filteredDepth.ptr<float>(img_y);

    for (int x = 0; x < clipped_bbox.width; x++) {
      int img_x  = clipped_bbox.x + x;
      int mask_x = mask_offset_x + x;

      if (human_mask.at<uchar>(mask_y, mask_x) == 0)
        continue;

      float d = depth_ptr[img_x];
      if (std::isfinite(d) && d > 0.f)
        out_ptr[img_x] = d;
    }
  }

  return filteredDepth;
}

cv::Mat ClientPublisher::overlayPersonMask(const cv::Mat &image,
                                               const cv::Mat &mask,
                                               const cv::Rect &bbox) {
  cv::Mat output = image.clone();
  if (output.empty() || mask.empty())
    return output;

  cv::Rect image_bounds(0, 0, output.cols, output.rows);
  cv::Rect clipped_bbox = bbox & image_bounds;
  if (clipped_bbox.width <= 0 || clipped_bbox.height <= 0)
    return output;

  int mask_x = clipped_bbox.x - bbox.x;
  int mask_y = clipped_bbox.y - bbox.y;
  cv::Rect mask_roi(mask_x, mask_y, clipped_bbox.width, clipped_bbox.height);
  if (mask_roi.x < 0 || mask_roi.y < 0 ||
      mask_roi.x + mask_roi.width > mask.cols ||
      mask_roi.y + mask_roi.height > mask.rows)
    return output;

  cv::Mat roi = output(clipped_bbox);
  cv::Mat roi_overlay = roi.clone();
  roi_overlay.setTo(cv::Scalar(0, 0, 255), mask(mask_roi));
  constexpr double alpha = 0.4;
  cv::addWeighted(roi_overlay, alpha, roi, 1.0 - alpha, 0.0, roi);
  return output;
}