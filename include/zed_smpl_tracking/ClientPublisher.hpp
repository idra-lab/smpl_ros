#ifndef __SENDER_RUNNER_HDR__
#define __SENDER_RUNNER_HDR__

#include "bodyStruct.hpp"
#include "yolov8_seg.h"
#include <Eigen/Dense>
#include <condition_variable>
#include <sl/Camera.hpp>
#include <sl/Fusion.hpp>
#include <thread>

struct Trigger {
  void notifyZED() {
    cv.notify_all();

    if (running) {
      bool wait_for_zed = true;
      const int nb_zed = states.size();
      while (wait_for_zed) {
        int count_r = 0;
        for (auto &it : states)
          count_r += it.second;
        wait_for_zed = count_r != nb_zed;
        sl::sleep_ms(1);
      }
      for (auto &it : states)
        it.second = false;
    }
  }

  std::condition_variable cv;
  bool running = true;
  std::map<int, bool> states;
};

class ClientPublisher {
public:
  int serial;
  sl::Camera zed;
  ClientPublisher();
  ~ClientPublisher();

  bool open(sl::InputType, sl::COORDINATE_SYSTEM coord_system,
            sl::RESOLUTION resolution, Trigger *ref, int sdk_gpu_id);
  void start();
  void stop();
  void setStartSVOPosition(unsigned pos);

  bool getYoloPredictionMask(YoloeSegDetector &detector,
                             cv::Mat &out_mask, cv::Rect &out_bbox,
                             int erode_kernel_size);
  std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>>
  getFilteredPointCloud(const Eigen::Matrix4d &T, const cv::Mat &human_mask,
                        const cv::Rect &human_bbox, bool include_normals);

  cv::Mat getFilteredDepthMap(const cv::Mat &human_mask,
                              const cv::Rect &human_bbox);
  std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d>>
  extractPointCloudFast(bool include_normals);
  cv::Mat overlayPersonMask(const cv::Mat &image, const cv::Mat &mask,
                            const cv::Rect &bbox);

private:
  void work();
  std::thread runner;
  std::mutex mtx;
  Trigger *p_trigger;
};

#endif // ! __SENDER_RUNNER_HDR__