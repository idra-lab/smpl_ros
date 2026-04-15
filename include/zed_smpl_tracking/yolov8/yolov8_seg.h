#pragma once
// YOLOE via YOLOs-CPP — replaces the old YOLOv8 OpenCV-DNN implementation.
// The ONNX (yoloe-26n-seg.onnx) was exported with set_classes(["person"]),
// so class 0 == "person" is baked into the model weights.

#include "yolos/tasks/yoloe.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using YoloeSegDetector = yolos::yoloe::YOLOESegDetector;

/// Load a YOLOE segmentation model.  class_names must match the vocabulary
/// that was passed to model.set_classes() at ONNX export time.
inline std::unique_ptr<YoloeSegDetector>
LoadYOLOModel(const std::string &model_path,
              const std::vector<std::string> &class_names = {"person"},
              bool use_gpu = true) {
  auto det = std::make_unique<YoloeSegDetector>(model_path, class_names,
                                                use_gpu, /*agnosticNms=*/true);
  std::cout << "YOLOE model loaded: " << model_path << std::endl;
  return det;
}
