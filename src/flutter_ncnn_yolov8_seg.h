#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

#if _WIN32
#define FFI_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FFI_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// A very short-lived native function.
//
// For very short-lived functions, it is fine to call them on the main isolate.
// They will block the Dart execution while running the native function, so
// only do this for native functions which are guaranteed to be short-lived.
FFI_PLUGIN_EXPORT intptr_t sum(intptr_t a, intptr_t b);

// A longer lived native function, which occupies the thread calling it.
//
// Do not call these kind of native functions in the main isolate. They will
// block Dart execution. This will cause dropped frames in Flutter applications.
// Instead, call these native functions on a separate isolate.
FFI_PLUGIN_EXPORT intptr_t sum_long_running(intptr_t a, intptr_t b);

FFI_PLUGIN_EXPORT typedef int yolov8_err_t;

#define YOLOvV8_OK 0
#define YOLOV8_ERROR -1

FFI_PLUGIN_EXPORT struct YoloV8 {
  const char *model_path;  // path to model file
  const char *param_path;  // path to param file

  const ncnn::Net *net;

  float nms_thresh;   // nms threshold
  float conf_thresh;  // threshold of bounding box prob
  float target_size;  // target image size after resize, might use 416 for small
                      // model
};

// ncnn::Mat::PixelType
FFI_PLUGIN_EXPORT enum PixelType {
  PIXEL_RGB = 1,
  PIXEL_BGR = 2,
  PIXEL_GRAY = 3,
  PIXEL_RGBA = 4,
  PIXEL_BGRA = 5,
  PIXEL_YUV = 6,
};

FFI_PLUGIN_EXPORT struct Object {
  cv::Rect_<float> rect;
  int label;
  float prob;
  cv::Mat mask;
  std::vector<float> mask_feat;
};

FFI_PLUGIN_EXPORT struct Result {
  int count;
  std::vector<Object> objects;
};

FFI_PLUGIN_EXPORT yolov8_err_t LoadModel(const char *model_path,
                                         const char *param_path,
                                         float nms_thresh, float conf_thresh,
                                         struct YoloV8 *yolov8);

FFI_PLUGIN_EXPORT yolov8_err_t Inference(const struct YoloV8 *yolov8,
                                         const uint8_t *pixels,
                                         enum PixelType pixel_type,
                                         struct Result *result);

FFI_PLUGIN_EXPORT yolov8_err_t ReleaseModel(const struct YoloV8 *yolov8);

#ifdef __cplusplus
}
#endif
