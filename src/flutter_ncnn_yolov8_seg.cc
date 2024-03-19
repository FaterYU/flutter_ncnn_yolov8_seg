#include "flutter_ncnn_yolov8_seg.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

// ncnn
#include "layer.h"
#include "net.h"

// A very short-lived native function.
//
// For very short-lived functions, it is fine to call them on the main isolate.
// They will block the Dart execution while running the native function, so
// only do this for native functions which are guaranteed to be short-lived.
FFI_PLUGIN_EXPORT intptr_t sum(intptr_t a, intptr_t b) { return a + b; }

// A longer-lived native function, which occupies the thread calling it.
//
// Do not call these kind of native functions in the main isolate. They will
// block Dart execution. This will cause dropped frames in Flutter applications.
// Instead, call these native functions on a separate isolate.
FFI_PLUGIN_EXPORT intptr_t sum_long_running(intptr_t a, intptr_t b) {
  // Simulate work.
#if _WIN32
  Sleep(5000);
#else
  usleep(5000 * 1000);
#endif
  return a + b;
}

namespace yolov8 {
// definition
struct GridAndStride {
  int grid0;
  int grid1;
  int stride;
};
static void generate_proposals(std::vector<GridAndStride> grid_strides,
                               const ncnn::Mat &pred, float prob_threshold,
                               std::vector<Object> &objects);
static void generate_grids_and_stride(const int target_w, const int target_h,
                                      std::vector<int> &strides,
                                      std::vector<GridAndStride> &grid_strides);
static void qsort_descent_inplace(std::vector<Object> &faceobjects);
static void nms_sorted_bboxes(const std::vector<Object> &faceobjects,
                              std::vector<int> &picked, float nms_threshold);
static void decode_mask(const ncnn::Mat &mask_feat, const int &img_w,
                        const int &img_h, const ncnn::Mat &mask_proto,
                        const ncnn::Mat &in_pad, const int &wpad,
                        const int &hpad, ncnn::Mat &mask_pred_result);

// implementation
static void generate_proposals(std::vector<GridAndStride> grid_strides,
                               const ncnn::Mat &pred, float prob_threshold,
                               std::vector<Object> &objects) {
  const int num_points = grid_strides.size();
  const int num_class = 14;
  const int reg_max_1 = 16;

  for (int i = 0; i < num_points; i++) {
    int grid0 = grid_strides[i].grid0;
    int grid1 = grid_strides[i].grid1;
    int stride = grid_strides[i].stride;

    const ncnn::Mat score = pred.channel(0).row(grid1) / 255.f;
    const ncnn::Mat bbox_pred = pred.channel(1).row(grid1);
    const ncnn::Mat mask_pred = pred.channel(2).row(grid1);
    const ncnn::Mat mask_feat = pred.channel(3).row(grid1);

    for (int c = 0; c < num_class; c++) {
      float prob = score[c];
      if (prob < prob_threshold) continue;

      ncnn::Mat bbox_xy = bbox_pred.row(c * 2);
      ncnn::Mat bbox_wh = bbox_pred.row(c * 2 + 1);

      float x = (grid0 + bbox_xy[0]) * stride;
      float y = (grid1 + bbox_xy[1]) * stride;
      float w = exp(bbox_wh[0]) * stride;
      float h = exp(bbox_wh[1]) * stride;

      Object obj;
      obj.rect.x = x - w * 0.5f;
      obj.rect.y = y - h * 0.5f;
      obj.rect.width = w;
      obj.rect.height = h;
      obj.label = c;
      obj.prob = prob;
      obj.mask = mask_pred;
      obj.mask_feat = mask_feat;

      objects.push_back(obj);
    }
  }
}

static void generate_grids_and_stride(
    const int target_w, const int target_h, std::vector<int> &strides,
    std::vector<GridAndStride> &grid_strides) {
  for (int s : strides) {
    GridAndStride gs;
    gs.stride = s;
    gs.grid0 = target_w / s;
    gs.grid1 = target_h / s;
    grid_strides.push_back(gs);
  }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects) {
  if (faceobjects.empty()) return;

  std::vector<int> stack;
  stack.push_back(0);
  stack.push_back(faceobjects.size() - 1);

  while (!stack.empty()) {
    int right = stack.back();
    stack.pop_back();
    int left = stack.back();
    stack.pop_back();

    if (left >= right) continue;

    int i = left;
    int j = right;
    float pivot = faceobjects[i].prob;

    while (i < j) {
      while (i < j && faceobjects[j].prob <= pivot) j--;
      if (i < j) {
        Object tmp = faceobjects[i];
        faceobjects[i] = faceobjects[j];
        faceobjects[j] = tmp;
      }
      while (i < j && faceobjects[i].prob >= pivot) i++;
      if (i < j) {
        Object tmp = faceobjects[i];
        faceobjects[i] = faceobjects[j];
        faceobjects[j] = tmp;
      }
    }

    faceobjects[i].prob = pivot;
    stack.push_back(left);
    stack.push_back(i - 1);
    stack.push_back(i + 1);
    stack.push_back(right);
  }
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects,
                              std::vector<int> &picked, float nms_threshold) {
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
  }

  for (int i = 0; i < n; i++) {
    const Object &a = faceobjects[i];

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const Object &b = faceobjects[picked[j]];

      // intersection over union
      float inter_x1 = std::max(a.rect.x, b.rect.x);
      float inter_y1 = std::max(a.rect.y, b.rect.y);
      float inter_x2 =
          std::min(a.rect.x + a.rect.width, b.rect.x + b.rect.width);
      float inter_y2 =
          std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height);
      float inter_w = inter_x2 - inter_x1 > 0 ? inter_x2 - inter_x1 : 0;
      float inter_h = inter_y2 - inter_y1 > 0 ? inter_y2 - inter_y1 : 0;
      float inter_area = inter_w * inter_h;
      float a_area = areas[i];
      float b_area = areas[picked[j]];
      float u = a_area + b_area - inter_area;
      if (inter_area / u > nms_threshold) {
        keep = 0;
        break;
      }
    }

    if (keep) {
      picked.push_back(i);
    }
  }
}

static void decode_mask(const ncnn::Mat &mask_feat, const int &img_w,
                        const int &img_h, const ncnn::Mat &mask_proto,
                        const ncnn::Mat &in_pad, const int &wpad,
                        const int &hpad, ncnn::Mat &mask_pred_result) {
  // pass
  mask_pred_result = mask_feat;
}
}  // namespace yolov8

FFI_PLUGIN_EXPORT yolov8_err_t LoadModel(const char *model_path,
                                         const char *param_path,
                                         float nms_thresh, float conf_thresh,
                                         struct YoloV8 *yolov8) {
  try {
    yolov8 = new YoloV8();
    yolov8->model_path = model_path;
    yolov8->param_path = param_path;
    yolov8->nms_thresh = nms_thresh;
    yolov8->conf_thresh = conf_thresh;
    yolov8->target_size = 640;

    yolov8->net = new ncnn::Net();
    yolov8->net->load_param(yolov8->param_path);
    yolov8->net->load_model(yolov8->model_path);

    return YOLOvV8_OK;
  } catch (const std::exception &e) {
    return YOLOV8_ERROR;
  }
}

FFI_PLUGIN_EXPORT yolov8_err_t ReleaseModel(const struct YoloV8 *yolov8) {
  try {
    delete yolov8->net;
    delete yolov8;
    return YOLOvV8_OK;
  } catch (const std::exception &e) {
    return YOLOV8_ERROR;
  }
}

FFI_PLUGIN_EXPORT yolov8_err_t Inference(const struct YoloV8 *yolov8,
                                         const uint8_t *pixels,
                                         enum PixelType pixel_type,
                                         struct Result *result) {
  try {
    cv::Mat img;
    if (pixel_type == PIXEL_RGB) {
      img = cv::Mat(height, width, CV_8UC3, (void *)pixels);
    } else if (pixel_type == PIXEL_BGR) {
      img = cv::Mat(height, width, CV_8UC3, (void *)pixels);
      cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    } else if (pixel_type == PIXEL_GRAY) {
      img = cv::Mat(height, width, CV_8UC1, (void *)pixels);
    } else if (pixel_type == PIXEL_RGBA) {
      img = cv::Mat(height, width, CV_8UC4, (void *)pixels);
      cv::cvtColor(img, img, cv::COLOR_RGBA2RGB);
    } else if (pixel_type == PIXEL_BGRA) {
      img = cv::Mat(height, width, CV_8UC4, (void *)pixels);
      cv::cvtColor(img, img, cv::COLOR_BGRA2RGB);
    } else if (pixel_type == PIXEL_YUV) {
      img = cv::Mat(height, width, CV_8UC3, (void *)pixels);
      cv::cvtColor(img, img, cv::COLOR_YUV2RGB);
    } else {
      return YOLOV8_ERROR;
    }

    int width = img.cols;
    int height = img.rows;

    int wpad = 0;
    int hpad = 0;
    int target_size = yolov8->target_size;

    float scale =
        std::min((float)target_size / width, (float)target_size / height);
    int w = round(width * scale);
    int h = round(height * scale);

    if (w != target_size) {
      wpad = (target_size - w) / 2;
    }
    if (h != target_size) {
      hpad = (target_size - h) / 2;
    }

    cv::Mat in_pad;
    cv::copyMakeBorder(img, in_pad, hpad, hpad, wpad, wpad, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    ncnn::Extractor ex = yolov8->net->create_extractor();
    ex.set_num_threads(4);
    ex.input("data", in_pad);

    ncnn::Mat out;
    ex.extract("output", out);

    ncnn::Mat mask;
    ex.extract("output_mask", mask);

    std::vector<int> strides = {8, 16, 32};
    std::vector<yolov8::GridAndStride> grid_strides;
    yolov8::generate_grids_and_stride(target_size, target_size, strides,
                                      grid_strides);

    std::vector<Object> proposals;
    std::vector<Object> object8;
    std::vector<Object> object16;
    std::vector<Object> object32;

    yolov8::generate_proposals(grid_strides, out, yolov8->conf_thresh, object8);
    yolov8::generate_proposals(grid_strides, out, yolov8->conf_thresh,
                               object16);
    yolov8::generate_proposals(grid_strides, out, yolov8->conf_thresh,
                               object32);

    proposals.insert(proposals.end(), object8.begin(), object8.end());
    proposals.insert(proposals.end(), object16.begin(), object16.end());
    proposals.insert(proposals.end(), object32.begin(), object32.end());

    yolov8::qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    yolov8::nms_sorted_bboxes(proposals, picked, yolov8->nms_thresh);

    int count = picked.size();
    result->count = count;

    for (int i = 0; i < count; i++) {
      int z = picked[i];
      const Object &obj = proposals[z];

      cv::Mat mask_pred;
      yolov8::decode_mask(mask.channel(z), width, height, mask, in_pad, wpad,
                          hpad, mask_pred);

      Object object;
      object.rect = obj.rect;
      object.label = obj.label;
      object.prob = obj.prob;
      object.mask = mask_pred;
      object.mask_feat = obj.mask_feat;

      result->objects.push_back(object);
    }

    return YOLOvV8_OK;
  } catch (const std::exception &e) {
    return YOLOV8_ERROR;
  }
}