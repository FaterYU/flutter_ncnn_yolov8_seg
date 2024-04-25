#include "flutter_ncnn_yolov8_seg.h"


#include <cstdlib>

#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

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

namespace yolo8 {
static void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end,
                  int axis) {
  ncnn::Option opt;
  opt.num_threads = 4;
  opt.use_fp16_storage = false;
  opt.use_packing_layout = false;

  ncnn::Layer* op = ncnn::create_layer("Crop");

  // set param
  ncnn::ParamDict pd;

  ncnn::Mat axes = ncnn::Mat(1);
  axes.fill(axis);
  ncnn::Mat ends = ncnn::Mat(1);
  ends.fill(end);
  ncnn::Mat starts = ncnn::Mat(1);
  starts.fill(start);
  pd.set(9, starts);  // start
  pd.set(10, ends);   // end
  pd.set(11, axes);   // axes

  op->load_param(pd);

  op->create_pipeline(opt);

  // forward
  op->forward(in, out, opt);

  op->destroy_pipeline(opt);

  delete op;
}
static void interp(const ncnn::Mat& in, const float& scale, const int& out_w,
                   const int& out_h, ncnn::Mat& out) {
  ncnn::Option opt;
  opt.num_threads = 1;
  opt.use_fp16_storage = false;
  opt.use_packing_layout = false;

  ncnn::Layer* op = ncnn::create_layer("Interp");

  // set param
  ncnn::ParamDict pd;
  pd.set(0, 2);      // resize_type
  pd.set(1, scale);  // height_scale
  pd.set(2, scale);  // width_scale
  pd.set(3, out_h);  // height
  pd.set(4, out_w);  // width

  op->load_param(pd);

  op->create_pipeline(opt);

  // forward
  op->forward(in, out, opt);

  op->destroy_pipeline(opt);

  delete op;
}
static void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w,
                    int d) {
  ncnn::Option opt;
  opt.num_threads = 4;
  opt.use_fp16_storage = false;
  opt.use_packing_layout = false;

  ncnn::Layer* op = ncnn::create_layer("Reshape");

  // set param
  ncnn::ParamDict pd;

  pd.set(0, w);              // start
  pd.set(1, h);              // end
  if (d > 0) pd.set(11, d);  // axes
  pd.set(2, c);              // axes
  op->load_param(pd);

  op->create_pipeline(opt);

  // forward
  op->forward(in, out, opt);

  op->destroy_pipeline(opt);

  delete op;
}
static void sigmoid(ncnn::Mat& bottom) {
  ncnn::Option opt;
  opt.num_threads = 4;
  opt.use_fp16_storage = false;
  opt.use_packing_layout = false;

  ncnn::Layer* op = ncnn::create_layer("Sigmoid");

  op->create_pipeline(opt);

  // forward

  op->forward_inplace(bottom, opt);
  op->destroy_pipeline(opt);

  delete op;
}
static void matmul(const std::vector<ncnn::Mat>& bottom_blobs,
                   ncnn::Mat& top_blob) {
  ncnn::Option opt;
  opt.num_threads = 2;
  opt.use_fp16_storage = false;
  opt.use_packing_layout = false;

  ncnn::Layer* op = ncnn::create_layer("MatMul");

  // set param
  ncnn::ParamDict pd;
  pd.set(0, 0);  // axis

  op->load_param(pd);

  op->create_pipeline(opt);
  std::vector<ncnn::Mat> top_blobs(1);
  op->forward(bottom_blobs, top_blobs, opt);
  top_blob = top_blobs[0];

  op->destroy_pipeline(opt);

  delete op;
}

struct Object {
  cv::Rect_<float> rect;
  int label;
  float prob;
  cv::Mat mask;
  std::vector<float> mask_feat;
};
struct GridAndStride {
  int grid0;
  int grid1;
  int stride;
};
static inline float intersection_area(const Object& a, const Object& b) {
  cv::Rect_<float> inter = a.rect & b.rect;
  return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left,
                                  int right) {
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].prob;

  while (i <= j) {
    while (faceobjects[i].prob > p) i++;

    while (faceobjects[j].prob < p) j--;

    if (i <= j) {
      // swap
      std::swap(faceobjects[i], faceobjects[j]);

      i++;
      j--;
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (left < j) qsort_descent_inplace(faceobjects, left, j);
    }
#pragma omp section
    {
      if (i < right) qsort_descent_inplace(faceobjects, i, right);
    }
  }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects) {
  if (faceobjects.empty()) return;

  qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects,
                              std::vector<int>& picked, float nms_threshold) {
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
  }

  for (int i = 0; i < n; i++) {
    const Object& a = faceobjects[i];

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const Object& b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = intersection_area(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold) keep = 0;
    }

    if (keep) picked.push_back(i);
  }
}
inline float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }
static void generate_proposals(std::vector<GridAndStride> grid_strides,
                               const ncnn::Mat& pred, float prob_threshold,
                               std::vector<Object>& objects) {
  const int num_points = grid_strides.size();
  const int num_class = 14;
  const int reg_max_1 = 16;

  for (int i = 0; i < num_points; i++) {
    const float* scores = pred.row(i) + 4 * reg_max_1;

    // find label with max score
    int label = -1;
    float score = -FLT_MAX;
    for (int k = 0; k < num_class; k++) {
      float confidence = scores[k];
      if (confidence > score) {
        label = k;
        score = confidence;
      }
    }
    float box_prob = sigmoid(score);
    if (box_prob >= prob_threshold) {
      ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
      {
        ncnn::Layer* softmax = ncnn::create_layer("Softmax");

        ncnn::ParamDict pd;
        pd.set(0, 1);  // axis
        pd.set(1, 1);
        softmax->load_param(pd);

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;

        softmax->create_pipeline(opt);

        softmax->forward_inplace(bbox_pred, opt);

        softmax->destroy_pipeline(opt);

        delete softmax;
      }

      float pred_ltrb[4];
      for (int k = 0; k < 4; k++) {
        float dis = 0.f;
        const float* dis_after_sm = bbox_pred.row(k);
        for (int l = 0; l < reg_max_1; l++) {
          dis += l * dis_after_sm[l];
        }

        pred_ltrb[k] = dis * grid_strides[i].stride;
      }

      float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
      float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

      float x0 = pb_cx - pred_ltrb[0];
      float y0 = pb_cy - pred_ltrb[1];
      float x1 = pb_cx + pred_ltrb[2];
      float y1 = pb_cy + pred_ltrb[3];

      Object obj;
      obj.rect.x = x0;
      obj.rect.y = y0;
      obj.rect.width = x1 - x0;
      obj.rect.height = y1 - y0;
      obj.label = label;
      obj.prob = box_prob;
      obj.mask_feat.resize(32);
      std::copy(pred.row(i) + 64 + num_class, pred.row(i) + 64 + num_class + 32,
                obj.mask_feat.begin());
      objects.push_back(obj);
    }
  }
}
static void generate_grids_and_stride(
    const int target_w, const int target_h, std::vector<int>& strides,
    std::vector<GridAndStride>& grid_strides) {
  for (int i = 0; i < (int)strides.size(); i++) {
    int stride = strides[i];
    int num_grid_w = target_w / stride;
    int num_grid_h = target_h / stride;
    for (int g1 = 0; g1 < num_grid_h; g1++) {
      for (int g0 = 0; g0 < num_grid_w; g0++) {
        GridAndStride gs;
        gs.grid0 = g0;
        gs.grid1 = g1;
        gs.stride = stride;
        grid_strides.push_back(gs);
      }
    }
  }
}
static void decode_mask(const ncnn::Mat& mask_feat, const int& img_w,
                        const int& img_h, const ncnn::Mat& mask_proto,
                        const ncnn::Mat& in_pad, const int& wpad,
                        const int& hpad, ncnn::Mat& mask_pred_result) {
  ncnn::Mat masks;
  matmul(std::vector<ncnn::Mat>{mask_feat, mask_proto}, masks);
  sigmoid(masks);
  reshape(masks, masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0);
  slice(masks, mask_pred_result, (wpad / 2) / 4, (in_pad.w - wpad / 2) / 4, 2);
  slice(mask_pred_result, mask_pred_result, (hpad / 2) / 4,
        (in_pad.h - hpad / 2) / 4, 1);
  interp(mask_pred_result, 4.0, img_w, img_h, mask_pred_result);
}
static int detect_yolov8(const uint8_t* pixels, std::vector<Object>& objects,
                         ncnn::Extractor ex, int width, int height) {
  const int target_size = 640;
  const float prob_threshold = 0.4f;
  const float nms_threshold = 0.5f;

  // pad to multiple of 32
  int w = width;
  int h = height;
  float scale = 1.f;
  if (w > h) {
    scale = (float)target_size / w;
    w = target_size;
    h = h * scale;
  } else {
    scale = (float)target_size / h;
    h = target_size;
    w = w * scale;
  }


  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      pixels, ncnn::Mat::PIXEL_RGBA2RGB, width, height, w, h);

  // pad to target_size rectangle
  int wpad = (w + 31) / 32 * 32 - w;
  int hpad = (h + 31) / 32 * 32 - h;
  ncnn::Mat in_pad;
  ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2,
                         wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  in_pad.substract_mean_normalize(0, norm_vals);

  ex.input("images", in_pad);

  //   ncnn::Mat out;
  //   ex.extract("output", out);

  //   ncnn::Mat mask_proto;
  //   ex.extract("seg", mask_proto);

  ncnn::Mat out;
  ex.extract("output0", out);

  ncnn::Mat mask_proto;
  ex.extract("output1", mask_proto);

  std::vector<int> strides = {8, 16, 32};
  std::vector<GridAndStride> grid_strides;
  generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);

  std::vector<Object> proposals;
  std::vector<Object> objects8;
  generate_proposals(grid_strides, out, prob_threshold, objects8);

  proposals.insert(proposals.end(), objects8.begin(), objects8.end());

  // sort all proposals by score from highest to lowest
  qsort_descent_inplace(proposals);

  // apply nms with nms_threshold
  std::vector<int> picked;
  nms_sorted_bboxes(proposals, picked, nms_threshold);

  int count = picked.size();

  ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
  for (int i = 0; i < count; i++) {
    float* mask_feat_ptr = mask_feat.row(i);
    std::memcpy(mask_feat_ptr, proposals[picked[i]].mask_feat.data(),
                sizeof(float) * proposals[picked[i]].mask_feat.size());
  }

  ncnn::Mat mask_pred_result;
  decode_mask(mask_feat, width, height, mask_proto, in_pad, wpad, hpad,
              mask_pred_result);

  objects.resize(count);
  for (int i = 0; i < count; i++) {
    objects[i] = proposals[picked[i]];

    // adjust offset to original unpadded
    float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
    float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
    float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
    float y1 =
        (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

    // clip
    x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

    objects[i].rect.x = x0;
    objects[i].rect.y = y0;
    objects[i].rect.width = x1 - x0;
    objects[i].rect.height = y1 - y0;

    objects[i].mask = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat mask =
        cv::Mat(height, width, CV_32FC1, (float*)mask_pred_result.channel(i));
    mask(objects[i].rect).copyTo(objects[i].mask(objects[i].rect));
  }

  return 0;
}
}  // namespace yolo8

FFI_PLUGIN_EXPORT struct Yolo8Result* createResult() {
  struct Yolo8Result* result = new struct Yolo8Result();
  return result;
}

FFI_PLUGIN_EXPORT int destroyResult(struct Yolo8Result* result) {
  if (result != NULL) {
    if (result->objects != NULL) {
      for (int i = 0; i < result->count; i++) {
        if (result->objects[i].mask != NULL) {
          delete[] result->objects[i].mask;
          result->objects[i].mask = NULL;
        }
      }
      delete[] result->objects;
      result->objects = NULL;
    }
    delete result;
    result = NULL;
    return 0;
  }
  return -1;
}

FFI_PLUGIN_EXPORT struct Yolo8Model* createModel() {
  struct Yolo8Model* model = new struct Yolo8Model();
  return model;
}

FFI_PLUGIN_EXPORT int destroyModel(struct Yolo8Model* model) {
  if (model != NULL) {
    free(model);
    model = NULL;
    return 0;
  }
  return -1;
}

FFI_PLUGIN_EXPORT int processImage(struct Yolo8Model* model, const uint8_t* pixels,
                                   struct Yolo8Result* result, int width,
                                   int height) {  
  using namespace yolo8;
  std::vector<struct yolo8::Object> objects;



  // ncnn::Extractor ex = *model->extractor;

  ncnn::Net model8;

  model8.load_param(model->paramPath);
  model8.load_model(model->binPath);

  ncnn::Extractor ex = model8.create_extractor();

  yolo8::detect_yolov8(pixels, objects, ex, width, height);

  result->count = objects.size();
  result->objects = new ObjectSeg[objects.size()];
  for (int i = 0; i < objects.size(); i++) {
    result->objects[i].label = objects[i].label;
    result->objects[i].prob = objects[i].prob;
    result->objects[i].rect.x = objects[i].rect.x;
    result->objects[i].rect.y = objects[i].rect.y;
    result->objects[i].rect.width = objects[i].rect.width;
    result->objects[i].rect.height = objects[i].rect.height;
    result->objects[i].mask = new uint8_t[width * height];
    for (int j = 0; j < width * height; j++) {
      result->objects[i].mask[j] = objects[i].mask.data[j];
    }
  }
  result->latency = objects.size();

  return objects.size();
}