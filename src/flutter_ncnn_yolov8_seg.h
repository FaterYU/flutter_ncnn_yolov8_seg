#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ncnn
#include "net.h"

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

FFI_PLUGIN_EXPORT struct Rect {
  int x;
  int y;
  int width;
  int height;
};

FFI_PLUGIN_EXPORT struct ObjectSeg {
  int label;
  float prob;
  struct Rect rect;
  uint8_t* mask;
};

FFI_PLUGIN_EXPORT struct Yolo8Result {
  int count;
  struct ObjectSeg* objects;
  int latency;
};

FFI_PLUGIN_EXPORT struct Yolo8Model {
  char* paramPath;
  char* binPath;
};

FFI_PLUGIN_EXPORT struct Yolo8Result* createResult();

FFI_PLUGIN_EXPORT int destroyResult(struct Yolo8Result* result);

FFI_PLUGIN_EXPORT struct Yolo8Model* createModel();

FFI_PLUGIN_EXPORT int destroyModel(struct Yolo8Model* model);

FFI_PLUGIN_EXPORT int processImage(struct Yolo8Model* model, const uint8_t* pixels,
                                   struct Yolo8Result* result, int width,
                                   int height);

#ifdef __cplusplus
}
#endif
