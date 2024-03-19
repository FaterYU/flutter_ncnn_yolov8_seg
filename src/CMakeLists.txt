# The Flutter tooling requires that developers have CMake 3.10 or later
# installed. You should not increase this version, as doing so will cause
# the plugin to fail to compile for some customers of the plugin.
cmake_minimum_required(VERSION 3.10)

project(flutter_ncnn_yolov8_seg_library VERSION 0.0.1 LANGUAGES C CXX)

get_filename_component(MY_PROJ ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

# packages
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(ncnn_DIR "${MY_PROJ}/linux/ncnn-20230517-ubuntu-2204-shared/lib/cmake")
  set(OpenCV_DIR "${MY_PROJ}/linux/opencv-mobile-4.6.0-ubuntu-2204/lib/cmake")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(ncnn_DIR "${MY_PROJ}/android/ncnn-20240102-android-vulkan-shared/${ANDROID_ABI}/lib/cmake/ncnn")
  set(OpenCV_DIR "${MY_PROJ}/android/opencv-mobile-4.9.0-android/sdk/native/jni")
else()
  message(FATAL_ERROR "system not support: ${CMAKE_SYSTEM_NAME}")
endif()

if(NOT EXISTS ${ncnn_DIR})
  message(FATAL_ERROR "ncnn_DIR not exists: ${ncnn_DIR}")
endif()

if(NOT EXISTS ${OpenCV_DIR})
  message(FATAL_ERROR "OpenCV_DIR not exists: ${OpenCV_DIR}")
endif()

# # ncnn
find_package(ncnn REQUIRED)
message(STATUS "ncnn_FOUND: ${ncnn_FOUND}")

# # opencv
find_package(OpenCV 4 REQUIRED)
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# targets
include_directories(
  ${MY_PROJ}/src
  ${OpenCV_INCLUDE_DIRS}
)

# # flutter_ncnn_yolov8_seg
add_library(flutter_ncnn_yolov8_seg SHARED
  "flutter_ncnn_yolov8_seg.cc"
)
target_link_libraries(flutter_ncnn_yolov8_seg ncnn ${OpenCV_LIBS})

set_target_properties(flutter_ncnn_yolov8_seg PROPERTIES
  PUBLIC_HEADER flutter_ncnn_yolov8_seg.h
  OUTPUT_NAME "flutter_ncnn_yolov8_seg"
)

target_compile_definitions(flutter_ncnn_yolov8_seg PUBLIC DART_SHARED_LIB)

# # flutter_ncnn_yolov8_seg_test
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(flutter_ncnn_yolov8_seg_test
    ${MY_PROJ}/linux/flutter_ncnn_yolov8_seg_test.cc
  )
  target_link_libraries(flutter_ncnn_yolov8_seg_test flutter_ncnn_yolov8_seg ${OpenCV_LIBS})
  set_target_properties(flutter_ncnn_yolov8_seg_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${MY_PROJ}/linux/build"
  )
endif()
