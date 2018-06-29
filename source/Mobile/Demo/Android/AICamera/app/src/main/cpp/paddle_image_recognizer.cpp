/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License */

#include "paddle_image_recognizer.h"
#include <string.h>
#include "binary_reader.h"

#define TAG "PaddlePaddle"

#ifdef __ANDROID__
#include <android/log.h>
#define LOGI(format, ...) \
  __android_log_print(ANDROID_LOG_INFO, TAG, format, ##__VA_ARGS__)
#define LOGW(format, ...) \
  __android_log_print(ANDROID_LOG_WARN, TAG, format, ##__VA_ARGS__)
#define LOGE(format, ...) \
  __android_log_print(ANDROID_LOG_ERROR, TAG, "Error: " format, ##__VA_ARGS__)
#else
#include <stdio.h>
#define LOGI(format, ...) \
  fprintf(stdout, "[" TAG "]" format "\n", ##__VA_ARGS__)
#define LOGW(format, ...) \
  fprintf(stdout, "[" TAG "]" format "\n", ##__VA_ARGS__)
#define LOGE(format, ...) \
  fprintf(stderr, "[" TAG "]Error: " format "\n", ##__VA_ARGS__)
#endif

static const char* paddle_error_string(paddle_error status) {
  switch (status) {
    case kPD_NULLPTR:
      return "nullptr error";
    case kPD_OUT_OF_RANGE:
      return "out of range error";
    case kPD_PROTOBUF_ERROR:
      return "protobuf error";
    case kPD_NOT_SUPPORTED:
      return "not supported error";
    case kPD_UNDEFINED_ERROR:
      return "undefined error";
    default:
      return "";
  };
}

#define CHECK(stmt)                                   \
  do {                                                \
    paddle_error __err__ = stmt;                      \
    if (__err__ != kPD_NO_ERROR) {                    \
      const char* str = paddle_error_string(__err__); \
      LOGE("%s (%d) in " #stmt "\n", str, __err__);   \
      exit(__err__);                                  \
    }                                                 \
  } while (0)

void ImageRecognizer::init_paddle() {
  static bool called = false;
  if (!called) {
    // Initalize Paddle
    char* argv[] = {const_cast<char*>("--use_gpu=False"),
                    const_cast<char*>("--pool_limit_size=0")};
    CHECK(paddle_init(2, (char**)argv));
    called = true;
  }
}

void ImageRecognizer::init(const char* merged_model_path,
                           const size_t normed_height,
                           const size_t normed_width,
                           const size_t normed_channel,
                           const std::vector<float>& means) {
  // Set the normed image size
  normed_height_ = normed_height;
  normed_width_ = normed_width;
  normed_channel_ = normed_channel;

  // Set means
  if (!means.empty()) {
    means_ = means;
  } else {
    means_.clear();
    for (size_t i = 0; i < normed_channel; ++i) {
      means_.push_back(0.0f);
    }
  }

  // Initialize PaddlePaddle environment.
  init_paddle();

  // Step 1: Reading merged model.
  LOGI("merged_model_path = %s", merged_model_path);
  long size;
  void* buf = BinaryReader()(merged_model_path, &size);

  // Step 2:
  //    Create a gradient machine for inference.
  CHECK(paddle_gradient_machine_create_for_inference_with_parameters(
      &gradient_machine_, buf, size));

  free(buf);
  buf = nullptr;
}

void ImageRecognizer::preprocess(const unsigned char* pixels,
                                 float* normed_pixels,
                                 const size_t height,
                                 const size_t width,
                                 const size_t channel,
                                 const image::Config& config) {
  bool need_resize = true;
  size_t resized_height = 0;
  size_t resized_width = 0;
  if (config.option == image::NO_ROTATE ||
      config.option == image::CLOCKWISE_R180) {
    if (height == normed_height_ && width == normed_width_) {
      need_resize = false;
    }
    resized_height = normed_height_;
    resized_width = normed_width_;
  } else if (config.option == image::CLOCKWISE_R90 ||
             config.option == image::CLOCKWISE_R270) {
    if (height == normed_width_ && width == normed_height_) {
      need_resize = false;
    }
    resized_height = normed_width_;
    resized_width = normed_height_;
  }

  unsigned char* resized_pixels = nullptr;
  if (!need_resize) {
    resized_pixels = const_cast<unsigned char*>(pixels);
  } else {
    // Bilinear Interpolation Resize
    resized_pixels = static_cast<unsigned char*>(malloc(
        resized_height * resized_width * channel * sizeof(unsigned char)));
    image::utils::resize_hwc(pixels,
                             resized_pixels,
                             height,
                             width,
                             channel,
                             resized_height,
                             resized_width);
  }

  unsigned char* rotated_pixels = nullptr;
  if (config.option == image::NO_ROTATE) {
    rotated_pixels = resized_pixels;
  } else {
    rotated_pixels = static_cast<unsigned char*>(malloc(
        normed_height_ * normed_width_ * channel * sizeof(unsigned char)));
    image::utils::rotate_hwc(resized_pixels,
                             rotated_pixels,
                             resized_height,
                             resized_width,
                             channel,
                             config.option);
  }

  if (true) {
    // HWC -> CHW
    size_t index = 0;
    if (config.format == image::kRGB) {
      // RGB/RGBA -> RGB
      for (size_t c = 0; c < normed_channel_; ++c) {
        for (size_t h = 0; h < normed_height_; ++h) {
          for (size_t w = 0; w < normed_width_; ++w) {
            normed_pixels[index] =
                static_cast<float>(
                    rotated_pixels[(h * normed_width_ + w) * channel + c]) -
                means_[c];
            index++;
          }
        }
      }
    } else if (config.format == image::kBGR) {
      // BGR/BGRA -> RGB
      for (size_t c = 0; c < normed_channel_; ++c) {
        for (size_t h = 0; h < normed_height_; ++h) {
          for (size_t w = 0; w < normed_width_; ++w) {
            normed_pixels[index] =
                static_cast<float>(
                    rotated_pixels[(h * normed_width_ + w) * channel +
                                   (normed_channel_ - 1 - c)]) -
                means_[c];
            index++;
          }
        }
      }
    }
  }

  if (rotated_pixels != nullptr && rotated_pixels != resized_pixels) {
    free(rotated_pixels);
    rotated_pixels = nullptr;
  }
  if (resized_pixels != nullptr && resized_pixels != pixels) {
    free(resized_pixels);
    resized_pixels = nullptr;
  }
}

void ImageRecognizer::infer(const unsigned char* pixels,
                            const size_t height,
                            const size_t width,
                            const size_t channel,
                            const image::Config& config,
                            Result& result) {
  if (height < normed_height_ || width < normed_width_) {
    LOGE(
        "Image size should be no less than the normed size (%u vs %u, %u vs "
        "%u).\n",
        height,
        normed_height_,
        width,
        normed_width_);
    return;
  }

  LOGI("height = %ld, width = %ld, channel = %ld\n", height, width, channel);

  // Step 3: Prepare input Arguments
  paddle_arguments in_args = paddle_arguments_create_none();

  // There is only one input of this network.
  CHECK(paddle_arguments_resize(in_args, 1));

  // Create input matrix.
  // Set the value
  paddle_matrix mat = paddle_matrix_create(
      /* sample_num */ 1,
      /* size */ normed_channel_ * normed_height_ * normed_width_,
      /* useGPU */ false);
  CHECK(paddle_arguments_set_value(in_args, 0, mat));

  // Get First row.
  paddle_real* array;
  CHECK(paddle_matrix_get_row(mat, 0, &array));

  preprocess(pixels, array, height, width, channel, config);

  // Step 4: Do inference.
  paddle_arguments out_args = paddle_arguments_create_none();
  {
    CHECK(paddle_gradient_machine_forward(gradient_machine_,
                                          in_args,
                                          out_args,
                                          /* isTrain */ false));
  }

  // Step 5: Get the result.
  paddle_matrix probs = paddle_matrix_create_none();
  CHECK(paddle_arguments_get_value(out_args, 0, probs));

  paddle_error err = paddle_matrix_get_row(probs, 0, &result.data);
  if (err == kPD_NO_ERROR) {
    CHECK(paddle_matrix_get_shape(probs, &result.height, &result.width));
  }

  // Step 6: Release the resources.
  CHECK(paddle_arguments_destroy(in_args));
  CHECK(paddle_matrix_destroy(mat));
  CHECK(paddle_arguments_destroy(out_args));
  CHECK(paddle_matrix_destroy(probs));
}

void ImageRecognizer::release() {
  if (gradient_machine_ != nullptr) {
    CHECK(paddle_gradient_machine_destroy(gradient_machine_));
  }
}
