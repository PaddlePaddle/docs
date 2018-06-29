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

#pragma once

#include <paddle/capi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "image_utils.h"

class ImageRecognizer {
public:
  struct Result {
    Result() : data(nullptr), height(0), width(0) {}

    float* data;
    uint64_t height;
    uint64_t width;
  };

public:
  ImageRecognizer()
      : gradient_machine_(nullptr),
        normed_height_(0),
        normed_width_(0),
        normed_channel_(0) {}

  void init(const char* merged_model_path,
            const size_t normed_height,
            const size_t normed_width,
            const size_t normed_channel,
            const std::vector<float>& means);
  void infer(const unsigned char* pixels,
             const size_t height,
             const size_t width,
             const size_t channel,
             const image::Config& config,
             Result& result);
  void release();

protected:
  void init_paddle();

  void preprocess(const unsigned char* pixels,
                  float* normed_pixels,
                  const size_t height,
                  const size_t width,
                  const size_t channel,
                  const image::Config& config);

private:
  paddle_gradient_machine gradient_machine_;

  size_t normed_height_;
  size_t normed_width_;
  size_t normed_channel_;
  std::vector<float> means_;
};
