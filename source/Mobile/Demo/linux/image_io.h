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

#include <string>
#include <vector>
#include "image.h"

namespace image {
namespace io {

class ImageReader {
public:
  bool operator()(const std::string& image_path,
                  unsigned char* data,
                  const size_t height,
                  const size_t width,
                  const size_t channel,
                  const Order order);
};

class ImageWriter {
public:
  bool operator()(const std::string& image_path,
                  const unsigned char* data,
                  const size_t height,
                  const size_t width,
                  const size_t channel,
                  const Order order);
};

}  // namespace io
}  // namespace image
