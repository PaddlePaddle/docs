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

#include "image_utils.h"
#include <stdlib.h>
#include <string.h>

namespace image {
namespace utils {

void resize_hwc(const unsigned char* pixels,
                unsigned char* resized_pixels,
                const size_t height,
                const size_t width,
                const size_t channel,
                const size_t resized_height,
                const size_t resized_width) {
  float ratio_x = static_cast<float>(width) / static_cast<float>(resized_width);
  float ratio_y =
      static_cast<float>(height) / static_cast<float>(resized_height);

  for (size_t i = 0; i < resized_height; i++) {
    float new_y = i * ratio_y;

    size_t y1 = (static_cast<size_t>(new_y) > (height - 1))
                    ? (height - 1)
                    : static_cast<size_t>(new_y);
    size_t y2 = y1 + 1;

    float b1 = y2 - new_y;
    float b2 = new_y - y1;

    for (size_t j = 0; j < resized_width; j++) {
      float new_x = j * ratio_x;

      size_t x1 = (static_cast<size_t>(new_x) > (width - 1))
                      ? (width - 1)
                      : static_cast<size_t>(new_x);
      int x2 = x1 + 1;

      float a1 = x2 - new_x;
      float a2 = new_x - x1;

      unsigned char* pt_dst =
          resized_pixels + (i * resized_width + j) * channel;
      const unsigned char* pt_src = pixels + (y1 * width + x1) * channel;
      int p1 = width * channel;
      int p2 = p1 + channel;

      if (x1 == width - 1 && y1 == height - 1) {
        memcpy(pt_dst, pt_src, channel * sizeof(unsigned char));
      } else if (x1 == width - 1) {
        for (size_t k = 0; k < channel; k++) {
          float pixel_00 = static_cast<float>(pt_src[k]);
          float pixel_10 = static_cast<float>(pt_src[p1 + k]);

          pt_dst[k] = static_cast<unsigned char>(pixel_00 * b1 + pixel_10 * b2);
        }
      } else if (y1 == height - 1) {
        for (size_t k = 0; k < channel; k++) {
          float pixel_00 = static_cast<float>(pt_src[k]);
          float pixel_01 = static_cast<float>(pt_src[channel + k]);

          pt_dst[k] = static_cast<unsigned char>(pixel_00 * a1 + pixel_01 * a2);
        }
      } else {
        // If x1 = width - 1 or y1 = height - 1, the memory accesses may be out
        // of range.
        for (size_t k = 0; k < channel; k++) {
          float pixel_00 = static_cast<float>(pt_src[k]);
          float pixel_01 = static_cast<float>(pt_src[channel + k]);
          float pixel_10 = static_cast<float>(pt_src[p1 + k]);
          float pixel_11 = static_cast<float>(pt_src[p2 + k]);

          pt_dst[k] =
              static_cast<unsigned char>((pixel_00 * a1 + pixel_01 * a2) * b1 +
                                         (pixel_10 * a1 + pixel_11 * a2) * b2);
        }
      }
    }  // j-loop
  }    // i-loop
}

void rotate_hwc(const unsigned char* pixels,
                unsigned char* rotated_pixels,
                const size_t height,
                const size_t width,
                const size_t channel,
                const RotateOption option) {
  switch (option) {
    case NO_ROTATE:
      memcpy(rotated_pixels,
             pixels,
             height * width * channel * sizeof(unsigned char));
      break;
    case CLOCKWISE_R90:
      for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
          // (i, j) -> (j, height - 1 - i)
          for (size_t k = 0; k < channel; ++k) {
            rotated_pixels[(j * height + height - 1 - i) * channel + k] =
                pixels[(i * width + j) * channel + k];
          }
        }
      }
      break;
    case CLOCKWISE_R180:
      for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
          // (i, j) -> (height - 1 - i, width - 1 - j)
          for (size_t k = 0; k < channel; ++k) {
            rotated_pixels[((height - 1 - i) * width + width - 1 - j) *
                               channel +
                           k] = pixels[(i * width + j) * channel + k];
          }
        }
      }
      break;
    case CLOCKWISE_R270:
      for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
          // (i, j) -> (width - 1 - j, i)
          for (size_t k = 0; k < channel; ++k) {
            rotated_pixels[((width - 1 - j) * height + i) * channel + k] =
                pixels[(i * width + j) * channel + k];
          }
        }
      }
      break;
    default:
      fprintf(stderr,
              "Illegal rotate option, please specify among [NO_ROTATE, "
              "CLOCKWISE_R90, CLOCKWISE_R180, CLOCKWISE_R270].\n");
  }
}

}  // namespace utils
}  // namespace image
