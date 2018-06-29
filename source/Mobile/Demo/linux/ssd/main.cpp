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

#include <string.h>
#include <iostream>
#include <vector>
#include "image_io.h"
#include "paddle_image_recognizer.h"

void profile(ImageRecognizer::Result& result, float threshold) {
  std::string labels[21] = {
      "background", "aeroplane",   "bicycle", "bird",  "boat",
      "bottle",     "bus",         "car",     "cat",   "chair",
      "cow",        "diningtable", "dog",     "horse", "motorbike",
      "person",     "pottedplant", "sheep",   "sofa",  "train",
      "tvmonitor"};

  for (uint64_t i = 0; i < result.height; i++) {
    if (result.width == 7UL && result.data[i * result.width + 2] >= threshold) {
      std::cout << "Object " << i << std::endl;
      std::cout << "\timage: "
                << static_cast<int>(result.data[i * result.width + 0])
                << std::endl;
      std::cout << "\ttype: "
                << labels[static_cast<int>(result.data[i * result.width + 1])]
                << std::endl;
      std::cout << "\tscore: " << result.data[i * result.width + 2]
                << std::endl;
      std::cout << "\trectangle information:" << std::endl;
      std::cout << "\t\txmin, " << result.data[i * result.width + 3]
                << std::endl;
      std::cout << "\t\tymin, " << result.data[i * result.width + 4]
                << std::endl;
      std::cout << "\t\txmax, " << result.data[i * result.width + 5]
                << std::endl;
      std::cout << "\t\tymax, " << result.data[i * result.width + 6]
                << std::endl;
    }
  }
  std::cout << std::endl;
}

void draw_rectangles(const unsigned char* raw_pixels,
                     unsigned char* rected_pixels,
                     const size_t height,
                     const size_t width,
                     const size_t channel,
                     const float threshold,
                     ImageRecognizer::Result& result) {
  if (rected_pixels != raw_pixels) {
    memcpy(rected_pixels,
           raw_pixels,
           height * width * channel * sizeof(unsigned char));
  }
  for (uint64_t i = 0; i < result.height; i++) {
    if (result.width == 7UL && result.data[i * result.width + 2] >= threshold) {
      size_t xmin = result.data[i * result.width + 3] * width;
      size_t ymin = result.data[i * result.width + 4] * height;
      size_t xmax = result.data[i * result.width + 5] * width;
      size_t ymax = result.data[i * result.width + 6] * height;
      size_t channel_of_red = 2;
      // y = ymin
      for (size_t x = xmin; x < xmax; ++x) {
        rected_pixels[(ymin * width + x) * channel + channel_of_red] = 255;
      }
      for (size_t y = ymin + 1; y < ymax - 1; ++y) {
        // x = xmin
        rected_pixels[(y * width + xmin) * channel + channel_of_red] = 255;
        // x = xmax - 1
        rected_pixels[(y * width + xmax - 1) * channel + channel_of_red] = 255;
      }
      // y = ymax - 1
      for (size_t x = xmin; x < xmax; ++x) {
        rected_pixels[((ymax - 1) * width + x) * channel + channel_of_red] =
            255;
      }
    }
  }
}

void test_noresize(ImageRecognizer& recognizer,
                   const size_t kImageHeight,
                   const size_t kImageWidth,
                   const size_t kImageChannel,
                   ImageRecognizer::Result& result) {
  // Read BGR data from image
  unsigned char* raw_pixels = (unsigned char*)malloc(
      kImageHeight * kImageWidth * kImageChannel * sizeof(unsigned char));
  image::io::ImageReader()("ssd/images/resized.jpg",
                           raw_pixels,
                           kImageHeight,
                           kImageWidth,
                           kImageChannel,
                           image::kHWC);

  image::Config config(image::kBGR, image::NO_ROTATE);
  recognizer.infer(
      raw_pixels, kImageHeight, kImageWidth, kImageChannel, config, result);

  free(raw_pixels);
  raw_pixels = nullptr;
}

void test_resize(ImageRecognizer& recognizer,
                 const size_t kImageHeight,
                 const size_t kImageWidth,
                 const size_t kImageChannel,
                 ImageRecognizer::Result& result) {
  const size_t height = 500;
  const size_t width = 353;
  const size_t channel = 3;

  // Read BGR data from image
  unsigned char* raw_pixels =
      (unsigned char*)malloc(height * width * channel * sizeof(unsigned char));
  image::io::ImageReader()(
      "ssd/images/origin.jpg", raw_pixels, height, width, channel, image::kHWC);

  image::Config config(image::kBGR, image::NO_ROTATE);
  recognizer.infer(raw_pixels, height, width, channel, config, result);

  // Draw rectangles
  unsigned char* rected_pixels =
      (unsigned char*)malloc(height * width * channel * sizeof(unsigned char));
  draw_rectangles(
      raw_pixels, rected_pixels, height, width, channel, 0.3, result);
  image::io::ImageWriter()("ssd/images/origin_result.jpg",
                           rected_pixels,
                           height,
                           width,
                           channel,
                           image::kHWC);

  free(raw_pixels);
  raw_pixels = nullptr;
  free(rected_pixels);
  raw_pixels = nullptr;
}

void test_rgba(ImageRecognizer& recognizer,
               const size_t kImageHeight,
               const size_t kImageWidth,
               const size_t kImageChannel,
               ImageRecognizer::Result& result) {
  const size_t height = 500;
  const size_t width = 353;
  const size_t channel = 3;

  // Read BGR data from image
  unsigned char* raw_pixels =
      (unsigned char*)malloc(height * width * channel * sizeof(unsigned char));
  image::io::ImageReader()(
      "ssd/images/origin.jpg", raw_pixels, height, width, channel, image::kHWC);

  // Padding to BGRA, for testing
  // Only BGR is needed
  const size_t channel_rgba = 4;
  unsigned char* pixels = (unsigned char*)malloc(height * width * channel_rgba *
                                                 sizeof(unsigned char));
  for (size_t i = 0; i < height * width; ++i) {
    pixels[i * channel_rgba + 0] = raw_pixels[i * channel + 0];
    pixels[i * channel_rgba + 1] = raw_pixels[i * channel + 1];
    pixels[i * channel_rgba + 2] = raw_pixels[i * channel + 2];
    pixels[i * channel_rgba + 3] = 0;  // alpha
  }

  image::Config config(image::kBGR, image::NO_ROTATE);
  recognizer.infer(pixels, height, width, channel_rgba, config, result);

  free(raw_pixels);
  raw_pixels = nullptr;
  free(pixels);
  pixels = nullptr;
}

void test_rotate(ImageRecognizer& recognizer,
                 const size_t kImageHeight,
                 const size_t kImageWidth,
                 const size_t kImageChannel,
                 ImageRecognizer::Result& result) {
  const size_t height = 353;
  const size_t width = 500;
  const size_t channel = 3;

  // Read BGR data from image
  unsigned char* raw_pixels =
      (unsigned char*)malloc(height * width * channel * sizeof(unsigned char));
  image::io::ImageReader()("ssd/images/rotated.jpg",
                           raw_pixels,
                           height,
                           width,
                           channel,
                           image::kHWC);

  image::Config config(image::kBGR, image::CLOCKWISE_R90);
  recognizer.infer(raw_pixels, height, width, channel, config, result);

  free(raw_pixels);
  raw_pixels = nullptr;
}

int main() {
  ImageRecognizer::init_paddle();

  const char* merged_model_path = "ssd/models/vgg_ssd_net.paddle";

  const size_t kImageHeight = 300;
  const size_t kImageWidth = 300;
  const size_t kImageChannel = 3;

  const std::vector<float> means({104, 117, 124});

  ImageRecognizer recognizer;
  recognizer.init(
      merged_model_path, kImageHeight, kImageWidth, kImageChannel, means);

  ImageRecognizer::Result result;
  test_resize(recognizer, kImageHeight, kImageWidth, kImageChannel, result);

  // Print the direct result
  std::cout << "Direct Result: " << result.height << " x " << result.width
            << std::endl;
  for (uint64_t i = 0; i < result.height; i++) {
    std::cout << "row " << i << ":";
    for (uint64_t j = 0; j < result.width; j++) {
      std::cout << " " << result.data[i * result.width + j];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  // Print the profiled result
  std::cout << "Profiled result" << std::endl;
  profile(result, /* threshold */ 0);

  // You may need to use a threshold to filter out objects with low score
  std::cout << "Profiled result (threshold = 0.3)" << std::endl;
  profile(result, /* threshold */ 0.3);

  recognizer.release();

  return 0;
}
