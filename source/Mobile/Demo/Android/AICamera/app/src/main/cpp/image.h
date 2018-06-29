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

namespace image {

enum Order { kCHW = 0, kHWC = 1 };

enum Format {
  kRGB = 0x1,  // support RGB, RGBA
  kBGR = 0x2   // support BGR, BGRA
};

enum RotateOption {
  NO_ROTATE = 0,
  CLOCKWISE_R90 = 1,
  CLOCKWISE_R180 = 2,
  CLOCKWISE_R270 = 3
};

struct Config {
  Config() : format(kRGB), option(NO_ROTATE) {}
  Config(Format f, RotateOption o) : format(f), option(o) {}
  Format format;
  RotateOption option;
};

}  // namespace image
