# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

set(OPENCV_ROOT $ENV{OPENCV_ROOT} CACHE PATH "OpenCV Path")
set(OPENCV_FOUND OFF)
if(OPENCV_ROOT)
  find_path(OPENCV_INC_DIR opencv2/opencv.hpp PATHS ${OPENCV_ROOT}/include)
  find_library(OPENCV_CORE_LIB NAMES opencv_core PATHS ${OPENCV_ROOT}/lib)
  find_library(OPENCV_HIGHGUI_LIB NAMES opencv_highgui PATHS
      ${OPENCV_ROOT}/lib)
  find_library(OPENCV_IMGPROC_LIB NAMES opencv_imgproc PATHS
      ${OPENCV_ROOT}/lib)
  if(OPENCV_INC_DIR AND OPENCV_CORE_LIB AND OPENCV_HIGHGUI_LIB AND OPENCV_IMGPROC_LIB)
    include_directories(${OPENCV_INC_DIR})
    add_definitions(-DUSE_OPENCV)
    message(STATUS "Found OpenCV: ${OPENCV_CORE_LIB} "
            "${OPENCV_HIGHGUI_LIB} ${OPENCV_IMGPROC_LIB}")
  
    set(OPENCV_FOUND ON)
    set(OPENCV_LIBRARIES ${OPENCV_CORE_LIB} ${OPENCV_HIGHGUI_LIB} ${OPENCV_IMGPROC_LIB})
  endif()
endif()
