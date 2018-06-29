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

cmake_minimum_required(VERSION 3.0)
project(paddle_demo CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")
find_package(Paddle)
find_package(OpenCV)

set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
  
function(add_paddle_demo TARGET_NAME)
  if(NOT ARGN)
    message(FATAL_ERROR "Please specify source files for ${TARGET_NAME}")
  endif()
  set(SRC_FILES ${ARGN})
  add_executable(${TARGET_NAME} ${SRC_FILES})
  target_link_libraries(${TARGET_NAME}
                        ${PADDLE_LIBRARIES}
                        ${PADDLE_THIRD_PARTY_LIBRARIES})
  if(NOT ANDROID_ABI)
    target_link_libraries(${TARGET_NAME} -lrt -ldl -lpthread)
  endif()
  if(OPENCV_FOUND)
    target_link_libraries(${TARGET_NAME} ${OPENCV_LIBRARIES})
  endif()
endfunction()

set(COMMON_SRC_FILES image_io.cpp image_utils.cpp paddle_image_recognizer.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ssd
set(SSD_SRC_FILES ${COMMON_SRC_FILES} ssd/main.cpp)
add_paddle_demo(ssd_demo ${SSD_SRC_FILES})
