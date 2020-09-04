mkdir -p build
cd build
rm -rf *

# same with the resnet50_test.cc
DEMO_NAME=resnet50_trt_test

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=ON

LIB_DIR=/paddle/build/fluid_inference_install_dir
CUDNN_LIB=/paddle/nvidia-downloads/cudnn_v7.6_cuda10.1/lib64
CUDA_LIB=/paddle/nvidia-downloads/cuda-10.1/lib64
TENSORRT_ROOT=/paddle/nvidia-downloads/TensorRT-6.0.1.5

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT}

make -j
