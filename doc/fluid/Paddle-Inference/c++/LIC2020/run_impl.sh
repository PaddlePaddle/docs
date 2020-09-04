work_path=$(dirname $(readlink -f $0))

mkdir -p build
cd build
rm -rf *

# same with the demo.cc
DEMO_NAME=demo

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

LIB_DIR=${work_path}/fluid_inference_install_dir
CUDNN_LIB=/usr/local/cudnn/lib64
CUDA_LIB=/usr/local/cuda/lib64
# TENSORRT_ROOT=/usr/local/TensorRT-6.0.1.5

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
