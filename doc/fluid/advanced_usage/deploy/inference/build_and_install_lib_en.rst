.. _install_or_build_cpp_inference_lib:

Install and Compile C++ Prediction Libraries
=============================================

Direct Download and Installation
---------------------------------

..  csv-table:: c++ inference library list
    :header: "version desciption", "predict library(1.2 version)", "predict library(develop version)"
    :widths: 1, 3, 3

    "cpu_avx_mkl", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.2.0-cpu-avx-mkl/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-mkl/fluid_inference.tgz>`_"
    "cpu_avx_openblas", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.2.0-cpu-avx-openblas/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-openblas/fluid_inference.tgz>`_"
    "cpu_noavx_openblas","`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.2.0-cpu-noavx-openblas/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-noavx-openblas/fluid_inference.tgz>`_"
    "cuda8.0_cudnn5_avx_mkl", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn5-avx-mkl/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda8-cudnn5-avx-mkl/fluid_inference.tgz>`_"
    "cuda8.0_cudnn7_avx_mkl","`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.2.0-gpu-cuda8-cudnn7-avx-mkl/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda8-cudnn7-avx-mkl/fluid_inference.tgz>`_"
    "cuda9.0_cudnn7_avx_mkl", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.2.0-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz>`_"


Build Source Code
---------------------
Users can also compile C++ predictive libraries from the PaddlePaddle core code by compiling the following compile options at compile time:
============================  =========
Option                        Value
============================  =========
CMAKE_BUILD_TYPE              Release
FLUID_INFERENCE_INSTALL_DIR   Path of installation
WITH_FLUID_ONLY               ON（recommend）
WITH_SWIG_PY                  OFF（recommend）
WITH_PYTHON                   OFF（recommend）
ON_INFER                      ON（recommend）
WITH_GPU                      ON/OFF
WITH_MKL                      ON/OFF
============================  =========

It is recommended to configure options according to recommended value to avoid the link to unecessary library. Other options can be set if it is necessary.


The following code snippet pulls the latest code from github and compiles the build options (you need to replace PADDLE_ROOT with the installation path of the PaddlePaddle prediction library):

  .. code-block:: bash

     PADDLE_ROOT=/path/of/capi
     git clone https://github.com/PaddlePaddle/Paddle.git
     cd Paddle
     mkdir build
     cd build
     cmake -DFLUID_INFERENCE_INSTALL_DIR=$PADDLE_ROOT \
           -DCMAKE_BUILD_TYPE=Release \
           -DWITH_FLUID_ONLY=ON \
           -DWITH_SWIG_PY=OFF \
           -DWITH_PYTHON=OFF \
           -DWITH_MKL=OFF \
           -DWITH_GPU=OFF  \
           -DON_INFER=ON \
           ..
      make
      make inference_lib_dist

After successful compilation, use C++ to predict the dependencies required by the library (including: (1) compiled PaddlePaddle prediction library and head files; (2) third-party link libraries and head files; (3) version information and compilation option information) Will be stored in the PADDLE_ROOT directory. The directory structure is as follows:

  .. code-block:: text

     PaddleRoot/
     ├── CMakeCache.txt
     ├── paddle
     │   ├── include
     │   │   ├── paddle_anakin_config.h
     │   │   ├── paddle_analysis_config.h
     │   │   ├── paddle_api.h
     │   │   ├── paddle_inference_api.h
     │   │   └── paddle_pass_builder.h
     │   └── lib
     │       ├── libpaddle_fluid.a
     │       └── libpaddle_fluid.so
     ├── third_party
     │   ├── boost
     │   │   └── boost
     │   ├── eigen3
     │   │   ├── Eigen
     │   │   └── unsupported
     │   └── install
     │       ├── gflags
     │       ├── glog
     │       ├── mkldnn
     │       ├── mklml
     │       ├── protobuf
     │       ├── snappy
     │       ├── snappystream
     │       ├── xxhash
     │       └── zlib
     └── version.txt

Version information about the predict library has been recorded in version.txt, including Git Commit ID, usage of OpenBlas or MKL math library, CUDA/CUDNN version number, for example:

  .. code-block:: text

     GIT COMMIT ID: cc9028b90ef50a825a722c55e5fda4b7cd26b0d6
     WITH_MKL: ON
     WITH_MKLDNN: ON
     WITH_GPU: ON
     CUDA version: 8.0
     CUDNN version: v5
