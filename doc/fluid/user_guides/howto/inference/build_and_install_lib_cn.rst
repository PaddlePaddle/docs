.. _install_or_build_cpp_inference_lib:

安装与编译C++预测库
===========================

直接下载安装
-------------

======================   ========================================
版本说明                            C++预测库   
======================   ========================================
cpu_avx_mkl              `fluid_inference.tgz <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxCp27cp27mu/.lastSuccessful/fluid_inference.tgz>`_ 
cpu_avx_openblas         `fluid_inference.tgz <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuAvxOpenblas/.lastSuccessful/fluid_inference.tgz>`_
cpu_noavx_openblas       `fluid_inference.tgz <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_CpuNoavxOpenblas/.lastSuccessful/fluid_inference.tgz>`_
cuda7.5_cudnn5_avx_mkl   `fluid_inference.tgz <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda75cudnn5cp27cp27mu/.lastSuccessful/fluid_inference.tgz>`_
cuda8.0_cudnn5_avx_mkl   `fluid_inference.tgz <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda80cudnn5cp27cp27mu/.lastSuccessful/fluid_inference.tgz>`_
cuda8.0_cudnn7_avx_mkl   `fluid_inference.tgz <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda8cudnn7cp27cp27mu/.lastSuccessful/fluid_inference.tgz>`_
cuda9.0_cudnn7_avx_mkl   `fluid_inference.tgz <https://guest:@paddleci.ngrok.io/repository/download/Manylinux1_Cuda90cudnn7avxMkl/.lastSuccessful/fluid_inference.tgz>`_
======================   ========================================

从源码编译
----------
用户也可以从 PaddlePaddle 核心代码编译C++预测库，只需在编译时配制下面这些编译选项：

============================  =========
选项                           值   
============================  =========
CMAKE_BUILD_TYPE              Release
FLUID_INFERENCE_INSTALL_DIR   安装路径    
WITH_FLUID_ONLY               ON（推荐）
WITH_SWIG_PY                  OFF（推荐）
WITH_PYTHON                   OFF（推荐）
ON_INFER                      ON（推荐）
WITH_GPU                      ON/OFF
WITH_MKL                      ON/OFF
============================  =========

建议按照推荐值设置，以避免链接不必要的库。其它可选编译选项按需进行设定。

下面的代码片段从github拉取最新代码，配制编译选项（需要将PADDLE_ROOT替换为PaddlePaddle预测库的安装路径）：

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

成功编译后，使用C++预测库所需的依赖（包括：（1）编译出的PaddlePaddle预测库和头文件；（2）第三方链接库和头文件；（3）版本信息与编译选项信息）
均会存放于PADDLE_ROOT目录中。目录结构如下：

  .. code-block:: text

     PaddleRoot/
     ├── CMakeCache.txt
     ├── paddle
     │   ├── include
     │   │   ├── paddle_anakin_config.h
     │   │   ├── paddle_analysis_config.h
     │   │   ├── paddle_api.h
     │   │   ├── paddle_inference_api.h
     │   │   ├── paddle_inference_pass.h
     │   │   └── paddle_pass_builder.h
     │   └── lib
     │       ├── libpaddle_fluid.a
     │       └── libpaddle_fluid.so
     ├── third_party
     │   ├── boost
     │   │   └── boost
     │   ├── eigen3
     │   │   ├── Eigen
     │   │   └── unsupported
     │   └── install
     │       ├── gflags
     │       ├── glog
     │       ├── mkldnn
     │       ├── mklml
     │       ├── protobuf
     │       ├── snappy
     │       ├── snappystream
     │       ├── xxhash
     │       └── zlib
     └── version.txt
     
version.txt 中记录了该预测库的版本信息，包括Git Commit ID、使用OpenBlas或MKL数学库、CUDA/CUDNN版本号，如：

  .. code-block:: text

     GIT COMMIT ID: cc9028b90ef50a825a722c55e5fda4b7cd26b0d6
     WITH_MKL: ON
     WITH_MKLDNN: ON
     WITH_GPU: ON
     CUDA version: 8.0
     CUDNN version: v5
