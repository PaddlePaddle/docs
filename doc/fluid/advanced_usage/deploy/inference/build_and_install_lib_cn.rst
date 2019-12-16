.. _install_or_build_cpp_inference_lib:

安装与编译C++预测库
===========================

直接下载安装
-------------

..  csv-table:: 
    :header: "版本说明", "预测库(1.6.2版本)", "预测库(develop版本)"
    :widths: 3, 2, 2

    "ubuntu14.04_cpu_avx_mkl", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.2-cpu-avx-mkl/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-mkl/fluid_inference.tgz>`_"
    "ubuntu14.04_cpu_avx_openblas", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.2-cpu-avx-openblas/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-openblas/fluid_inference.tgz>`_"
    "ubuntu14.04_cpu_noavx_openblas", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.2-cpu-noavx-openblas/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-noavx-openblas/fluid_inference.tgz>`_"
    "ubuntu14.04_cuda9.0_cudnn7_avx_mkl", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.2-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz>`_"
    "ubuntu14.04_cuda10.0_cudnn7_avx_mkl", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.2-gpu-cuda10-cudnn7-avx-mkl/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda10-cudnn7-avx-mkl/fluid_inference.tgz>`_"
    "ubuntu14.04_cuda8.0_cudnn7_avx_mkl_trt4", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.2-gpu-cuda8-cudnn7-avx-mkl-trt4/fluid_inference.tgz>`_", 
    "ubuntu14.04_cuda9.0_cudnn7_avx_mkl_trt5", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.2-gpu-cuda9-cudnn7-avx-mkl-trt5/fluid_inference.tgz>`_", 
    "ubuntu14.04_cuda10.0_cudnn7_avx_mkl_trt5", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.2-gpu-cuda10-cudnn7-avx-mkl-trt5/fluid_inference.tgz>`_", 
    "nv-jetson-cuda10-cudnn7.5-trt5", "`fluid_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/1.6.2-nv-jetson-cuda10-cudnn7.5-trt5/fluid_inference.tar.gz>`_", 

**Note:所提供的C++预测库均使用GCC 4.8编译。**

从源码编译.. _install_or_build_cpp_inference_lib:

安装与编译C++预测库
===========================

直接下载安装
-------------

..  csv-table:: 
    :header: "版本说明", "预测库(1.6.1版本)", "预测库(develop版本)"
    :widths: 3, 2, 2

    "ubuntu14.04_cpu_avx_mkl", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.1-cpu-avx-mkl/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-mkl/fluid_inference.tgz>`_"
    "ubuntu14.04_cpu_avx_openblas", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.1-cpu-avx-openblas/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-openblas/fluid_inference.tgz>`_"
    "ubuntu14.04_cpu_noavx_openblas", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.1-cpu-noavx-openblas/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-noavx-openblas/fluid_inference.tgz>`_"
    "ubuntu14.04_cuda9.0_cudnn7_avx_mkl", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.1-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz>`_"
    "ubuntu14.04_cuda10.0_cudnn7_avx_mkl", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.1-gpu-cuda10-cudnn7-avx-mkl/fluid_inference.tgz>`_", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda10-cudnn7-avx-mkl/fluid_inference.tgz>`_"
    "ubuntu14.04_cuda8.0_cudnn7_avx_mkl_trt4", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.1-gpu-cuda8-cudnn7-avx-mkl-trt4/fluid_inference.tgz>`_", 
    "ubuntu14.04_cuda9.0_cudnn7_avx_mkl_trt5", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.1-gpu-cuda9-cudnn7-avx-mkl-trt5/fluid_inference.tgz>`_", 
    "ubuntu14.04_cuda10.0_cudnn7_avx_mkl_trt5", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.6.1-gpu-cuda10-cudnn7-avx-mkl-trt5/fluid_inference.tgz>`_", 

**Note:所提供的C++预测库均使用GCC 4.8编译。**

从源码编译
----------
用户也可以从 PaddlePaddle 核心代码编译C++预测库，只需在编译时配制下面这些编译选项：

============================  =============
选项                           值
============================  =============
CMAKE_BUILD_TYPE              Release
FLUID_INFERENCE_INSTALL_DIR   安装路径
WITH_PYTHON                   OFF（推荐）
ON_INFER                      ON（推荐）
WITH_GPU                      ON/OFF
WITH_MKL                      ON/OFF
============================  =============

建议按照推荐值设置，以避免链接不必要的库。其它可选编译选项按需进行设定。

1. **Server端预测库源码编译**

下面的代码片段从github拉取最新代码，配制编译选项（需要将PADDLE_ROOT替换为PaddlePaddle预测库的安装路径）：

  .. code-block:: bash

     PADDLE_ROOT=/path/of/capi
     git clone https://github.com/PaddlePaddle/Paddle.git
     cd Paddle
     mkdir build
     cd build
     cmake -DFLUID_INFERENCE_INSTALL_DIR=$PADDLE_ROOT \
           -DCMAKE_BUILD_TYPE=Release \
           -DWITH_PYTHON=OFF \
           -DWITH_MKL=OFF \
           -DWITH_GPU=OFF  \
           -DON_INFER=ON \
           ..
      make
      make inference_lib_dist

2. **NVIDIA Jetson嵌入式硬件预测库源码编译**

NVIDIA Jetson是NVIDIA推出的嵌入式AI平台，Paddle Inference支持在 NVIDIA Jetson平台上编译预测库。具体步骤如下：

    i) 从github拉取paddle代码

      .. code-block:: bash
    
        git clone https://github.com/paddlepaddle/paddle
        # 切换到1.6.2稳定版本
        git checkout v1.6.2
    ii) 准备环境
      安装nccl

      .. code-block:: bash
        
        git clone https://github.com/NVIDIA/nccl.git
        make -j4
        make install
      **note**： 单卡机器上不会用到nccl但仍存在依赖， 后续会考虑将此依赖去除。
      开启硬件性能模式
      .. code-block:: bash
        
        sudo nvpmodel -m 0 && sudo jetson_clocks

      如果硬件为Nano，增加swap空间

      .. code-block:: bash

        #增加DDR可用空间，Xavier默认内存为16G，所以内存足够，如想在Nano上尝试，请执行如下操作。
        sudo fallocate -l 5G /var/swapfile
        sudo chmod 600 /var/swapfile
        sudo mkswap /var/swapfile
        sudo swapon /var/swapfile
        sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'

    iii) 编译Paddle Inference预测库
      .. code-block:: bash
 
        cd Paddle
        mkdir build
        cmake .. \
          -DWITH_CONTRIB=OFF \
          -DWITH_MKL=OFF  \
          -DWITH_MKLDNN=OFF \
          -DWITH_TESTING=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          -DON_INFER=ON \
          -DWITH_PYTHON=OFF \
          -DWITH_XBYAK=OFF  \
          -DWITH_NV_JETSON=ON 
        make -j4       
        # 生成预测lib
        make inference_lib_dist -j4

    iiii) 样例测试
      请参照官网样例：https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_usage/deploy/inference/paddle_tensorrt_infer.html#id2
    
    **FAQ**

    i) 报错：

      .. code-block:: bash

        ERROR: ../aarch64-linux-gpn/crtn.o: Too many open files.

      则增加系统同一时间最多可开启的文件数至2048

      .. code-block:: bash
        
        ulimit -n 2048

    ii) 编译卡住
      可能是下载第三方库较慢的原因，耐心等待或kill掉编译进程重新编译
     

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
     │   │   ├── paddle_mkldnn_quantizer_config.h
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
     CUDNN version: v7

----------
用户也可以从 PaddlePaddle 核心代码编译C++预测库，只需在编译时配制下面这些编译选项：

============================  =============
选项                           值
============================  =============
CMAKE_BUILD_TYPE              Release
FLUID_INFERENCE_INSTALL_DIR   安装路径
WITH_PYTHON                   OFF（推荐）
ON_INFER                      ON（推荐）
WITH_GPU                      ON/OFF
WITH_MKL                      ON/OFF
============================  =============

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
     │   │   ├── paddle_mkldnn_quantizer_config.h
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
     CUDNN version: v7
