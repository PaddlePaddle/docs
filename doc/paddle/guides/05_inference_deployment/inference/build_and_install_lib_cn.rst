.. _install_or_build_cpp_inference_lib:

安装与编译 Linux 预测库
===========================

直接下载安装
-------------

..  csv-table:: 
    :header: "版本说明", "预测库(1.8.5版本)", "预测库(2.0.1版本)", "预测库(develop版本)"
    :widths: 3, 2, 2, 2

    "ubuntu14.04_cpu_avx_mkl_gcc82", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-avx-mkl/fluid_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-cpu-avx-mkl/paddle_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-mkl/paddle_inference.tgz>`__"
    "ubuntu14.04_cpu_avx_openblas_gcc82", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-avx-openblas/fluid_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-cpu-avx-openblas/paddle_inference.tgz>`__","`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-openblas/paddle_inference.tgz>`__"
    "ubuntu14.04_cpu_noavx_openblas_gcc82", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-noavx-openblas/fluid_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-cpu-noavx-openblas/paddle_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-noavx-openblas/paddle_inference.tgz>`__"
    "ubuntu14.04_cuda9.0_cudnn7_avx_mkl_gcc482", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.8.5-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-gpu-cuda9-cudnn7-avx-mkl/paddle_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddle_inference.tgz>`__"    
    "ubuntu14.04_cuda10.0_cudnn7_avx_mkl_gcc482", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.8.5-gpu-cuda10-cudnn7-avx-mkl/fluid_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-gpu-cuda10-cudnn7-avx-mkl/paddle_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda10-cudnn7-avx-mkl/paddle_inference.tgz>`__"
    "ubuntu14.04_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc82", , "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-gpu-cuda10.1-cudnn7-avx-mkl/paddle_inference.tgz>`__",
    "ubuntu14.04_cuda10.2_cudnn8_avx_mkl_trt7_gcc82", , "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-gpu-cuda10.2-cudnn8-avx-mkl/paddle_inference.tgz>`__",
    "ubuntu14.04_cuda11_cudnn8_avx_mkl_trt7_gcc82", , "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-gpu-cuda11-cudnn8-avx-mkl/paddle_inference.tgz>`__",
    "nv_jetson_cuda10_cudnn7.6_trt6_all(jetpack4.3)", , "`paddle_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.3-all/paddle_inference.tgz>`__",
    "nv_jetson_cuda10_cudnn7.6_trt6_nano(jetpack4.3)", , "`paddle_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.3-nano/paddle_inference.tgz>`__",
    "nv_jetson_cuda10_cudnn7.6_trt6_tx2(jetpack4.3)", , "`paddle_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.3-tx2/paddle_inference.tgz>`__",
    "nv_jetson_cuda10_cudnn7.6_trt6_xavier(jetpack4.3)", , "`paddle_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.3-xavier/paddle_inference_install_dir.tgz>`__",
    "nv_jetson_cuda10.2_cudnn8_trt7_all(jetpack4.4)", , "`paddle_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.4-all/paddle_inference.tgz>`__",
    "nv_jetson_cuda10.2_cudnn8_trt7_nano(jetpack4.4)", , "`paddle_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.4-nano/paddle_inference.tgz>`__",
    "nv_jetson_cuda10.2_cudnn8_trt7_tx2(jetpack4.4)", , "`paddle_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.4-tx2/paddle_inference.tgz>`__",
    "nv_jetson_cuda10.2_cudnn8_trt7_xavier(jetpack4.4)", , "`paddle_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/2.0.1-nv-jetson-jetpack4.4-xavier/paddle_inference.tgz>`__",


从源码编译
----------
用户也可以从 PaddlePaddle 核心代码编译C++预测库，只需在编译时配制下面这些编译选项：

============================  =============  ==================
选项                           值             说明
============================  =============  ==================
CMAKE_BUILD_TYPE              Release        编译方式，仅使用预测库设为Release即可
FLUID_INFERENCE_INSTALL_DIR   安装路径         预测库安装路径
WITH_PYTHON                   OFF(推荐)       编译python预测库与whl包
ON_INFER                      ON(推荐)        预测时使用，必须设为ON
WITH_GPU                      ON/OFF         编译支持GPU的预测库
WITH_MKL                      ON/OFF         编译支持MKL的预测库
WITH_MKLDNN                   ON/OFF         编译支持MKLDNN的预测库
WITH_XBYAK                    ON             使用XBYAK编译，在jetson硬件上编译需要设置为OFF
WITH_TENSORRT                 OFF            编译支持NVIDIA TensorRT的预测库，需要另外配置TENSORRT_ROOT选项指定TRT根目录
============================  =============  ==================

建议按照推荐值设置，以避免链接不必要的库。其它可选编译选项按需进行设定。

首先从github拉取最新代码

.. code-block:: bash

  git clone https://github.com/paddlepaddle/Paddle
  cd Paddle
  # 建议使用git checkout切换到Paddle稳定的版本，如：
  git checkout release/2.0

**note**: 如果您是多卡机器，建议安装NCCL；如果您是单卡机器则可以在编译时显示指定WITH_NCCL=OFF来跳过这一步。注意如果WITH_NCCL=ON，且没有安装NCCL，则编译会报错。

.. code-block:: bash

  git clone https://github.com/NVIDIA/nccl.git
  cd nccl
  make -j4
  make install


**Server端预测库源码编译**

下面的代码片段配制编译选项并进行编译（需要将PADDLE_ROOT替换为PaddlePaddle预测库的安装路径，WITH_NCCL根据实际情况进行修改）：

  .. code-block:: bash

     PADDLE_ROOT=/path/of/paddle
     cd Paddle
     mkdir build
     cd build
     cmake -DFLUID_INFERENCE_INSTALL_DIR=$PADDLE_ROOT \
           -DCMAKE_BUILD_TYPE=Release \
           -DWITH_PYTHON=OFF \
           -DWITH_MKL=OFF \
           -DWITH_GPU=OFF  \
           -DON_INFER=ON \
           -DWITH_NCCL=OFF \
           ..
      make
      make inference_lib_dist

**NVIDIA Jetson嵌入式硬件预测库源码编译**

NVIDIA Jetson是NVIDIA推出的嵌入式AI平台，Paddle Inference支持在 NVIDIA Jetson平台上编译预测库。具体步骤如下：

    1. 准备环境

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

    2. 编译Paddle Inference预测库
      .. code-block:: bash
 
        cd Paddle
        mkdir build
        cd build
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

    3. 样例测试
      请参照官网样例：https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/performance_improving/inference_improving/paddle_tensorrt_infer.html#id2
    
    **FAQ**

    1. 报错：

      .. code-block:: bash

        ERROR: ../aarch64-linux-gpn/crtn.o: Too many open files.

      则增加系统同一时间最多可开启的文件数至2048

      .. code-block:: bash
        
        ulimit -n 2048

    2. 编译卡住
      可能是下载第三方库较慢的原因，耐心等待或kill掉编译进程重新编译

    3. 使用TensorRT报错IPluginFactory或IGpuAllocator缺少虚析构函数
      下载安装TensorRT后，在NvInfer.h文件中为class IPluginFactory和class IGpuAllocator分别添加虚析构函数：

      .. code-block:: bash
        
        virtual ~IPluginFactory() {};
        virtual ~IGpuAllocator() {};
     

成功编译后，使用C++预测库所需的依赖（包括:（1）编译出的PaddlePaddle预测库和头文件；（2）第三方链接库和头文件；（3）版本信息与编译选项信息）
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
     │   └── install
     │       ├── gflags
     │       ├── glog
     │       ├── mkldnn
     │       ├── mklml
     │       └── protobuf
     └── version.txt

version.txt 中记录了该预测库的版本信息，包括Git Commit ID、使用OpenBlas或MKL数学库、CUDA/CUDNN版本号，如：

  .. code-block:: text

     GIT COMMIT ID: 0231f58e592ad9f673ac1832d8c495c8ed65d24f
     WITH_MKL: ON
     WITH_MKLDNN: ON
     WITH_GPU: ON
     CUDA version: 10.1
     CUDNN version: v7




