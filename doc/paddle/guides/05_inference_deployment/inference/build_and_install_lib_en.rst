.. _install_or_build_cpp_inference_lib_en:

Install and Compile C++ Inference Library on Linux
====================================================

Direct Download and Installation
---------------------------------

..  csv-table:: c++ inference library list
    :header: "version description", "inference library(1.8.5 version)", "inference library(2.0.0-rc1 version)", "inference library(develop version)"
    :widths: 3, 2, 2, 2

    "ubuntu14.04_cpu_avx_mkl_gcc482", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-avx-mkl/fluid_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc1-cpu-avx-mkl/paddle_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-mkl/paddle_inference.tgz>`__"
    "ubuntu14.04_cpu_avx_openblas_gcc482", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-avx-openblas/fluid_inference.tgz>`__", ,"`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-avx-openblas/paddle_inference.tgz>`__"
    "ubuntu14.04_cpu_noavx_openblas_gcc482", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.8.5-cpu-noavx-openblas/fluid_inference.tgz>`__", ,"`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-cpu-noavx-openblas/paddle_inference.tgz>`__"
    "ubuntu14.04_cuda9.0_cudnn7_avx_mkl_gcc482", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.8.5-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc1-gpu-cuda9-cudnn7-avx-mkl/paddle_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-mkl/paddle_inference.tgz>`__"
    "ubuntu14.04_cuda9.0_cudnn7_avx_openblas_gcc482", , "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc1-gpu-cuda9-cudnn7-avx-openblas/paddle_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda9-cudnn7-avx-openblas/paddle_inference.tgz>`__"
    "ubuntu14.04_cuda10.0_cudnn7_avx_mkl_gcc482", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.8.5-gpu-cuda10-cudnn7-avx-mkl/fluid_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc1-gpu-cuda10-cudnn7-avx-mkl/paddle_inference.tgz>`__", "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/latest-gpu-cuda10-cudnn7-avx-mkl/paddle_inference.tgz>`__"
    "ubuntu14.04_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc482", "`fluid_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/1.8.5-gpu-cuda10.1-cudnn7.6-avx-mkl-trt6/fluid_inference.tgz>`__", ,
    "ubuntu14.04_cuda10.1_cudnn7.6_avx_mkl_trt6_gcc82", , "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc1-gpu-cuda10.1-cudnn7-avx-mkl-trt6/paddle_inference.tgz>`__",
    "ubuntu14.04_cuda10.2_cudnn8_avx_mkl_trt7_gcc82", , "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc1-gpu-cuda10.2-cudnn8-avx-mkl-trt7/paddle_inference.tgz>`__",
    "ubuntu14.04_cuda11_cudnn8_avx_mkl_trt7_gcc82", , "`paddle_inference.tgz <https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc1-gpu-cuda11-cudnn8-avx-mkl-trt7/paddle_inference.tgz>`__",
    "nv_jetson_cuda10_cudnn7.6_trt6_nano(jetpack4.3, for Nano)", , "`paddle_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc1-nv-jetson-jetpack4.3-nano/paddle_inference_install_dir.zip>`__",
    "nv_jetson_cuda10_cudnn7.6_trt6_tx2(jetpack4.3, for TX2 series)", , "`paddle_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc1-nv-jetson-jetpack4.3-tx2/paddle_inference_install_dir.zip>`__",
    "nv_jetson_cuda10_cudnn7.6_trt6_xavier(jetpack4.3, for AGX Xavier and Xavier NX)", , "`paddle_inference.tar.gz <https://paddle-inference-lib.bj.bcebos.com/2.0.0-rc1-nv-jetson-jetpack4.3-xavier/paddle_inference_install_dir.zip>`__",

Build from Source Code
-----------------------

Users can also compile C++ inference libraries from the PaddlePaddle core code by specifying the following compile options at compile time:

============================  ===============  ==================
Option                        Value            Description
============================  ===============  ==================
CMAKE_BUILD_TYPE              Release          cmake build type, set to Release if debug messages are not needed
FLUID_INFERENCE_INSTALL_DIR   path             install path of inference libs
WITH_PYTHON                   OFF(recomended)  build python libs and whl package
ON_INFER                      ON(recomended)   build with inference settings
WITH_GPU                      ON/OFF           build inference libs on GPU
WITH_MKL                      ON/OFF           build inference libs supporting MKL
WITH_MKLDNN                   ON/OFF           build inference libs supporting MKLDNN
WITH_XBYAK                    ON               build with XBYAK, must be OFF when building on NV Jetson platforms
WITH_NV_JETSON                OFF              build inference libs on NV Jetson platforms
WITH_TENSORRT                 OFF              build inference libs with NVIDIA TensorRT
============================  ===============  ==================

It is recommended to configure options according to the recommended values to avoid linking unnecessary libraries. Other options can be set if it is necessary.


Firstly we pull the latest code from github.

.. code-block:: bash

  git clone https://github.com/paddlepaddle/Paddle
  cd Paddle
  # Use git checkout to switch to stable versions such as v1.8.5
  git checkout v1.8.5


**note**: If your environment is a multi-card machine, it is recommended to install nccl; otherwise, you can skip this step by specifying WITH_NCCL = OFF during compilation. Note that if WITH_NCCL = ON, and NCCL is not installed, the compiler will report an error.

.. code-block:: bash

  git clone https://github.com/NVIDIA/nccl.git
  cd nccl
  make -j4
  make install


**build inference libs on server**

Following codes set the configurations and execute building(PADDLE_ROOT should be set to the actual installing path of inference libs, WITH_NCCL should be modified according to the actual environment.).

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
           -DWITH_NCCL=OFF \
           ..
      make
      make inference_lib_dist

**build inference libs on NVIDIA Jetson platforms**

NVIDIA Jetson is an AI computing platform in embedded systems introduced by NVIDIA. Paddle Inference supports building inference libs on NVIDIA Jetson platforms. The steps are as following.

    1. Prepare environments

      Turn on hardware performance mode

      .. code-block:: bash
        
        sudo nvpmodel -m 0 && sudo jetson_clocks

      if building on Nano hardwares, increase swap memory

      .. code-block:: bash

        # Increase DDR valid space. Default memory allocated is 16G, which is enough for Xavier. Following steps are for Nano hardwares.
        sudo fallocate -l 5G /var/swapfile
        sudo chmod 600 /var/swapfile
        sudo mkswap /var/swapfile
        sudo swapon /var/swapfile
        sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'

    2. Build paddle inference libs

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
        # Generate inference libs
        make inference_lib_dist -j4
      
    3. Test with samples
      Please refer to samples on https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/performance_improving/inference_improving/paddle_tensorrt_infer.html#id2

    **FAQ**

    1. Error:

      .. code-block:: bash

        ERROR: ../aarch64-linux-gpn/crtn.o: Too many open files.

      Fix this by increasing the number of files the system can open at the same time to 2048.

      .. code-block:: bash
        
        ulimit -n 2048

    2. The building process hangs.
      Might be downloading third-party libs. Wait or kill the building process and start again.

    3. Lacking virtual destructors for IPluginFactory or IGpuAllocator when using TensorRT.
      After downloading and installing TensorRT, add virtual destructors for IPluginFactory and IGpuAllocator in NvInfer.h:

      .. code-block:: bash
        
        virtual ~IPluginFactory() {};
        virtual ~IGpuAllocator() {};      


After successful compilation, dependencies required by the C++ inference library Will be stored in the PADDLE_ROOT directory. (dependencies including: (1) compiled PaddlePaddle inference library and header files; (2) third-party link libraries and header files; (3) version information and compilation option information)

The directory structure is:

  .. code-block:: text

     PaddleRoot/
     ├── CMakeCache.txt
     ├── paddle
     │   ├── include
     │   │   ├── paddle_anakin_config.h
     │   │   ├── paddle_analysis_config.h
     │   │   ├── paddle_api.h
     │   │   ├── paddle_inference_api.h
     │   │   ├── paddle_mkldnn_quantizer_config.h
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

The version information of the inference library is recorded in version.txt, including Git Commit ID, version of OpenBlas, MKL math library, or CUDA/CUDNN. For example:

  .. code-block:: text

     GIT COMMIT ID: cc9028b90ef50a825a722c55e5fda4b7cd26b0d6
     WITH_MKL: ON
     WITH_MKLDNN: ON
     WITH_GPU: ON
     CUDA version: 8.0
     CUDNN version: v7



