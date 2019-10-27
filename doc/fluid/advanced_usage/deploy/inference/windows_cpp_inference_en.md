
Install and Compile C++ Inference Library on Windows
===========================

Direct Download and Install
-------------

| Version      |     Inference Libraries(v1.6.0)   |
|:---------|:-------------------|
|    cpu_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.0/win-infer/mkl/cpu/fluid_inference_install_dir.zip) |
|    cpu_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.0/win-infer/open/cpu/fluid_inference_install_dir.zip) |
|    cuda9.0_cudnn7_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.0/win-infer/open/post97/fluid_inference_install_dir.zip) |
|    cuda9.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.0/win-infer/mkl/post97/fluid_inference_install_dir.zip) |
|    cuda10.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.0/win-infer/mkl/post107/fluid_inference_install_dir.zip) |

Build From Source Code
--------------

Users can also compile C++ inference libraries from the PaddlePaddle core code by specifying the following compile options at compile time:

|Option                        |   Value     |
|:-------------|:-------------------|
|CMAKE_BUILD_TYPE             | Release    |
|ON_INFER                     | ON(recommended)   |
|WITH_GPU                     | ON/OFF     | 
|WITH_MKL                     | ON/OFF     |
|WITH_PYTHON                  | OFF        |

**Paddle Windows Inference Library Compilation Steps**

1. Clone Paddle source code from GitHub:
   ```bash
   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   ```

2. Run Cmake command
   ```bash
   # create build directory
   mkdir build

   # change to the build directory
   cd build

   cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=OFF -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF
   # use -DWITH_GPU to control we are building CPU or GPU version
   # use -DWITH_MKL to select math library: Intel MKL or OpenBLAS

   # By default on Windows we use /MT for C Runtime Library, If you want to use /MD, please use the below command
   # If you have no ideas the differences between the two, use the above one
   cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=OFF -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF -DMSVC_STATIC_CRT=OFF
   ```

3. Open the `paddle.sln` using VisualStudio 2015，choose the`x64` for Solution Platforms，and `Release` for Solution Configurations，then build the `inference_lib_dist` project in the Solution Explorer(Rigth click the project and click Build)

The inference library will be installed in `fluid_inference_install_dir`:

     fluid_inference_install_dir/
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

version.txt constains the detailed configurations about the library，including git commit ID、math library, CUDA, CUDNN versions：


     GIT COMMIT ID: cc9028b90ef50a825a722c55e5fda4b7cd26b0d6
     WITH_MKL: ON
     WITH_MKLDNN: ON
     WITH_GPU: ON
     CUDA version: 8.0
     CUDNN version: v7


Inference Demo Compilation
-------------------

### Hardware Environment

Hardware Configuration of the experimental environment:

| CPU           |      I7-8700K      |
|:--------------|:-------------------|
| Memory        | 16G               |
| Hard Disk     | 1T hdd + 256G ssd |
| Graphics Card | GTX1080 8G        |

The operating system is win10 family version in the experimental environment.

### Steps to Configure Environment

**Please strictly follow the subsequent steps to install, otherwise the installation may fail**

**Install Visual Studio 2015 update3**

Install Visual Studio 2015. Please choose "customize" for the options of contents to be installed and choose to install all functions relevant to c, c++ and vc++.


Usage of Inference demo
------------------------

Decompress Paddle, Release and fluid_install_dir compressed package.

First enter into Paddle/paddle/fluid/inference/api/demo_ci, then create and enter into directory /build, finally use cmake to generate vs2015 solution file.
Commands are as follows:

`cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=simple_on_word2vec -DPADDLE_LIB=path_to_the_patddle\paddle_fluid.lib`

Note:

-DDEMO_NAME is the file to be built

-DPADDLE_LIB is the path of fluid_install_dir, for example:
-DPADDLE_LIB=D:\fluid_install_dir


Cmake can be [downloaded at official site](https://cmake.org/download/) and added to environment variables.

After the execution, the directory build is shown in the picture below. Then please open the solution file that which the arrow points at:

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image3.png">
</p>

Modify `Runtime Library` to `/MT`(default) or `/MD` according to the inference library version :

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/project_property.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/runtime_library.png">
</p>

Modify option of building and generating as `Release` .

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image6.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image7.png">
</p>

In the dependent packages provided, please copy openblas and model files under Release directory to the directory of Release built and generated.

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image8.png">
</p>

Enter into Release in cmd and run:

  1.  Open GLOG

  	`set GLOG_v=100`

  2.  Start inference

  	`simple_on_word2vec.exe --dirname=.\word2vec.inference.model`

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image9.png">
</p>
