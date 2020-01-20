
安装与编译Windows预测库
===========================

直接下载安装
-------------

| 版本说明      |     预测库(1.6.3版本)     |
|:---------|:-------------------|
|    cpu_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.3/win-infer/mkl/cpu/fluid_inference_install_dir.zip) |
|    cpu_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.3/win-infer/open/cpu/fluid_inference_install_dir.zip) |
|    cuda9.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.3/win-infer/mkl/post97/fluid_inference_install_dir.zip) |
|    cuda10.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.3/win-infer/mkl/post107/fluid_inference_install_dir.zip) |
|    cuda9.0_cudnn7_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.3/win-infer/open/post97/fluid_inference_install_dir.zip) |

从源码编译预测库
--------------
用户也可以从 PaddlePaddle 核心代码编译C++预测库，只需在编译时配制下面这些编译选项：

|选项                        |   值     |
|:-------------|:-------------------|
|CMAKE_BUILD_TYPE             | Release    |
|ON_INFER                     | ON (推荐)     |
|WITH_GPU                     | ON/OFF     | 
|WITH_MKL                     | ON/OFF     |
|WITH_PYTHON                  | OFF        |


请按照推荐值设置，以避免链接不必要的库。其它可选编译选项按需进行设定。

Windows下安装与编译预测库步骤：(在Windows命令提示符下执行以下指令)

1. 将PaddlePaddle的源码clone在当下目录的Paddle文件夹中，并进入Paddle目录：
   ```bash
   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   ```

2. 执行cmake：
   ```bash
   # 创建build目录用于编译
   mkdir build

   cd build

   cmake .. -G "Visual Studio 14 2015" -A x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=OFF -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF
   # -DWITH_GPU`为是否使用GPU的配置选项，-DWITH_MKL 为是否使用Intel MKL(数学核心库)的配置选项，请按需配置。

   # Windows默认使用 /MT 模式进行编译，如果想使用 /MD 模式，请使用以下命令。如不清楚两者的区别，请使用上面的命令
   cmake .. -G "Visual Studio 14 2015" -A x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=OFF -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF -DMSVC_STATIC_CRT=OFF
   ```

3. 使用Blend for Visual Studio 2015 打开 `paddle.sln` 文件，选择平台为`x64`，配置为`Release`，编译inference_lib_dist项目。
   操作方法：在Visual Studio中选择相应模块，右键选择"生成"（或者"build"）

编译成功后，使用C++预测库所需的依赖（包括：（1）编译出的PaddlePaddle预测库和头文件；（2）第三方链接库和头文件；（3）版本信息与编译选项信息）
均会存放于`fluid_inference_install_dir`目录中。目录结构如下：


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
     │       ├── xxhash
     │       └── zlib
     └── version.txt

version.txt 中记录了该预测库的版本信息，包括Git Commit ID、使用OpenBlas或MKL数学库、CUDA/CUDNN版本号，如：


     GIT COMMIT ID: cc9028b90ef50a825a722c55e5fda4b7cd26b0d6
     WITH_MKL: ON
     WITH_MKLDNN: ON
     WITH_GPU: ON
     CUDA version: 8.0
     CUDNN version: v7


编译预测demo
-------------

### 硬件环境

测试环境硬件配置：

| CPU      |      I7-8700K      |
|:---------|:-------------------|
| 内存 | 16G               |
| 硬盘 | 1T hdd + 256G ssd |
| 显卡 | GTX1080 8G        |

测试环境操作系统使用 win10 家庭版本。

### 软件要求

**请您严格按照以下步骤进行安装，否则可能会导致安装失败！**

**安装Visual Studio 2015 update3**

安装Visual Studio 2015，安装选项中选择安装内容时勾选自定义，选择安装全部关于c，c++，vc++的功能。


### 编译demo

下载并解压 fluid_inference_install_dir.zip 压缩包。

进入 Paddle/paddle/fluid/inference/api/demo_ci 目录，新建build目录并进入，然后使用cmake生成vs2015的solution文件。
指令为：

`cmake .. -G "Visual Studio 14 2015" -A x64 -DWITH_GPU=OFF -DWITH_MKL=ON -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=simple_on_word2vec -DPADDLE_LIB=path_to_the_paddle_lib`

注：

-DDEMO_NAME 是要编译的文件

-DPADDLE_LIB fluid_inference_install_dir，例如
-DPADDLE_LIB=D:\fluid_inference_install_dir


Cmake可以在[官网进行下载](https://cmake.org/download/)，并添加到环境变量中。

执行完毕后，build 目录如图所示，打开箭头指向的 solution 文件：

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image3.png">
</p>

根据实际的预测库版本选择`运行库`为 `/MT` 或 `/MD` ：

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image4.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image5.png">
</p>

编译生成选项改成 `Release` 。

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image6.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image7.png">
</p>


通过cmd进到Release目录执行：

  1.  开启GLOG

  	`set GLOG_v=100`

  2.  进行预测

  	`simple_on_word2vec.exe --dirname=.\word2vec.inference.model`

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image9.png">
</p>

