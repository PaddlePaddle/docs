
安装与编译Windows预测库
===========================

下载安装包与对应的测试环境
-------------

| 版本说明      |     预测库(1.6.2版本)     |       编译器        |    构建工具      |  cuDNN  |  CUDA  |
|:---------|:-------------------|:-------------------|:----------------|:--------|:-------|
|    cpu_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.2/win-infer/mkl/cpu/fluid_inference_install_dir.zip) | MSVC 2015 update 3|  CMake v3.11.1  |
|    cpu_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.2/win-infer/open/cpu/fluid_inference_install_dir.zip) | MSVC 2015 update 3|  CMake v3.11.1  |
|    cuda9.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.2/win-infer/mkl/post97/fluid_inference_install_dir.zip) |  MSVC 2015 update 3 |  CMake v3.11.1  |  7.3.1  |   9    |
|    cuda10.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.2/win-infer/mkl/post107/fluid_inference_install_dir.zip) | MSVC 2015 update 3 |  CMake v3.11.1  |  7.4.1  |   10    |
|    cuda9.0_cudnn7_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.2/win-infer/open/post97/fluid_inference_install_dir.zip) | MSVC 2015 update 3 |  CMake v3.11.1  |  7.3.1  |   9    |

### 硬件环境

测试环境硬件配置：

| CPU      |      I7-8700K      |
|:---------|:-------------------|
| 内存 | 16G               |
| 硬盘 | 1T hdd + 256G ssd |
| 显卡 | GTX1080 8G        |

测试环境操作系统使用 win10 家庭版本

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

   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=OFF -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF
   # -DWITH_GPU`为是否使用GPU的配置选项，-DWITH_MKL 为是否使用Intel MKL(数学核心库)的配置选项，请按需配置。

   # Windows默认使用 /MT 模式进行编译，如果想使用 /MD 模式，请使用以下命令。如不清楚两者的区别，请使用上面的命令
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=OFF -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF -DMSVC_STATIC_CRT=OFF
   ```

3. 使用Blend for Visual Studio 2015 打开 `paddle.sln` 文件，选择平台为`x64`，配置为`Release`，编译inference_lib_dist项目。
   操作方法：在Visual Studio中选择相应模块，右键选择"生成"（或者"build"）

编译成功后，使用C++预测库所需的依赖（包括：（1）编译出的PaddlePaddle预测库和头文件；（2）第三方链接库和头文件；（3）版本信息与编译选项信息）
均会存放于`fluid_inference_install_dir`目录中。

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

### 其他要求

1. 你需要直接下载Windows预测库或者从Paddle源码编译预测库，确保windows预测库存在。

2. 你需要下载Paddle源码，确保demo文件和脚本文件存在：
```bash
git clone https://github.com/PaddlePaddle/Paddle.git
``` 
### 编译demo

#### 使用脚本编译运行
 
打开cmd窗口，使用下面的命令:
```dos
# path为下载Paddle的目录
cd path\Paddle\paddle\fluid\inference\api\demo_ci 
run_windows_demo.bat
```

其中，run_windows_demo.bat 的部分选项如下，请根据提示按需输入参数：

```dos
gpu_inference=Y #是否使用GPU预测库，默认使用CPU预测库
use_mkl=Y #该预测库是否使用MKL，默认为Y
use_gpu=Y  #是否使用GPU进行预测，默认为N。使用GPU预测需要下载GPU版本预测库

paddle_inference_lib=path\fluid_inference_install_dir #设置paddle预测库的路径
cuda_lib_dir=path\lib\x64  #设置cuda库的路径
vcvarsall_dir=path\vc\vcvarsall.bat  #设置visual studio #本机工具命令提示符路径
```
#### 手动编译运行
 
打开cmd窗口，使用下面的命令:
```dos
# path为下载Paddle的目录
cd path\Paddle\paddle\fluid\inference\api\demo_ci
mkdir build
cd build
```
`cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DWITH_GPU=OFF -DWITH_MKL=ON -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=simple_on_word2vec -DPADDLE_LIB=path_to_the_paddle_lib -DMSVC_STATIC_CRT=ON`

注：

-DDEMO_NAME 是要编译的文件

-DPADDLE_LIB fluid_inference_install_dir，例如
-DPADDLE_LIB=D:\fluid_inference_install_dir


Cmake可以在[官网进行下载](https://cmake.org/download/)，并添加到环境变量中。

执行完毕后，build 目录如图所示，打开箭头指向的 solution 文件：

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image3.png">
</p>

编译生成选项改成 `Release` 。

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image6.png">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image7.png">
</p>

[下载模型](http://paddle-inference-dist.bj.bcebos.com/word2vec.inference.model.tar.gz)并解压到当前目录，执行命令：

  1. 开启GLOG

     `set GLOG_v=100`

  2. 进行预测，path为模型解压后的目录

     `Release\simple_on_word2vec.exe --dirname=path\word2vec.inference.model`

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image9.png">
</p>

## 使用AnalysisConfig管理预测配置

[完整的代码示例](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/api/demo_ci/windows_mobilenet.cc)

本示例使用了AnalysisConfig管理AnalysisPredictor的预测配置，提供了模型路径设置、预测引擎运行设备选择以及使用ZeroCopyTensor管理输入/输出的设置。配置方法如下：

#### 创建AnalysisConfig
```C++
AnalysisConfig config;
```
**Note:** 使用ZeroCopyTensor，务必在创建config时设置`config->SwitchUseFeedFetchOps(false);`。
```C++
config->SwitchUseFeedFetchOps(false);  // 关闭feed和fetch OP使用，使用ZeroCopy接口必须设置此项
config->EnableUseGpu(100 /*设定GPU初始显存池为MB*/,  0 /*设定GPU ID为0*/); //开启GPU预测
```

#### 设置模型和参数路径
从磁盘加载模型时，根据模型和参数文件存储方式不同，设置AnalysisConfig加载模型和参数的路径有两种形式，此处使用combined形式：

* combined形式：模型文件夹`model_dir`下只有一个模型文件`__model__`和一个参数文件`__params__`时，传入模型文件和参数文件路径。
```C++
config->SetModel("./model_dir/__model__", "./model_dir/__params__");
```

#### 使用ZeroCopyTensor管理输入
ZeroCopyTensor是AnalysisPredictor的输入/输出数据结构

**Note:** 使用ZeroCopyTensor，务必在创建config时设置`config->SwitchUseFeedFetchOps(false);`。

```C++
// 通过创建的AnalysisPredictor获取输入tensor
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputTensor(input_names[0]);

// 对tensor进行reshape，channels，height，width三个参数的设置必须与模型中输入所要求的一致
input_t->Reshape({batch_size, channels, height, width});
```

#### 运行预测引擎
```C++
predictor->ZeroCopyRun();
```

#### 使用ZeroCopyTensor管理输出
```C++
auto output_names = predictor->GetOutputNames();
auto output_t = predictor->GetOutputTensor(output_names[0]);
```
**Note:** 关于AnalysisPredictor的更多介绍，请参考[C++预测API介绍](./native_infer.html)
