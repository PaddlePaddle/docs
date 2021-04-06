
安装与编译 Windows 预测库
===========================

直接下载安装
-------------

| 版本说明      |     预测库(1.8.4版本)  |预测库(2.0.0)   |     编译器     |    构建工具      |  cuDNN  |  CUDA  |
|:---------|:-------------------|:-------------------|:----------------|:--------|:-------|:-------|
|    cpu_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/cpu/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/mkl/cpu/paddle_inference_install_dir.zip)| MSVC 2015 update 3|  CMake v3.17.0  | - | - |
 |    cpu_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/open/cpu/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/open/cpu/paddle_inference_install_dir.zip)| MSVC 2015 update 3|  CMake v3.17.0  | - | - |
 |    cuda9.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/post97/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/mkl/post100/paddle_inference_install_dir.zip)  | MSVC 2015 update 3 |  CMake v3.17.0  |  7.3.1  |   9.0    |
|    cuda9.0_cudnn7_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/open/post97/fluid_inference_install_dir.zip) | - | MSVC 2015 update 3 |  CMake v3.17.0  |  7.3.1  |   9.0    |
|    cuda10.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.4/win-infer/mkl/post107/fluid_inference_install_dir.zip) | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/mkl/post100/paddle_inference_install_dir.zip) | MSVC 2015 update 3 |  CMake v3.17.0  |  7.4.1  |   10.0    |
 |    cuda10.0_cudnn7_avx_mkl_trt6 | | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/trt_mkl/post100/paddle_inference.zip) | MSVC 2015 update 3 |  CMake v3.17.0  |  7.4.1  |   10.0    |
 |    cuda10.1_cudnn7_avx_mkl_trt6 | | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/trt_mkl/post101/paddle_inference.zip) | MSVC 2015 update 3 |  CMake v3.17.0  |  7.6  |   10.1    |
 |    cuda10.2_cudnn7_avx_mkl_trt7 | | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/trt_mkl/post102/paddle_inference.zip)| MSVC 2015 update 3 |  CMake v3.17.0  |  7.6  |   10.2    |
 |    cuda11.0_cudnn8_avx_mkl_trt7 | | [paddle_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.1/win-infer/trt_mkl/post11/paddle_inference.zip)| MSVC 2015 update 3 |  CMake v3.17.0  |  8.0  |   11.0    |

### 硬件环境

测试环境硬件配置：

| 操作系统      |    win10 家庭版本      |
|:---------|:-------------------|
| CPU      |      I7-8700K      |
| 内存 | 16G               |
| 硬盘 | 1T hdd + 256G ssd |
| 显卡 | GTX1080 8G        |


从源码编译
--------------
用户也可以从 PaddlePaddle 核心代码编译C++预测库，只需在编译时配制下面这些编译选项：

|选项       |说明               |   值     |
|:-------------|:-------|:------------|
|CMAKE_BUILD_TYPE |  配置生成器上的构建类型，windows预测库目前只支持Release          | Release    |
|ON_INFER |   是否生成预测库，编译预测库时必须设置为ON                | ON         |
|WITH_GPU |   是否支持GPU                  | ON/OFF     |
|WITH_MKL |   是否使用Intel MKL(数学核心库)或者OPENBLAS     | ON/OFF     |
|WITH_PYTHON | 是否编译Python包                | OFF(推荐)        |
|MSVC_STATIC_CRT|是否使用/MT 模式进行编译，默认使用 /MT 模式进行编译 |ON/OFF|
|CUDA_TOOLKIT_ROOT_DIR|编译GPU预测库时，需设置CUDA的根目录|YOUR_CUDA_PATH|

请按照推荐值设置，以避免链接不必要的库。其它可选编译选项按需进行设定。

更多具体编译选项含义请参见[编译选项表](../../../beginners_guide/install/Tables.html/#Compile)

Windows下安装与编译预测库步骤：(在Windows命令提示符下执行以下指令)

1. 将PaddlePaddle的源码clone在当下目录的Paddle文件夹中，并进入Paddle目录，创建build目录：
   ```bash
   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   # 创建并进入build目录
   mkdir build
   cd build
   ```

2. 执行cmake：
   - 编译CPU预测库
   ```bash
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=ON -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF

   # Windows默认使用 /MT 模式进行编译，如果想使用 /MD 模式，请使用以下命令。如不清楚两者的区别，请使用上面的命令
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=ON -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF -DMSVC_STATIC_CRT=OFF
   ```
   - 编译GPU预测库:
   ```bash
   # -DCUDA_TOOLKIT_ROOT_DIR为你所安装的cuda根目录，例如-DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0"
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=ON -DWITH_GPU=ON -DON_INFER=ON -DWITH_PYTHON=OFF -DCUDA_TOOLKIT_ROOT_DIR=YOUR_CUDA_PATH
   ```

3. 使用Blend for Visual Studio 2015 打开 `paddle.sln` 文件，选择平台为`x64`，配置为`Release`，编译inference_lib_dist项目。
   操作方法：在Visual Studio中选择相应模块，右键选择"生成"（或者"build"）

编译成功后，使用C++预测库所需的依赖（包括：1. 编译出的PaddlePaddle预测库和头文件；2. 第三方链接库和头文件；3. 版本信息与编译选项信息）均会存放于`fluid_inference_install_dir`目录中。

version.txt 中记录了该预测库的版本信息，包括Git Commit ID、使用OpenBlas或MKL数学库、CUDA/CUDNN版本号、C++编译器版本，如：

```text
GIT COMMIT ID: 264e76cae6861ad9b1d4bcd8c3212f7a78c01e4d
WITH_MKL: ON
WITH_MKLDNN: ON
WITH_GPU: ON
CUDA version: 10.0
CUDNN version: v7.4
CXX compiler version: 19.0.24215.1
```

编译预测demo
-------------

### 硬件环境

测试环境硬件配置：

| 操作系统      |    win10 家庭版本      |
|:---------|:-------------------|
| CPU      |      I7-8700K      |
| 内存 | 16G               |
| 硬盘 | 1T hdd + 256G ssd |
| 显卡 | GTX1080 8G        |

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
Windows下编译预测demo步骤：(在Windows命令提示符下执行以下指令)
#### 使用脚本编译运行

进入到demo_ci目录，运行脚本`run_windows_demo.bat`，根据提示按需输入参数:
```dos
# path为下载Paddle的目录
cd path\Paddle\paddle\fluid\inference\api\demo_ci
run_windows_demo.bat
```

其中，run_windows_demo.bat 的部分选项如下：

```dos
gpu_inference=Y #是否使用GPU预测库，默认使用CPU预测库
use_mkl=Y #该预测库是否使用MKL，默认为Y
use_gpu=Y  #是否使用GPU进行预测，默认为N。使用GPU预测需要下载GPU版本预测库

paddle_inference_lib=path\fluid_inference_install_dir #设置paddle预测库的路径
cuda_lib_dir=path\lib\x64  #设置cuda库的路径
vcvarsall_dir=path\vc\vcvarsall.bat  #设置visual studio #本机工具命令提示符路径
```
#### 手动编译运行

1. 进入demo_ci目录，创建并进入build目录
   ```dos
   # path为下载Paddle的目录
   cd path\Paddle\paddle\fluid\inference\api\demo_ci
   mkdir build
   cd build
   ```

2. 执行cmake（cmake可以在[官网进行下载](https://cmake.org/download/)，并添加到环境变量中):
   - 使用CPU预测库编译demo
   ```dos
   # -DDEMO_NAME 是要编译的文件
   # -DDPADDLE_LIB是预测库目录，例如-DPADDLE_LIB=D:\fluid_inference_install_dir
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DWITH_GPU=OFF -DWITH_MKL=ON -DWITH_STATIC_LIB=ON ^
   -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=simple_on_word2vec -DPADDLE_LIB=path_to_the_paddle_lib -DMSVC_STATIC_CRT=ON
   ```
   - 使用GPU预测库编译demo
   ```dos
   # -DCUDA_LIB CUDA的库目录，例如-DCUDA_LIB=D:\cuda\lib\x64
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DWITH_GPU=ON -DWITH_MKL=ON -DWITH_STATIC_LIB=ON ^
   -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=simple_on_word2vec -DPADDLE_LIB=path_to_the_paddle_lib -DMSVC_STATIC_CRT=ON -DCUDA_LIB=YOUR_CUDA_LIB
   ```
3. 使用Blend for Visual Studio 2015 打开 `cpp_inference_demo.sln` 文件，选择平台为`x64`，配置为`Release`，编译simple_on_word2vec项目。
   操作方法: 在Visual Studio中选择相应模块，右键选择"生成"（或者"build"）

4. [下载模型](http://paddle-inference-dist.bj.bcebos.com/word2vec.inference.model.tar.gz)并解压到当前目录，执行命令：
   ```dos
   # 开启GLOG
   set GLOG_v=100
   # 进行预测，path为模型解压后的目录
   Release\simple_on_word2vec.exe --dirname=path\word2vec.inference.model
   ```

### 实现一个简单预测demo

[完整的代码示例](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/api/demo_ci/windows_mobilenet.cc)

本示例使用了AnalysisConfig管理AnalysisPredictor的预测配置，提供了模型路径设置、预测引擎运行设备选择以及使用ZeroCopyTensor管理输入/输出的设置。具体步骤如下：

1. 创建AnalysisConfig
   ```C++
   AnalysisConfig config;
   config->SwitchUseFeedFetchOps(false);  // 关闭feed和fetch OP使用，使用ZeroCopy接口必须设置此项
   // config->EnableUseGpu(100 /*设定GPU初始显存池为MB*/,  0 /*设定GPU ID为0*/); //开启GPU预测
   ```

2. 在config中设置模型和参数路径

   从磁盘加载模型时，根据模型和参数文件存储方式不同，设置AnalysisConfig加载模型和参数的路径有两种形式，此处使用combined形式：
   - 非combined形式：模型文件夹`model_dir`下存在一个模型文件和多个参数文件时，传入模型文件夹路径，模型文件名默认为`__model__`。
   ``` c++
   config->SetModel("path\\model_dir")
   ```
   - combined形式：模型文件夹`model_dir`下只有一个模型文件`__model__`和一个参数文件`__params__`时，传入模型文件和参数文件路径。
   ```C++
   config->SetModel("path\\model_dir\\__model__", "path\\model_dir\\__params__");
   ```
3. 创建predictor，准备输入数据
   ```C++
   std::unique_ptr<PaddlePredictor> predictor = CreatePaddlePredictor(config);
   int batch_size = 1;
   int channels = 3; // channels，height，width三个参数必须与模型中对应输入的shape一致
   int height = 300;
   int width = 300;
   int nums = batch_size * channels * height * width;

   float* input = new float[nums];
   for (int i = 0; i < nums; ++i) input[i] = 0;
   ```
4. 使用ZeroCopyTensor管理输入
   ```C++
   // 通过创建的AnalysisPredictor获取输入Tensor，该Tensor为ZeroCopyTensor
   auto input_names = predictor->GetInputNames();
   auto input_t = predictor->GetInputTensor(input_names[0]);

   // 对Tensor进行reshape，将准备好的输入数据从CPU拷贝到ZeroCopyTensor中
   input_t->Reshape({batch_size, channels, height, width});
   input_t->copy_from_cpu(input);
   ```

5. 运行预测引擎
   ```C++
   predictor->ZeroCopyRun();
   ```

6. 使用ZeroCopyTensor管理输出
   ```C++
   auto output_names = predictor->GetOutputNames();
   auto output_t = predictor->GetOutputTensor(output_names[0]);
   std::vector<int> output_shape = output_t->shape();
   int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                 std::multiplies<int>());

   out_data.resize(out_num);
   output_t->copy_to_cpu(out_data.data()); // 将ZeroCopyTensor中数据拷贝到cpu中，得到输出数据
   delete[] input;
   ```
**Note:** 关于AnalysisPredictor的更多介绍，请参考[C++预测API介绍](./native_infer.html)
