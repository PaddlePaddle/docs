
Install and Compile C++ Inference Library on Windows
===========================

Direct Download and Install
-------------

| Version      |     Inference Libraries(v1.6.1)   | Compiler | Build tools | cuDNN | CUDA |
|:---------|:-------------------|:-------------------|:----------------|:--------|:-------|
|    cpu_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.1/win-infer/mkl/cpu/fluid_inference_install_dir.zip) | MSVC 2015 update 3|  CMake v3.11.1  |
|    cpu_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.1/win-infer/open/cpu/fluid_inference_install_dir.zip) | MSVC 2015 update 3|  CMake v3.11.1  |
|    cuda9.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.1/win-infer/mkl/post97/fluid_inference_install_dir.zip) |  MSVC 2015 update 3 |  CMake v3.11.1  |  7.3.1  |   9    |
|    cuda10.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.1/win-infer/mkl/post107/fluid_inference_install_dir.zip) | MSVC 2015 update 3 |  CMake v3.11.1  |  7.4.1  |   10    |
|    cuda9.0_cudnn7_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.6.1/win-infer/open/post97/fluid_inference_install_dir.zip) | MSVC 2015 update 3 |  CMake v3.11.1  |  7.3.1  |   9    |

### Hardware Environment

Hardware Configuration of the experimental environment:

| CPU           |      I7-8700K      |
|:--------------|:-------------------|
| Memory        | 16G               |
| Hard Disk     | 1T hdd + 256G ssd |
| Graphics Card | GTX1080 8G        |

The operating system is win10 family version in the experimental environment.

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
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=OFF -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF
   # use -DWITH_GPU to control we are building CPU or GPU version
   # use -DWITH_MKL to select math library: Intel MKL or OpenBLAS

   # By default on Windows we use /MT for C Runtime Library, If you want to use /MD, please use the below command
   # If you have no ideas the differences between the two, use the above one
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=OFF -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF -DMSVC_STATIC_CRT=OFF
   ```

3. Open the `paddle.sln` using VisualStudio 2015, choose the`x64` for Slution Platforms, and `Release` for Solution Configurations, then build the `inference_lib_dist` project in the Solution Explorer(Rigth click the project and click Build).

The inference library will be installed in `fluid_inference_install_dir`.

version.txt constains the detailed configurations about the library, including git commit ID、math library, CUDA, CUDNN versions：


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

### Other requirements

1. You need to download the Windows inference library or compile the inference library from Paddle source code.

2. You need to run the command to get the Paddle source code.
```bash
git clone https://github.com/PaddlePaddle/Paddle.git
```

### Usage of Inference demo

#### Compile with script

Run the run_windows_demo.bat from cmd on windows, and input parameters as required according to the prompts.
```dos
# Path is the directory where you downloaded paddle.
cd path\Paddle\paddle\fluid\inference\api\demo_ci
run_windows_demo.bat 
```
Some options of the script are as follows:

```dos
gpu_inference=Y # Use gpu_inference_lib or not(Y/N), default: N.
use_mkl=Y # Use MKL or not(Y/N), default: Y.
use_gpu=Y  # Whether to use GPU for prediction, defalut: N.

paddle_inference_lib=path\fluid_inference_install_dir # Set the path of paddle inference library.
cuda_lib_dir=path\lib\x64  # Set the path of cuda library.
vcvarsall_dir=path\vc\vcvarsall.bat  # Set the path of visual studio command prompt.
```

#### Compile manually

```dos
# Path is the directory where you downloaded paddle.
cd path\Paddle\paddle\fluid\inference\api\demo_ci
mkdir build
cd build
cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=simple_on_word2vec -DPADDLE_LIB=path_to_the_patddle\paddle_fluid.lib -DMSVC_STATIC_CRT=ON
```

Note:

-DDEMO_NAME is the file to be built

-DPADDLE_LIB is the path of fluid_install_dir, for example:
-DPADDLE_LIB=D:\fluid_install_dir


Cmake can be [downloaded at official site](https://cmake.org/download/) and added to environment variables.

After the execution, the directory build is shown in the picture below. Then please open the solution file that which the arrow points at:

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image3.png">
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

[Download model](http://paddle-inference-dist.bj.bcebos.com/word2vec.inference.model.tar.gz) and decompress it to the current directory. Run the command:

  1.  Open GLOG

  	`set GLOG_v=100`

  2.  Start inference. Path is the directory where you decompres model.

  	`Release\simple_on_word2vec.exe --dirname=path\word2vec.inference.model`

<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image9.png">
</p>

## Using AnalysisConfig to manage prediction configurations

[Complete code example](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/api/demo_ci/windows_mobilenet.cc)

This example uses Analysisconfig to manage the Analysispredictor prediction configuration. The configuration method is as follows:

#### Create AnalysisConfig
``` c++
AnalysisConfig config;
```
**Note:** With ZeroCopyTensor, you must set `config->SwitchUseFeedFetchOps(false)` when creating the config.
``` c++
config->SwitchUseFeedFetchOps(false);  // Turn off the use of feed and fetch OP, this must be set when using the ZeroCopy interface.
config->EnableUseGpu(100 /*Set the GPU initial memory pool to 100MB*/,  0 /*Set GPU ID to 0*/); // Turn on GPU prediction
```

#### Set paths of models and parameters
``` c++
config->SetModel("./model_dir/__model__", "./model_dir/__params__");
```

#### Manage input with ZeroCopyTensor
ZeroCopyTensor is the input/output data structure of AnalysisPredictor

**Note:** With ZeroCopyTensor, you must set `config->SwitchUseFeedFetchOps(false)` when creating the config.

``` c++
auto input_names = predictor->GetInputNames();
auto input_t = predictor->GetInputTensor(input_names[0]);

// Reshape the input tensor, where the parameters of channels, height, and width must be the same as those required by the input in the model.
input_t->Reshape({batch_size, channels, height, width});
```

#### Run prediction engine
```C++
predictor->ZeroCopyRun();
```

#### Manage input with ZeroCopyTensor
```C++
auto output_names = predictor->GetOutputNames();
auto output_t = predictor->GetOutputTensor(output_names[0]);
```
**Note:** For more introduction to AnalysisPredictor, please refer to the [introduction of C++ Prediction API](./native_infer_en.html).

