
Install and Compile C++ Inference Library on Windows
===========================

Download and Install directly
-------------

| Version      |  Inference Libraries(v1.8.5)   |Inference Libraries(v2.0.0-rc0)| Compiler | Build tools | cuDNN | CUDA |
|:---------|:-------------------|:-------------------|:----------------|:--------|:-------|:-------|
|    cpu_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.5/win-infer/mkl/cpu/fluid_inference_install_dir.zip) | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.0-rc0/win-infer/mkl/cpu/paddle_inference_install_dir.zip) | MSVC 2015 update 3|  CMake v3.17.0  |
|    cpu_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.5/win-infer/open/cpu/fluid_inference_install_dir.zip) |[fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.0-rc0/win-infer/open/cpu/paddle_inference_install_dir.zip)| MSVC 2015 update 3|  CMake v3.17.0  |
|    cuda9.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.5/win-infer/mkl/post97/fluid_inference_install_dir.zip) |[fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.0-rc0/win-infer/mkl/post90/paddle_inference_install_dir.zip)|  MSVC 2015 update 3 |  CMake v3.17.0  |  7.6.5  |   9.0    |
|    cuda9.0_cudnn7_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.5/win-infer/open/post97/fluid_inference_install_dir.zip) || MSVC 2015 update 3 |  CMake v3.17.0  |  7.6.5  |   9.0    |
|    cuda10.0_cudnn7_avx_mkl | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.5/win-infer/mkl/post107/fluid_inference_install_dir.zip) | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.0-rc0/win-infer/mkl/post100/paddle_inference_install_dir.zip) | MSVC 2015 update 3 |  CMake v3.17.0  |  7.6.5  |   10.0    |
|    cuda10.0_cudnn7_avx_openblas | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/1.8.5/win-infer/open/post107/fluid_inference_install_dir.zip) | | MSVC 2015 update 3 |  CMake v3.17.0  |  7.6.5  |   10.0    |
|    cuda10.1_cudnn7_avx_mkl | | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.0-rc0/win-infer/mkl/post101/paddle_inference_install_dir.zip) | MSVC 2015 update 3 |  CMake v3.17.0  |  7.6.5  |   10.1   |
|    cuda10.2_cudnn7_avx_mkl |  | [fluid_inference.zip](https://paddle-wheel.bj.bcebos.com/2.0.0-rc0/win-infer/mkl/post102/paddle_inference_install_dir.zip) | MSVC 2015 update 3 |  CMake v3.17.0  |  7.6.5  |   10.2    |


### Hardware Environment

Hardware Configuration of the experimental environment:

|Operating System| win10 family version|
|:---------------|:-------------------|
| CPU            | I7-8700K      |
| Memory         | 16G               |
| Hard Disk      | 1T hdd + 256G ssd |
| Graphics Card  | GTX1080 8G        |

The operating system is win10 family version in the experimental environment.

Build From Source Code
--------------

Users can also compile C++ inference libraries from the PaddlePaddle core code by specifying the following compile options at compile time:

|Option    | Description    |   Value     |
|:-------------|:-----|:--------------|
|CMAKE_BUILD_TYPE|Specifies the build type on single-configuration generators, Windows inference library currently only supports Release| Release    |
|ON_INFER|Whether to generate the inference library. Must be set to ON when compiling the inference library. | ON   |
|WITH_GPU|Whether to support GPU   | ON/OFF     |
|WITH_MKL|Whether to support MKL   | ON/OFF     |
|WITH_PYTHON|Whether to generate the Python whl package| OFF        |
|MSVC_STATIC_CRT|Whether to compile with /MT mode |   ON   |
|CUDA_TOOKIT_ROOT_DIR | When compiling the GPU inference library, you need to set the CUDA root directory | YOUR_CUDA_PATH |

For details on the compilation options, see [the compilation options list](../../../beginners_guide/install/Tables_en.html/#Compile)

**Paddle Windows Inference Library Compilation Steps**

1. Clone Paddle source code from GitHub:
   ```bash
   git clone https://github.com/PaddlePaddle/Paddle.git
   cd Paddle
   ```

2. Run Cmake command

   - compile CPU inference library
   ```bash
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=ON -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF
   # use -DWITH_MKL to select math library: Intel MKL or OpenBLAS

   # By default on Windows we use /MT for C Runtime Library, If you want to use /MD, please use the below command
   # If you have no ideas the differences between the two, use the above one
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=ON -DWITH_GPU=OFF -DON_INFER=ON -DWITH_PYTHON=OFF -DMSVC_STATIC_CRT=OFF
   ```
   - compile GPU inference library
   ```bash
   # -DCUDA_TOOKIT_ROOT_DIR is cuda root directory, such as -DCUDA_TOOKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0"
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=ON -DWITH_GPU=ON -DON_INFER=ON -DWITH_PYTHON=OFF -DCUDA_TOOKIT_ROOT_DIR=YOUR_CUDA_PATH
   ```

3. Open the `paddle.sln` using VisualStudio 2015, choose the`x64` for Slution Platforms, and `Release` for Solution Configurations, then build the `inference_lib_dist` project in the Solution Explorer(Rigth click the project and click Build).

The inference library will be installed in `fluid_inference_install_dir`.

version.txt constains the detailed configurations about the library, including git commit ID、math library, CUDA, CUDNN versions, CXX compiler version:


```text
GIT COMMIT ID: 264e76cae6861ad9b1d4bcd8c3212f7a78c01e4d
WITH_MKL: ON
WITH_MKLDNN: ON
WITH_GPU: ON
CUDA version: 10.0
CUDNN version: v7.4
CXX compiler version: 19.0.24215.1
```


Inference Demo Compilation
-------------------

### Hardware Environment

Hardware Configuration of the experimental environment:

|Operating System| win10 family version|
|:---------------|:-------------------|
| CPU            | I7-8700K      |
| Memory         | 16G               |
| Hard Disk      | 1T hdd + 256G ssd |
| Graphics Card  | GTX1080 8G        |

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

Open the windows command line and run the `run_windows_demo.bat`, and input parameters as required according to the prompts.
```dos
# Path is the directory of Paddle you downloaded.
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

1. Create and change to the build directory
   ```dos
   # path is the directory where Paddle is downloaded
   cd path\Paddle\paddle\fluid\inference\api\demo_ci
   mkdir build
   cd build
   ```
2. Run Cmake command, cmake can be [downloaded at official site](https://cmake.org/download/) and added to environment variables.
   - compile inference demo with CPU inference library
   ```dos
   # Path is the directory where you downloaded paddle.
   # -DDEMO_NAME is the file to be built
   # DPADDLE_LIB is the path of fluid_install_dir, for example: DPADDLE_LIB=D:\fluid_install_dir

   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DWITH_GPU=OFF -DWITH_MKL=OFF -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=simple_on_word2vec -DPADDLE_LIB=path_to_the_paddle_lib -DMSVC_STATIC_CRT=ON
   ```
   - compile inference demo with GPU inference library
   ```dos
   cmake .. -G "Visual Studio 14 2015" -A x64 -T host=x64 -DWITH_GPU=ON -DWITH_MKL=ON -DWITH_STATIC_LIB=ON ^
   -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=simple_on_word2vec -DPADDLE_LIB=path_to_the_paddle_lib -DMSVC_STATIC_CRT=ON -DCUDA_LIB=YOUR_CUDA_LIB
   ```
3. Open the `cpp_inference_demo.sln` using VisualStudio 2015, choose the`x64` for Slution Platforms, and `Release` for Solution Configurations, then build the `simple_on_word2vec` project in the Solution Explorer(Rigth click the project and click Build).

   In the dependent packages provided, please copy openblas and model files under Release directory to the directory of Release built and generated.

   <p align="center">
   <img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/advanced_usage/deploy/inference/image/image8.png">
   </p>

4. [Download model](http://paddle-inference-dist.bj.bcebos.com/word2vec.inference.model.tar.gz) and decompress it to the current directory. Run the command:
   ```dos
   #  Open GLOG
   set GLOG_v=100

   # Start inference, path is the directory where you decompres model
   Release\simple_on_word2vec.exe --dirname=path\word2vec.inference.model
   ```

### Implementing a simple inference demo

[Complete code example](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/inference/api/demo_ci/windows_mobilenet.cc)

This example uses Analysisconfig to manage the Analysispredictor prediction configuration. The configuration method is as follows:

1. Create AnalysisConfig
   ``` c++
   AnalysisConfig config;
   config->SwitchUseFeedFetchOps(false);  // Turn off the use of feed and fetch OP, this must be set when using the ZeroCopy interface.
   // config->EnableUseGpu(100 /*Set the GPU initial memory pool to 100MB*/,  0 /*Set GPU ID to 0*/); // Turn on GPU prediction
   ```

2. Set path of models and parameters
   - When there is a model file and multiple parameter files under the model folder `model_dir`, the model folder path is passed in, and the model file name defaults to `__model__`.
   ``` c++
   config->SetModel("path\\model_dir\\__model__", "path\\model_dir\\__params__");
   ```

   - When there is only one model file `__model__` and one parameter file `__params__` in the model folder `model_dir`, the model file and parameter file path are passed in.
   ```C++
   config->SetModel("path\\model_dir\\__model__", "path\\model_dir\\__params__");
   ```

3. Create predictor and prepare input data
   ``` C++
   std::unique_ptr<PaddlePredictor> predictor = CreatePaddlePredictor(config);
   int batch_size = 1;
   int channels = 3; // The parameters of channels, height, and width must be the same as those required by the input in the model.
   int height = 300;
   int width = 300;
   int nums = batch_size * channels * height * width;

   float* input = new float[nums];
   for (int i = 0; i < nums; ++i) input[i] = 0;
   ```

4. Manage input with ZeroCopyTensor
   ```C++
   auto input_names = predictor->GetInputNames();
   auto input_t = predictor->GetInputTensor(input_names[0]);

   // Reshape the input tensor, copy the prepared input data from the CPU to ZeroCopyTensor
   input_t->Reshape({batch_size, channels, height, width});
   input_t->copy_from_cpu(input);
   ```

5. Run prediction engine
   ```C++
   predictor->ZeroCopyRun();
   ```

6.  Manage input with ZeroCopyTensor
   ```C++
   auto output_names = predictor->GetOutputNames();
   auto output_t = predictor->GetOutputTensor(output_names[0]);
   std::vector<int> output_shape = output_t->shape();
   int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                 std::multiplies<int>());

   out_data.resize(out_num);
   output_t->copy_to_cpu(out_data.data()); // Copy data from ZeroCopyTensor to cpu
   delete[] input;
   ```  
**Note:** For more introduction to AnalysisPredictor, please refer to the [introduction of C++ Prediction API](./native_infer_en.html).
