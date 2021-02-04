# **Compile on Windows from Source Code**

## Environment preparation

* **Windows 7/8/10 Pro/Enterprise(64bit)**
* **GPU Version support CUDA 9.0/10.0/10.1/10.2/11.0, and only support single GPU**
* **Python version 2.7.15+/3.5.1+/3.6+/3.7+/3.8+(64bit)**
* **pip version 20.2.2+ (64bit)**
* **Visual Studio 2015 Update3**

## Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install CPU version of PaddlePaddle

* If your computer has NVIDIA® GPU, and the following conditions are met，GPU version of PaddlePaddle is recommended.
    * **CUDA toolkit 9.0/10.0/10.1/10.2 with cuDNN v7.6.5+**
    * **CUDA toolkit 11.0 with cuDNN v8.0.4**
    * **GPU's computing capability exceeds 3.0**

## Installation steps

There is one compilation methods in Windows system:

* [Direct native source code compilation](#compile_from_host)(NCCL, distributed and other related functions are not supported temporarily)

<a name="win_source"></a>
### <span id="compile_from_host">***Direct native source code compilation***</span>

**Please strictly follow the following instructions step by step**

1. Install the necessary tools i.e. cmake, git and python:

    > CMake requires version 3.10 and above, but there are official [Bug](https://cmake.org/pipermail/cmake/2018-September/068195.html) versions of 3.12/3.13/3.14 when the GPU is compiled, we recommend that you use CMake3. 16 version, available on the official website [download] (https://cmake.org/download/), and add to the ring Environment variables.

    > Python requires version 2.7 and above,  which can be downloaded from the [official website](https://www.python.org/download/releases/2.7/).

    * After installing python, please check whether the python version is the expected version by `python-version`, because you may have more than one python installed on your computer. You can handle conflicts of multiple pythons by changing the order of the environment variables.

    > `numpy, protobuf, wheel` are needed to be installed. Use the 'pip' command.

    * To Install numpy package you can use command
        ```
        pip install numpy
        ```

    * To Install protobuf package you can use command
        ```
        pip install protobuf
        ```

    * To Install Wheel package you can use command
        ```
        pip install wheel
        ```

    > Git can be downloaded on the [official website](https://gitforwindows.org/) and added to the environment variable.

2. Clone the PaddlePaddle source code to the Paddle subdirectories of the current directory and go to the Paddle subdirectories:

    ```
    git clone https://github.com/PaddlePaddle/Paddle.git
    ```
    ```
    cd Paddle
    ```

3. Switch to `develop` branch for compilation:

    ```
    git checkout develop
    ```

    Note: python3.6、python3.7 version started supporting from release/1.2, python3.8 version started supporting from release/1.8

4. Create a directory called build and enter it:

    ```
    mkdir build
    ```
    ```
    cd build
    ```

5. Execute cmake:

    > For details on the compilation options, see [the compilation options list](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/Tables.html#Compile).
    * For users who need to compile **the CPU version PaddlePaddle**:

        ```
        cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
        ```

    * For users who need to compile **the GPU version PaddlePaddle**:

        ```
        cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
        ```

    Python2 by default，Python3 please add：

    > -DPY_VERSION=3 (or 3.5、3.6、3.7、3.8)

    If your device information contains multiple Python or CUDA, you can also specify a specific version of Python or CUDA by setting the corresponding compile options:

    > -DPYTHON_EXECUTABLE: the installation path of python

    > -DCUDA_TOOLKIT_ROOT_DIR: the installation path of CUDA

    For example: (for instance only, please set it according to your actual installation path)

    ```
    cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=ON -DWITH_TESTING=OFF -DPYTHON_EXECUTABLE=C:\\Python36\\python.exe -DCUDA_TOOLKIT_ROOT_DIR="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\v10.0"
    ```

6. Use Visual Studio 2015 to open `paddle.sln` file, select the platform `x64`, configure with `Release`, then begin to compile

7. After compilation successfully, go to the `\paddle\build\python\dist` directory and find the generated `.whl` package:

    ```
    cd \paddle\build\python\dist
    ```

8. Install the generated `.whl` package:

     ```
     pip install -U (whl package name)
     ```

Congratulations, you have completed the process of compiling PaddlePaddle successfully!

### ***Verify installation***

After the compilation and installation is completed, you can use `python` to enter the Python interface, input
```
import paddle
```
and then
```
paddle.utils.run_check()
```
to verify that the installation was successful.

If `PaddlePaddle is installed successfully!` appears, it means the compilation and installation was successful.


### ***How to uninstall***

Please use the following command to uninstall PaddlePaddle:

* ***CPU version of PaddlePaddle*** :
    ```
    pip uninstall paddlepaddle
    ```

* ***GPU version of PaddlePaddle*** :
    ```
    pip uninstall paddlepaddle-gpu
    ```
