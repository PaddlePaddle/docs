# **Compile on Windows from Source Code**

## Environment preparation

* **Windows 7/8/10 Pro/Enterprise(64bit)**
* **GPU Version support CUDA 10.1/10.2/11.0/11.1/11.2, and only support single GPU**
* **Python version 3.7+/3.8+/3.9+(64bit)**
* **pip version 20.2.2 or above (64bit)**
* **Visual Studio 2017**

## Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install CPU version of PaddlePaddle

* If your computer has NVIDIA® GPU, and the following conditions are met，GPU version of PaddlePaddle is recommended.
    * **CUDA toolkit 10.1/10.2 with cuDNN v7.6.5+**
    * **CUDA toolkit 11.0 with cuDNN v8.0.2+**
    * **CUDA toolkit 11.1 with cuDNN v8.1.1+**
    * **CUDA toolkit 11.2 with cuDNN v8.2.1**
    * **GPU's computing capability exceeds 3.5**

## Installation steps

There is one compilation methods in Windows system:

* [Direct native source code compilation](#compile_from_host)(NCCL, distribution are not supported on windows now)

<a name="win_source"></a>
### <span id="compile_from_host">***Direct native source code compilation***</span>

**Please strictly follow the following instructions step by step**

1. Install the necessary tools i.e. cmake, git and python:

    > CMake requires version 3.15 and above, but there are official [Bug](https://cmake.org/pipermail/cmake/2018-September/068195.html) versions of 3.12/3.13/3.14 when the GPU is compiled, we recommend that you use CMake3. 16 version, available on the official website [download] (https://cmake.org/download/), and add to the ring Environment variables.

    > Python requires version 3.6 and above,  which can be downloaded from the [official website](https://www.python.org/downloads/release/python-3610/).

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

    cd Paddle
    ```

3. Switch to `develop` branch for compilation:

    ```
    git checkout develop
    ```

    Note: python3.7 version started supporting from release/1.2, python3.8 version started supporting from release/1.8, python3.9 version started supporting from release/2.1, python3.10 version started supporting from release/2.3 branch

4. Create a directory called build and enter it:

    ```
    mkdir build

    cd build
    ```

5. Execute cmake:

    > For details on the compilation options, see [the compilation options list](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/Tables.html#Compile). On Windows,
    you can compile by `Ninja(recommended)` or `Visual Studio IDE`, as follow:

    *  1）Compile by `Ninja(recommended)` method:

        Firstly, install ninja:
        ```
        pip install ninja
        ```

        Then, search "x64 Native Tools Command Prompt for VS" in Windows search bar, run it as Administrator. Here is the cmake command:
        ```
        cmake .. -GNinja -DWITH_GPU=OFF -DWITH_UNITY_BUILD=ON
        ```

    *  2）Compile by `Visual Studio IDE` method:

        ```
        cmake .. -G "Visual Studio 15 2017" -A x64 -T host=x64 -DWITH_GPU=OFF -DWITH_UNITY_BUILD=ON
        ```

        In the above command, change to `-DWITH_GPU=ON` to compile the GPU version Paddle.

        > Note:
        > 1. If more than one CUDA are installed, the latest installed CUDA will be used, and you can't specify CUDA version.
        > 2. If more than one Python are installed, the latest installed Python will be used by default, and you can choose the Python version by `-DPYTHON_EXECUTABLE` . for example:
        ```
        cmake .. -GNinja -DWITH_GPU=ON -DPYTHON_EXECUTABLE=C:\\Python36\\python.exe  -DWITH_UNITY_BUILD=ON
        ```

6. Execute compile:
    * 1) For `Ninja` method(recommended), run `ninja all` , it will begin to compile.

    * 2) For `Visual Studio IDE` method, use Visual Studio to open `paddle.sln` file, select the platform `x64`, configure with `Release`, click the button, it will begin to compile.

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
