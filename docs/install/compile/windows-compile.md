# **Windows 下从源码编译**

在 Windows 系统下提供 1 种编译方式：

* [本机编译](#compile_from_host)

## 环境准备

* **Windows 7/8/10 专业版/企业版 (64bit)**
* **Python 版本 3.7/3.8/3.9/3.10 (64 bit)**
* **Visual Studio 2017/2019 社区版/专业版/企业版**

## 选择 CPU/GPU

* 如果你的计算机硬件没有 NVIDIA® GPU，请编译 CPU 版本的 PaddlePaddle

* 如果你的计算机硬件有 NVIDIA® GPU，推荐编译 GPU 版本的 PaddlePaddle，建议安装 **CUDA 10.1/10.2/11.0/11.1/11.2/11.6**

## 本机编译过程

1. 安装必要的工具 cmake, git, python, Visual studio 2017/2019：

    > **cmake**：建议安装 CMake3.17 版本, 官网下载[链接](https://cmake.org/files/v3.17/cmake-3.17.0-win64-x64.msi)。安装时注意勾选 `Add CMake to the system PATH for all users`，将 CMake 添加到环境变量中。

    > **git**：官网下载[链接](https://github.com/git-for-windows/git/releases/download/v2.35.1.windows.2/Git-2.35.1.2-64-bit.exe)，使用默认选项安装。

    > **python**：官网[链接](https://www.python.org/downloads/windows/)，可选择 3.7/3.8/3.9/3.10 中任一版本的 Windows installer(64-bit)安装。安装时注意勾选 `Add Python 3.x to PATH`，将 Python 添加到环境变量中。

    > **Visual studio**：需根据 CUDA 版本选择对应的 Visual studio 版本，当只编译 CPU 版本或者 CUDA 版本 < 11.2 时，安装 VS2017；当 CUDA 版本 >= 11.2 时，安装 VS2019。官网[链接](https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/)，需要登录后下载，建议下载 Community 社区版。在安装时需要在工作负荷一栏中勾选 `使用 C++的桌面开发` 和 `通用 Windows 平台开发`，并在语言包一栏中选择 `英语`。

2. 打开 Visual studio 终端：在 Windows 桌面下方的搜索栏中搜索终端，若安装的是 VS2017 版本，则搜索 `x64 Native Tools Command Prompt for VS 2017` 或 `适用于 VS 2017 的 x64 本机工具命令提示符`；若安装的是 VS2019 版本，则搜索 `x64 Native Tools Command Prompt for VS 2019` 或 `适用于 VS 2019 的 x64 本机工具命令提示符`，然后右键以管理员身份打开终端。后续的命令将在该终端执行。

3. 使用`pip`命令安装 Python 依赖：
    * 通过 `python --version` 检查默认 python 版本是否是预期版本，因为你的计算机可能安装有多个 python，可通过修改系统环境变量的顺序来修改默认 Python 版本。
    * 安装 numpy, protobuf, wheel, ninja
        ```
        pip install numpy protobuf wheel ninja
        ```

4. 创建编译 Paddle 的文件夹（例如 D:\workspace），进入该目录并下载源码：

    ```
    mkdir D:\workspace && cd /d D:\workspace

    git clone https://github.com/PaddlePaddle/Paddle.git

    cd Paddle
    ```

5. 创建名为 build 的目录并进入：

    ```
    mkdir build

    cd build
    ```

6. 执行 cmake：

    编译 CPU 版本的 Paddle：

    ```
    cmake .. -GNinja -DWITH_GPU=OFF -DWITH_UNITY_BUILD=ON
    ```

    编译 GPU 版本的 Paddle：

    ```
    cmake .. -GNinja -DWITH_GPU=ON -DWITH_UNITY_BUILD=ON
    ```

    其他编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)。

    > 注意：
    > 1. 如果本机安装了多个 CUDA，将使用最新安装的 CUDA 版本。若需要指定 CUDA 版本，则需要设置环境变量和 cmake 选项，例如：
    ```
    set CUDA_TOOLKIT_ROOT_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2
    set PATH=%CUDA_TOOLKIT_ROOT_DIR:/=\%\bin;%CUDA_TOOLKIT_ROOT_DIR:/=\%\libnvvp;%PATH%
    cmake .. -GNinja -DWITH_GPU=ON -DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_ROOT_DIR%" -DWITH_UNITY_BUILD=ON
    ```
    > 2. 如果本机安装了多个 Python，将自动使用最新安装的 Python 版本。若需要指定 Python 版本，则需要指定 Python 路径，例如：
    ```
    cmake .. -GNinja -DWITH_GPU=ON -DPYTHON_EXECUTABLE=C:\Python38\python.exe -DPYTHON_INCLUDE_DIR=C:\Python38\include -DPYTHON_LIBRARY=C:\Python38\libs\python38.lib
    -DWITH_UNITY_BUILD=ON
    ```

7. 执行编译：

    ```
    ninja
    ```

8. 编译成功后进入 `python\dist` 目录下找到生成的 `.whl` 包：

    ```
    cd python\dist
    ```

9. 安装编译好的 `.whl` 包：

    ```
    pip install（whl 包的名字）--force-reinstall
    ```

恭喜，至此你已完成 PaddlePaddle 的编译安装


## **验证安装**

安装完成后你可以使用 `python` 进入 python 解释器，输入：

```
import paddle
```

```
paddle.utils.run_check()
```

如果出现`PaddlePaddle is installed successfully!`，说明你已成功安装。

## **如何卸载**
请使用以下命令卸载 PaddlePaddle：

* **CPU 版本的 PaddlePaddle**:
    ```
    pip uninstall paddlepaddle
    ```

* **GPU 版本的 PaddlePaddle**:
    ```
    pip uninstall paddlepaddle-gpu
    ```
