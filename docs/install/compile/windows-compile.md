# **Windows下从源码编译**

在Windows系统下提供1种编译方式：

* [本机编译](#compile_from_host)

## 环境准备

* **Windows 7/8/10 专业版/企业版 (64bit)**
* **Python 版本 3.6/3.7/3.8/3.9/3.10 (64 bit)**
* **Visual Studio 2017 社区版/专业版/企业版**

## 选择CPU/GPU

* 如果你的计算机硬件没有 NVIDIA® GPU，请编译CPU版本的PaddlePaddle

* 如果你的计算机硬件有 NVIDIA® GPU，推荐编译GPU版本的PaddlePaddle，建议安装 **CUDA 10.1/10.2/11.0/11.1/11.2**

## 本机编译过程

1. 安装必要的工具 cmake, git, python, Visual studio 2017：

    > **cmake**：建议安装CMake3.17版本, 官网下载[链接](https://cmake.org/files/v3.17/cmake-3.17.0-win64-x64.msi)。安装时注意勾选 `Add CMake to the system PATH for all users`，将CMake添加到环境变量中。

    > **git**：官网下载[链接](https://github.com/git-for-windows/git/releases/download/v2.35.1.windows.2/Git-2.35.1.2-64-bit.exe)，使用默认选项安装。

    > **python**：官网[链接](https://www.python.org/downloads/windows/)，可选择3.6/3.7/3.8/3.9/3.10中任一版本的 Windows installer(64-bit)安装。安装时注意勾选 `Add Python 3.x to PATH`，将Python添加到环境变量中。

    > **Visual studio 2017**：官网[链接](https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/#visual-studio-2017-and-other-products)，需要登录后下载，建议下载Community社区版。在安装时需要在工作负荷一栏中勾选 `使用C++的桌面开发` 和 `通用Windows平台开发`，并在语言包一栏中选择 `英语`。

2. 在Windows桌面下方的搜索栏中搜索 `x64 Native Tools Command Prompt for VS 2017` 或 `适用于VS 2017 的x64本机工具命令提示符`，右键以管理员身份打开终端。之后的命令均在该终端中执行。

3. 使用`pip`命令安装Python依赖：
    * 通过 `python --version` 检查默认python版本是否是预期版本，因为你的计算机可能安装有多个python，你可通过修改系统环境变量的顺序来修改默认Python版本。
    * 安装 numpy, protobuf, wheel, ninja
        ```
        pip install numpy protobuf wheel ninja
        ```

4. 创建编译Paddle的文件夹（例如D:\workspace），进入该目录并下载源码：

    ```
    mkdir D:\workspace && cd /d D:\workspace

    git clone https://github.com/PaddlePaddle/Paddle.git

    cd Paddle
    ```

5. 切换到2.2分支下进行编译：

    ```
    git checkout release/2.3
    ```

6. 创建名为build的目录并进入：

    ```
    mkdir build

    cd build
    ```

7. 执行cmake：

    编译CPU版本的Paddle：

    ```
    cmake .. -GNinja -DWITH_GPU=OFF
    ```

    编译GPU版本的Paddle：

    ```
    cmake .. -GNinja -DWITH_GPU=ON
    ```

    其他编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)。

    > 注意：
    > 1. 如果本机安装了多个CUDA，将使用最新安装的CUDA版本，且无法指定。
    > 2. 如果本机安装了多个Python，将使用最新安装的Python版本。若需要指定Python版本，则需要指定Python路径，例如：
    ```
    cmake .. -GNinja -DWITH_GPU=ON -DPYTHON_EXECUTABLE=C:\Python38\python.exe -DPYTHON_INCLUDE_DIR=C:\Python38\include -DPYTHON_LIBRARY=C:\Python38\libs\python38.lib
    ```

8. 执行编译：

    ```
    ninja
    ```

9. 编译成功后进入 `python\dist` 目录下找到生成的 `.whl` 包：

    ```
    cd python\dist
    ```

10. 安装编译好的 `.whl` 包：

    ```
    pip install（whl包的名字）--force-reinstall
    ```

恭喜，至此你已完成PaddlePaddle的编译安装


## **验证安装**

安装完成后你可以使用 `python` 进入python解释器，输入：

```
import paddle
```

```
paddle.utils.run_check()
```

如果出现`PaddlePaddle is installed successfully!`，说明你已成功安装。

## **如何卸载**
请使用以下命令卸载PaddlePaddle：

* **CPU版本的PaddlePaddle**:
    ```
    pip uninstall paddlepaddle
    ```

* **GPU版本的PaddlePaddle**:
    ```
    pip uninstall paddlepaddle-gpu
    ```
