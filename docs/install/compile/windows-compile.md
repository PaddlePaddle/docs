# **Windows下从源码编译**

## 环境准备

* **Windows 7/8/10 专业版/企业版 (64bit)**
* **GPU版本支持CUDA 10.1/10.2/11.0/11.1/11.2，且仅支持单卡**
* **Python 版本 3.6/3.7/3.8/3.9 (64 bit)**
* **pip 版本 20.2.2或更高版本 (64 bit)**
* **Visual Studio 2017**

## 选择CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请编译CPU版的PaddlePaddle

* 如果您的计算机有NVIDIA® GPU，并且满足以下条件，推荐编译GPU版的PaddlePaddle
    * **CUDA 工具包 10.1/10.2 配合 cuDNN v7.6.5**
    * **CUDA 工具包 11.0 配合 cuDNN v8.0.2**
    * **CUDA 工具包 11.1 配合 cuDNN v8.1.1**
    * **CUDA 工具包 11.2 配合 cuDNN v8.2.1**
    * **GPU运算能力超过3.5的硬件设备**

## 安装步骤

在Windows的系统下提供1种编译方式：

* [本机编译](#compile_from_host)（暂不支持NCCL，分布式等相关功能）

<a name="win_source"></a>
### <span id="compile_from_host">**本机编译**</span>

1. 安装必要的工具 cmake, git, python, Visual studio 2017：

    > cmake：建议安装CMake3.17版本, 官网下载[链接]](https://cmake.org/files/v3.17/cmake-3.17.0-win64-x64.msi)，注意勾选 **Add CMake to the system PATH for all users**，将CMake添加到环境变量中。

    > git：官网下载[链接](https://github.com/git-for-windows/git/releases/download/v2.35.1.windows.2/Git-2.35.1.2-64-bit.exe)，使用默认选项安装。

    > python：官网[链接](https://www.python.org/downloads/windows/)，可选择3.6/3.7/3.8/3.9中任一版本的 Windows installer(64-bit)安装。安装时需要勾选 **Add Python 3.x to PATH**，将Python添加到环境变量中。

    > Visual studio 2017：社区版下载[链接](https://paddle-ci.gz.bcebos.com/window_requirement/VS2017/vs_Community.exe)。在安装时需要在工作负荷一栏中勾选 **使用C++的桌面开发** 和 **通用Windows平台开发**，并在语言包一栏中选择 **英语**。

2. 在Windows桌面下方的搜索栏中搜索 "x64 Native Tools Command Prompt for VS 2017" 或 "适用于VS 2017 的x64本机工具命令提示符"，以管理员身份运行，打开终端。之后的命令均在该终端中执行。

3. 使用`pip`命令安装Python依赖：`numpy, protobuf, wheel, ninja`
    * 通过 `python --version` 检查默认python版本是否是预期版本，因为您的计算机可能安装有多个python，您可通过修改系统环境变量的顺序修改默认的Python版本。
    * 更新 pip
        ```
        pip install --upgrade pip --user
        ```
    * 安装 numpy
        ```
        pip install numpy
        ```
    * 安装 protobuf
        ```
        pip install protobuf
        ```
    * 安装 wheel
        ```
        pip install wheel
        ```
    * 安装 ninja
        ```
        pip install ninja
        ```

4. 创建要拉取PaddlePaddle源码的目录并进入（例如D:\workspace）。将PaddlePaddle的源码clone在当前目录下的Paddle的文件夹中，并进入Padde目录下：

    ```
    mkdir D:\workspace && cd /d D:\workspace

    git clone https://github.com/PaddlePaddle/Paddle.git

    cd Paddle
    ```

5. 切换到`develop`分支下进行编译：

    ```
    git checkout develop
    ```

    注意：python3.6、python3.7版本从release/1.2分支开始支持, python3.8版本从release/1.8分支开始支持, python3.9版本从release/2.1分支开始支持

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

    上述命令中，改成 `-DWITH_GPU=ON` 即可编译GPU版本的Paddle。具体编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)。

    > 注意：
    > 1. 如果本机安装了多个CUDA，则生效的为最新安装的CUDA，且无法指定。
    > 2. 如果本机安装了多个Python，将使用系统默认的Python版本。若需要指定Python版本，需要指定Python路径，例如：
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
    pip install -U（whl包的名字）
    ```

恭喜，至此您已完成PaddlePaddle的编译安装

## **验证安装**
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入
```
import paddle
```
再输入
```
paddle.utils.run_check()
```

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

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
