# **Windows下从源码编译**

## 环境准备

* **Windows 7/8/10 专业版/企业版 (64bit)**
* **GPU版本支持CUDA 9.0/10.0/10.1/10.2/11.0，且仅支持单卡**
* **Python 版本 2.7.15+/3.5.1+/3.6+/3.7+/3.8+ (64 bit)**
* **pip 版本 20.2.2+ (64 bit)**
* **Visual Studio 2015 Update3**

## 选择CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请编译CPU版的PaddlePaddle

* 如果您的计算机有NVIDIA® GPU，并且满足以下条件，推荐编译GPU版的PaddlePaddle
    * **CUDA 工具包 9.0/10.0/10.1/10.2 配合 cuDNN v7.6.5+**
    * **CUDA 工具包 11.0 配合 cuDNN v8.0.4**
    * **GPU运算能力超过3.0的硬件设备**

## 安装步骤

在Windows的系统下提供1种编译方式：

* [本机编译](#compile_from_host)（暂不支持NCCL，分布式等相关功能）

<a name="win_source"></a>
### <span id="compile_from_host">**本机编译**</span>

1. 安装必要的工具 cmake，git 以及 python：

    > cmake我们支持3.10以上版本,但GPU编译时3.12/3.13/3.14版本存在官方[Bug](https://cmake.org/pipermail/cmake/2018-September/068195.html),我们建议您使用CMake3.16版本, 可在官网[下载](https://cmake.org/download/)，并添加到环境变量中。

    > python 需要 2.7 及以上版本, 可在官网[下载](https://www.python.org/download/releases/2.7/)。

    * 安装完python 后请通过 `python --version` 检查python版本是否是预期版本，因为您的计算机可能安装有多个python，您可通过修改环境变量的顺序来处理多个python时的冲突。

    > 需要安装`numpy, protobuf, wheel` 。 请使用`pip`命令;

    * 安装 numpy 包可以通过命令
        ```
        pip install numpy
        ```
    * 安装 protobuf 包可以通过命令
        ```
        pip install protobuf
        ```
    * 安装 wheel 包可以通过命令
        ```
        pip install wheel
        ```

    > git可以在官网[下载](https://gitforwindows.org/)，并添加到环境变量中。

2. 将PaddlePaddle的源码clone在当前目录下的Paddle的文件夹中，并进入Padde目录下：

    ```
    git clone https://github.com/PaddlePaddle/Paddle.git
    ```
    ```
    cd Paddle
    ```

3. 切换到`develop`分支下进行编译：

    ```
    git checkout develop
    ```

    注意：python3.6、python3.7版本从release/1.2分支开始支持, python3.8版本从release/1.8分支开始支持

4. 创建名为build的目录并进入：

    ```
    mkdir build
    ```
    ```
    cd build
    ```

5. 执行cmake：

    > 具体编译选项含义请参见[编译选项表](../Tables.html#Compile)

    *  编译**CPU版本PaddlePaddle**：

        ```
        cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
        ```

    *  编译**GPU版本PaddlePaddle**：

        ```
        cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
        ```

    默认为Python2，Python3请添加：

    > -DPY_VERSION=3（或3.5、3.6、3.7、3.8）

    如果你的设备信息包含多个Python或CUDA版本，你也可以通过设置路径变量，来指定特定版本的Python或CUDA：

    > -DPYTHON_EXECUTABLE: python的安装目录

    > -DCUDA_TOOLKIT_ROOT_DIR: cuda的安装目录

    例如：（仅作示例，请根据你的设备路径信息进行设置）

    ```
    cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Release -DWITH_GPU=ON -DWITH_TESTING=OFF -DPYTHON_EXECUTABLE=C:\\Python36\\python.exe -DCUDA_TOOLKIT_ROOT_DIR="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\v10.0"
    ```

6. 使用Visual Studio 2015 打开 `paddle.sln` 文件，选择平台为 `x64`，配置为 `Release`，开始编译。

7. 编译成功后进入 `\Paddle\build\python\dist` 目录下找到生成的 `.whl` 包：

    ```
    cd \Paddle\build\python\dist
    ```

8. 安装编译好的 `.whl` 包：

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
    python -m pip uninstall paddlepaddle
    ```

* **GPU版本的PaddlePaddle**:
    ```
    python -m pip uninstall paddlepaddle-gpu
    ```
