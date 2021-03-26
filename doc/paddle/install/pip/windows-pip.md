# Windows下的PIP安装

## 一、环境准备

### 1.1目前飞桨支持的环境

* **Windows 7/8/10 专业版/企业版 (64bit)**
  * **GPU版本支持CUDA 9.0/10.0/10.1/10.2/11.0，且仅支持单卡**

* **Python 版本 2.7.15+/3.5.1+/3.6+/3.7+/3.8+ (64 bit)**

* **pip 版本 20.2.2+ (64 bit)**

### 1.2如何查看您的环境

* 可以使用以下命令查看本机的操作系统和位数信息：

  ```
  uname -m && cat /etc/*release
  ```



* 确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python

  * 如果您是使用 Python 2，使用以下命令输出 Python 路径，根据的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径

    ```
    which python
    ```

  * 如果您是使用 Python 3，使用以下命令输出 Python 路径，根据您的环境您可能需要将说明中所有命令行中的 python3 替换为 python 或者替换为具体的 Python 路径

    ```
    which python3
    ```



* 需要确认python的版本是否满足要求

  * 如果您是使用 Python 2，使用以下命令确认是 2.7.15+

    ```
    python --version
    ```

  * 如果您是使用 Python 3，使用以下命令确认是 3.5.1+/3.6/3.7/3.8

    ```
    python3 --version
    ```

* 需要确认pip的版本是否满足要求，要求pip版本为20.2.2+

  * 如果您是使用 Python2

    ```
    python -m ensurepip
    ```

    ```
    python -m pip --version
    ```

  * 如果您是使用 Python 3

    ```
    python3 -m ensurepip
    ```

    ```
    python3 -m pip --version
    ```


* 需要确认Python和pip是64bit，并且处理器架构是x86_64（或称作x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构。下面的第一行输出的是"64bit"，第二行输出的是"x86_64"、"x64"或"AMD64"即可：

  * 如果您是使用 Python 2

    ```
    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```

  * 如果您是使用 Python 3

    ```
    python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```



* 默认提供的安装包需要计算机支持MKL
* 当前版本暂不支持NCCL，分布式等相关功能

* 如果您对机器环境不了解，请下载使用[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)，配套说明请参考[这里](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。



## 二、开始安装

本文档为您介绍pip安装方式

### 首先请您选择您的版本

* 如果您的计算机没有 NVIDIA® GPU，请安装[CPU版的PaddlePaddle](#cpu)

* 如果您的计算机有NVIDIA® GPU，请确保满足以下条件并且安装GPU版PaddlePaddle

  * **CUDA 工具包9.0/10.0/10.1/10.2 配合 cuDNN v7.6.5+**

  * **CUDA 工具包11.0配合cuDNN v8.0.4**

  * **GPU运算能力超过3.0的硬件设备**

  * 注：目前官方发布的windows安装包仅包含 CUDA 9.0/10.0/10.1/10.2/11.0，不包含 CUDA 9.1/9.2，如需使用，请通过源码自行编译。您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)



### 根据版本进行安装
确定您的环境满足条件后可以开始安装了，选择下面您要安装的PaddlePaddle

#### 2.1 <span id="cpu">CPU版的PaddlePaddle</span>


  ```
  python -m pip install --pre paddlepaddle -f https://paddlepaddle.org.cn/whl/cpu/mkl/develop.html
  ```



#### 2.2<span id="gpu"> GPU版的PaddlePaddle</span>

Windows的develop版本安装只支持CUDA10.2

  ```
  python -m pip install --pre paddlepaddle-gpu==2.1.0.dev0.post102 -f https://paddlepaddle.org.cn/whl/cu102/mkl/develop.html
  ```




注：

* 请确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python。根据您的环境您可能需要将说明中所有命令行中的 python 替换为 python3 或者替换为具体的 Python 路径。


## **三、验证安装**

安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

## **四、如何卸载**

请使用以下命令卸载PaddlePaddle：

* **CPU版本的PaddlePaddle**: `python -m pip uninstall paddlepaddle`

* **GPU版本的PaddlePaddle**: `python -m pip uninstall paddlepaddle-gpu`
