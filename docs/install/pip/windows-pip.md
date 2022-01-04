# Windows下的PIP安装

## 一、环境准备

### 1.1目前飞桨支持的环境

* **Windows 7/8/10 专业版/企业版 (64bit)**
* **GPU版本支持CUDA 10.1/10.2/11.0/11.1/11.2，且仅支持单卡**
* **Python 版本 3.6+/3.7+/3.8+/3.9+ (64 bit)**
* **pip 版本 20.2.2或更高版本 (64 bit)**

### 1.2如何查看您的环境

* 需要确认python的版本是否满足要求

  * 使用以下命令确认是 3.6/3.7/3.8/3.9

    ```
    python --version
    ```

* 需要确认pip的版本是否满足要求，要求pip版本为20.2.2或更高版本

    ```
    python -m ensurepip
    ```

    ```
    python -m pip --version
    ```


* 需要确认Python和pip是64bit，并且处理器架构是x86_64（或称作x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构。下面的第一行输出的是"64bit"，第二行输出的是"x86_64"、"x64"或"AMD64"即可：

    ```
    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```


* 默认提供的安装包需要计算机支持MKL
* Windows暂不支持NCCL，分布式等相关功能


## 二、开始安装

本文档为您介绍pip安装方式

### 首先请您选择您的版本

* 如果您的计算机没有 NVIDIA® GPU，请安装[CPU版的PaddlePaddle](#cpu)

* 如果您的计算机有NVIDIA® GPU，请确保满足以下条件并且安装GPU版PaddlePaddle

  * **CUDA 工具包10.1/10.2 配合 cuDNN v7.6.5+**

  * **CUDA 工具包 11.0 配合 cuDNN v8.0.2**

  * **CUDA 工具包 11.1 配合 cuDNN v8.1.1**

  * **CUDA 工具包 11.2 配合 cuDNN v8.2.1**

  * **GPU运算能力超过3.5的硬件设备**

  * 注：目前官方发布的windows安装包仅包含 CUDA 10.1/10.2/11.2，如需使用其他cuda版本，请通过源码自行编译。您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)



### 根据版本进行安装
确定您的环境满足条件后可以开始安装了，选择下面您要安装的PaddlePaddle



#### 2.1 CPU版的PaddlePaddle


  ```
  python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/windows/cpu-mkl-avx/develop.html
  ```

#### 2.2 GPU版的PaddlePaddle



2.2.1 CUDA10.1的PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/windows/gpu/develop.html
  ```


2.2.2 CUDA10.2的PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/windows/gpu/develop.html
  ```


2.2.3 CUDA11.0的PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post110 -f https://www.paddlepaddle.org.cn/whl/windows/gpu/develop.html
  ```


2.2.4 CUDA11.1的PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post111 -f https://www.paddlepaddle.org.cn/whl/windows/gpu/develop.html
  ```


2.2.5 CUDA11.2的PaddlePaddle

  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/windows/gpu/develop.html
  ```


注：

* 如果你使用的是安培架构的GPU，推荐使用CUDA11.2。如果你使用的是非安培架构的GPU，推荐使用CUDA10.2，性能更优。

* 请确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python。根据您的环境，可能需要将上述命令行中所有 `python` 替换为具体的 `Python解释器` 路径（例如C:\Python37\python.exe）。


## **三、验证安装**

安装完成后您可以使用 `python` 进入python解释器，输入`import paddle` ，再输入 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

## **四、如何卸载**

请使用以下命令卸载PaddlePaddle：

* **CPU版本的PaddlePaddle**: `python -m pip uninstall paddlepaddle`

* **GPU版本的PaddlePaddle**: `python -m pip uninstall paddlepaddle-gpu`
