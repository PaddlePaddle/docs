# Windows 下的 PIP 安装

[The Python Package Index(PyPI)](https://pypi.org/)是 Python 的包管理器。本文档为你介绍 PyPI 安装方式，飞桨提供的 PyPI 安装包支持 TensorRT 推理功能。

## 一、环境准备

### 1.1 如何查看您的环境

* 需要确认 python 的版本是否满足要求

  * 使用以下命令确认是 3.8/3.9/3.10/3.11/3.12

    ```
    python --version
    ```

* 需要确认 pip 的版本是否满足要求，要求 pip 版本为 20.2.2 或更高版本

    ```
    python -m ensurepip
    ```

    ```
    python -m pip --version
    ```


* 需要确认 Python 和 pip 是 64bit，并且处理器架构是 x86_64（或称作 x64、Intel 64、AMD64）架构。下面的第一行输出的是"64bit"，第二行输出的是"x86_64"、"x64"或"AMD64"即可：

    ```
    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```


* 默认提供的安装包需要计算机支持 MKL
* Windows 暂不支持 NCCL，分布式等相关功能


## 二、开始安装

本文档为您介绍 pip 安装方式

### 首先请您选择您的版本

* 如果您的计算机没有 NVIDIA® GPU，请安装[CPU 版的 PaddlePaddle](#cpu)

* 如果您的计算机有 NVIDIA® GPU，请确保满足以下条件并且安装 GPU 版 PaddlePaddle

  * **CUDA 工具包 11.8 配合 cuDNN v8.6.0，如需使用 PaddleTensorRT 推理，需配合 TensorRT8.5.1.7**

  * **CUDA 工具包 12.3 配合 cuDNN v9.0.0, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.6.1.6**

  * **GPU 运算能力超过 6.0 的硬件设备**

  * 注：目前官方发布的 windows 安装包仅包含 CUDA 11.2/11.6/11.7/11.8/12.0，如需使用其他 cuda 版本，请通过源码自行编译。您可参考 NVIDIA 官方文档了解 CUDA、CUDNN 和 TensorRT 的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，[TensorRT](https://developer.nvidia.com/tensorrt)



### 根据版本进行安装
确定您的环境满足条件后可以开始安装了，选择下面您要安装的 PaddlePaddle



#### 2.1 <span id="cpu">CPU 版的 PaddlePaddle</span>


  ```
  python -m pip install paddlepaddle==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  ```


#### 2.2 <span id="gpu">GPU 版的 PaddlePaddle</span>


2.2.4 CUDA11.8 的 PaddlePaddle

  ```
  python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
  ```

2.2.5 CUDA12.3 的 PaddlePaddle

  ```
  python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
  ```

注：

* 请确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python。根据您的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径。

* 上述命令默认安装`avx`、`mkl`的包。判断你的机器是否支持`avx`，可以安装[CPU-Z](https://www.cpuid.com/softwares/cpu-z.html)工具查看“处理器-指令集”。飞桨不再支持`noavx`指令集的安装包。

* 如果你想安装`avx`、`openblas`的 Paddle 包，可以通过以下命令将 wheel 包下载到本地，再使用`python -m pip install [name].whl`本地安装（[name]为 wheel 包名称）：

  ```
  python -m pip install https://paddle-wheel.bj.bcebos.com/3.0.0-beta0/windows/windows-cpu-avx-openblas-vs2017/paddlepaddle-3.0.0b1-cp38-cp38-win_amd64.whl
  ```




## **三、验证安装**

安装完成后您可以使用 `python` 进入 python 解释器，输入`import paddle` ，再输入 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

## **四、如何卸载**

请使用以下命令卸载 PaddlePaddle：

* **CPU 版本的 PaddlePaddle**: `python -m pip uninstall paddlepaddle`

* **GPU 版本的 PaddlePaddle**: `python -m pip uninstall paddlepaddle-gpu`
