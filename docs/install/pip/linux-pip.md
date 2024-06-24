# Linux 下的 PIP 安装

[The Python Package Index(PyPI)](https://pypi.org/)是 Python 的包管理器。本文档为你介绍 PyPI 安装方式，飞桨提供的 PyPI 安装包支持分布式训练（多机多卡）、TensorRT 推理功能。

## 一、环境准备

### 1.1 如何查看您的环境

* 可以使用以下命令查看本机的操作系统和位数信息：

  ```
  uname -m && cat /etc/*release
  ```



* 确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python

  * 根据您的环境您可能需要将说明中所有命令行中的 python3 替换为具体的 Python 路径

    ```
    which python3
    ```


* 需要确认 python 的版本是否满足要求

  * 使用以下命令确认是 3.8/3.9/3.10/3.11/3.12

        python3 --version

* 需要确认 pip 的版本是否满足要求，要求 pip 版本为 20.2.2 或更高版本

    ```
    python3 -m ensurepip
    ```

    ```
    python3 -m pip --version
    ```



* 需要确认 Python 和 pip 是 64bit，并且处理器架构是 x86_64（或称作 x64、Intel 64、AMD64）架构。下面的第一行输出的是"64bit"，第二行输出的是"x86_64"、"x64"或"AMD64"即可：


    ```
    python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```



* 默认提供的安装包需要计算机支持 MKL

* 如果您对机器环境不了解，请下载使用[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)，配套说明请参考[这里](https://github.com/PaddlePaddle/FluidDoc/tree/develop/docs/install/install_script.md)。



## 二、开始安装

### 首先请选择您的版本

* 如果您的计算机没有 NVIDIA® GPU，请安装[CPU 版的 PaddlePaddle](#cpu)

* 如果您的计算机有 NVIDIA® GPU，请确保满足以下条件并且安装[GPU 版 PaddlePaddle](#gpu)，依赖库环境版本要求如下：

  * **CUDA 工具包 11.8 配合 cuDNN v8.6.0, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.5.3.1**

  * **CUDA 工具包 12.3 配合 cuDNN v9.0.0, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.6.1.6**

  * **GPU 运算能力超过 6.0 的硬件设备**

    您可参考 NVIDIA 官方文档了解 CUDA、CUDNN 和 TensorRT 的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，[TensorRT](https://developer.nvidia.com/tensorrt)



#### 2.1 <span id="cpu">CPU 版的 PaddlePaddle</span>


  ```
  python3 -m pip install paddlepaddle==3.0.0b0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  ```


#### 2.2 <span id="gpu">GPU 版的 PaddlePaddle</span>


2.2.1 CUDA11.8 的 PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==3.0.0b0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
  ```


2.2.2 CUDA12.3 的 PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==3.0.0b0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
  ```


注：

* 飞桨对于主流各 python 版本均提供了对应的安装包，而您环境中可能有多个 Python，请确认你想使用的 python 版本并下载对应的 paddlepaddle 安装包。例如您想使用 python3.10 的环境，则安装命令为 python3.10 -m pip install paddlepaddle。

* 上述命令默认安装`avx`、`mkl`的包。判断你的机器是否支持`avx`，可以输入以下命令，如果输出中包含`avx`，则表示机器支持`avx`。飞桨不再支持`noavx`指令集的安装包。
  ```
  cat /proc/cpuinfo | grep -i avx
  ```

* 如果你想安装`avx`、`openblas`的 Paddle 包，可以通过以下命令将 wheel 包下载到本地，再使用`python3 -m pip install [name].whl`本地安装（[name]为 wheel 包名称）：

  ```
  python3 -m pip install https://paddle-wheel.bj.bcebos.com/3.0.0-beta0/linux/linux-cpu-openblas-avx/paddlepaddle-3.0.0b0-cp38-cp38-linux_x86_64.whl
  ```


## **三、验证安装**

安装完成后您可以使用 `python3` 进入 python 解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。


## **四、如何卸载**

请使用以下命令卸载 PaddlePaddle：

* **CPU 版本的 PaddlePaddle**: `python3 -m pip uninstall paddlepaddle`

* **GPU 版本的 PaddlePaddle**: `python3 -m pip uninstall paddlepaddle-gpu`
