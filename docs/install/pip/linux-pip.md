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

* 如果您对机器环境不了解，请下载使用[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)，配套说明请参考[这里](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。



## 二、开始安装

### 首先请选择您的版本

* 如果您的计算机没有 NVIDIA® GPU，请安装[CPU 版的 PaddlePaddle](#cpu)

* 如果您的计算机有 NVIDIA® GPU，请确保满足以下条件并且安装[GPU 版 PaddlePaddle](#gpu)，依赖库环境版本要求如下：

  * **CUDA 工具包 11.2 配合 cuDNN v8.2.1, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.0.3.4**

  * **CUDA 工具包 11.6 配合 cuDNN v8.4.0, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.4.0.6**

  * **CUDA 工具包 11.7 配合 cuDNN v8.4.1, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.4.2.4**

  * **CUDA 工具包 11.8 配合 cuDNN v8.6.0, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.5.1.7**

  * **CUDA 工具包 12.0 配合 cuDNN v8.9.1, 如需使用 PaddleTensorRT 推理，需配合 TensorRT8.6.1.6**

  * **如需使用分布式多卡环境，需配合 NCCL>=2.7**

  * **GPU 运算能力超过 6.0 的硬件设备**

    您可参考 NVIDIA 官方文档了解 CUDA、CUDNN 和 TensorRT 的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，[TensorRT](https://developer.nvidia.com/tensorrt)

* 如果您需要使用多卡环境请确保您已经正确安装 nccl2，或者按照以下指令安装 nccl2（这里提供的是 CUDA11.2，cuDNN7 下 nccl2 的安装指令，更多版本的安装信息请参考 NVIDIA[官方网站](https://developer.nvidia.com/nccl)）:


    ```
    rm -f /usr/local/lib/libnccl.so
    wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-2.10.3-1+cuda11.4.x86_64.rpm
    wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-devel-2.10.3-1+cuda11.4.x86_64.rpm
    wget --no-check-certificate -q https://nccl2-deb.cdn.bcebos.com/libnccl-static-2.10.3-1+cuda11.4.x86_64.rpm
    rpm -ivh libnccl-2.10.3-1+cuda11.4.x86_64.rpm
    rpm -ivh libnccl-devel-2.10.3-1+cuda11.4.x86_64.rpm
    rpm -ivh libnccl-static-2.10.3-1+cuda11.4.x86_64.rpm
    ```


#### 2.1 <span id="cpu">CPU 版的 PaddlePaddle</span>


  ```
  python3 -m pip install paddlepaddle==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```



#### 2.2 <span id="gpu">GPU 版的 PaddlePaddle</span>



2.2.1 CUDA11.2 的 PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.6.2.post112 -i https://www.paddlepaddle.org.cn/packages/stable/cu112/
  ```


     CUDA11.2 包含 cuDNN 动态链接库的 PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post112 -i https://www.paddlepaddle.org.cn/packages/stable/cudnnin/
  ```


2.2.3 CUDA11.6 的 PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.6.2.post116 -i https://www.paddlepaddle.org.cn/packages/stable/cu116/
  ```


     CUDA11.6 包含 cuDNN 动态链接库的 PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.6.2.post116 -i https://www.paddlepaddle.org.cn/packages/stable/cudnnin/
  ```


2.2.4 CUDA11.7 的 PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post117 -i https://www.paddlepaddle.org.cn/packages/stable/cu117/
  ```


     CUDA11.7 包含 cuDNN 动态链接库的 PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post117 -i https://www.paddlepaddle.org.cn/packages/stable/cudnnin/
  ```


2.2.5 CUDA11.8 的 PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```


     CUDA11.8 包含 cuDNN 动态链接库的 PaddlePaddle


  ```
  python3 -m pip download paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/packages/stable/cudnnin/

  ```


2.2.6 CUDA12.0 的 PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cu120/
  ```


     CUDA12.0 包含 cuDNN 动态链接库的 PaddlePaddle


  ```
  python3 -m pip install paddlepaddle-gpu==2.6.2.post120 -i https://www.paddlepaddle.org.cn/packages/stable/cudnnin/
  ```



## **三、验证安装**

安装完成后您可以使用 `python3` 进入 python 解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。


## **四、如何卸载**

请使用以下命令卸载 PaddlePaddle：

* **CPU 版本的 PaddlePaddle**: `python3 -m pip uninstall paddlepaddle`

* **GPU 版本的 PaddlePaddle**: `python3 -m pip uninstall paddlepaddle-gpu`
