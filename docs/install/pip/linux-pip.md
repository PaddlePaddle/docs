# Linux 下的 PIP 安装

## 一、环境准备

### 1.1 目前飞桨支持的环境

* **Linux 版本 (64 bit)**

  * **CentOS 7 (GPU 版本支持 CUDA 10.1/10.2/11.2)**
  * **Ubuntu 16.04 (GPU 版本支持 CUDA 10.1/10.2/11.1/11.2)**
  * **Ubuntu 18.04 (GPU 版本支持 CUDA 10.1/10.2/11.1/11.2)**

* **Python 版本 3.7/3.8/3.9/3.10 (64 bit)**

* **pip 或 pip3 版本 20.2.2 或更高版本 (64 bit)**

### 1.2 如何查看您的环境

* 可以使用以下命令查看本机的操作系统和位数信息：

  ```
  uname -m && cat /etc/*release
  ```



* 确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python

  * 根据您的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径

    ```
    which python
    ```


* 需要确认 python 的版本是否满足要求

  * 使用以下命令确认是 3.7/3.8/3.9/3.10

        python --version

* 需要确认 pip 的版本是否满足要求，要求 pip 版本为 20.2.2 或更高版本

    ```
    python -m ensurepip
    ```

    ```
    python -m pip --version
    ```



* 需要确认 Python 和 pip 是 64bit，并且处理器架构是 x86_64（或称作 x64、Intel 64、AMD64）架构，目前 PaddlePaddle 不支持 arm64 架构。下面的第一行输出的是"64bit"，第二行输出的是"x86_64"、"x64"或"AMD64"即可：


    ```
    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```



* 默认提供的安装包需要计算机支持 MKL

* 如果您对机器环境不了解，请下载使用[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)，配套说明请参考[这里](https://github.com/PaddlePaddle/docs/blob/develop/docs/install/install_script.md)。



## 二、开始安装

本文档为您介绍 pip 安装方式

### 首先请您选择您的版本

* 如果您的计算机没有 NVIDIA® GPU，请安装[CPU 版的 PaddlePaddle](#cpu)

* 如果您的计算机有 NVIDIA® GPU，请确保满足以下条件并且安装[GPU 版 PaddlePaddle](#gpu)

  * **CUDA 工具包 10.1/10.2 配合 cuDNN v7.6+(如需多卡支持，需配合 NCCL2.7 及更高)**

  * **CUDA 工具包 11.2 配合 cuDNN v8.1.1(如需多卡支持，需配合 NCCL2.7 及更高)**

  * **GPU 运算能力超过 3.5 的硬件设备**

    您可参考 NVIDIA 官方文档了解 CUDA 和 CUDNN 的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

* 如果您需要使用多卡环境请确保您已经正确安装 nccl2，或者按照以下指令安装 nccl2（这里提供的是 CUDA9，cuDNN7 下 nccl2 的安装指令，更多版本的安装信息请参考 NVIDIA[官方网站](https://developer.nvidia.com/nccl)）:

  * **CentOS 系统可以参考以下命令**

        wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm

    ```
    rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
    ```

    ```
    yum update -y
    ```

    ```
    yum install -y libnccl-2.3.7-2+cuda9.0 libnccl-devel-2.3.7-2+cuda9.0 libnccl-static-2.3.7-2+cuda9.0
    ```

  * **Ubuntu 系统可以参考以下命令**

    ```
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    ```

    ```
    dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    ```

    ```
    sudo apt-get install -y libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0
    ```


#### 2.1 CPU 版的 PaddlePaddle


  ```
  python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html
  ```



#### 2.2 GPU 版的 PaddlePaddle



2.2.1 CUDA10.1 的 PaddlePaddle

  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
  ```



2.2.2 CUDA10.2 的 PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
  ```

2.2.3 CUDA11.0 的 PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post110 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
  ```

2.2.4 CUDA11.1 的 PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post111 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
  ```


2.2.5 CUDA11.2 的 PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
  ```



注：

* 如果你使用的是安培架构的 GPU，推荐使用 CUDA11.2。如果你使用的是非安培架构的 GPU，推荐使用 CUDA10.2，性能更优。请参考: [GPU 架构对照表](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#nvidia-gpu)

* 请确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python。根据您的环境您可能需要将说明中所有命令行中的 python 替换为 python3 或者替换为具体的 Python 路径。




## **三、验证安装**

安装完成后您可以使用 `python` 或 `python3` 进入 python 解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。


## **四、如何卸载**

请使用以下命令卸载 PaddlePaddle：

* **CPU 版本的 PaddlePaddle**: `python -m pip uninstall paddlepaddle`

* **GPU 版本的 PaddlePaddle**: `python -m pip uninstall paddlepaddle-gpu`
