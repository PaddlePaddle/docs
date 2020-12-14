# Linux下的PIP安装

## 一、环境准备

### 1.1目前飞桨支持的环境

* **Linux 版本 (64 bit)**
* **CentOS 6 (GPU版本支持CUDA 9.0/9.1/9.2/10.0/10.1, 仅支持单卡)**

* **CentOS 7 (GPU版本支持CUDA 9.0/9.1/9.2/10.0/10.1, 其中CUDA 9.1仅支持单卡)**
  * **Ubuntu 14.04 (GPU 版本支持 CUDA 10.0/10.1)**
  * **Ubuntu 16.04 (GPU 版本支持 CUDA 9.0/9.1/9.2/10.0/10.1)**
  * **Ubuntu 18.04 (GPU 版本支持 CUDA 10.0/10.1)**

* **Python 版本 2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**

* **pip 或 pip3 版本 20.2.2+ (64 bit)**

### 1.2如何查看您的环境

* 可以使用以下命令查看本机的操作系统和位数信息：

  ```
  uname -m && cat /ect/*release
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


        python --version

  * 如果您是使用 Python 3，使用以下命令确认是 3.5.1+/3.6/3.7/3.8

        python3 --version

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

* 如果您对机器环境不了解，请下载使用[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)，配套说明请参考[这里](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。



## 二、开始安装

本文档为您介绍pip安装方式

### 首先请您选择您的版本

* 如果您的计算机没有 NVIDIA® GPU，请安装[CPU版的PaddlePaddle](#cpu)

* 如果您的计算机有NVIDIA® GPU，请确保满足以下条件并且安装GPU版PaddlePaddle

  * **CUDA 工具包9.0/10.0/10.1配合cuDNN v7.6+(如需多卡支持，需配合NCCL2.3.7及更高)**

  * **GPU运算能力超过1.0的硬件设备**

    您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

* 如果您需要使用多卡环境请确保您已经正确安装nccl2，或者按照以下指令安装nccl2（这里提供的是CUDA9，cuDNN7下nccl2的安装指令，更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl)）:

  * **Centos 系统可以参考以下命令**

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


### 根据版本进行安装
确定您的环境满足条件后可以开始安装了，选择下面您要安装的PaddlePaddle

* [CPU版的PaddlePaddle](#cpu)

* [GPU版的PaddlePaddle](#gpu)
  * [CUDA9.0的PaddlePaddle](#cuda9)
  * [CUDA10.0的PaddlePaddle](#cuda10)
  * 1.8.5版本的CUDA10.1没有wheel包，只能通过源码编译得到。



#### 2.1 <span id="cpu">CPU版的PaddlePaddle</span>

* 如果您是使用 Python 2

  ```
  python -m pip install paddlepaddle==1.8.5 -i https://mirror.baidu.com/pypi/simple
  ```

* 如果您是使用 Python 3

  ```
  python3 -m pip install paddlepaddle==1.8.5 -i https://mirror.baidu.com/pypi/simple
  ```



#### 2.2<span id="gpu"> GPU版的PaddlePaddle</span>



2.2.1 <span id="cuda9">CUDA9.0的PaddlePaddle</span>

* 如果您是使用 Python 2

  ```
  python -m pip install paddlepaddle-gpu==1.8.5.post97 -i https://mirror.baidu.com/pypi/simple
  ```

* 如果您是使用 Python 3

  ```
  python3 -m pip install paddlepaddle-gpu==1.8.5.post97 -i https://mirror.baidu.com/pypi/simple
  ```



2.2.1 <span id="cuda10">CUDA10.0的PaddlePaddle</span>

* 如果您是使用 Python 2

  ```
  python -m pip install paddlepaddle-gpu==1.8.5.post107 -i https://mirror.baidu.com/pypi/simple
  ```

* 如果您是使用 Python 3

  ```
  python3 -m pip install paddlepaddle-gpu==1.8.5.post107 -i https://mirror.baidu.com/pypi/simple
  ```



注：

* 如果是python2.7, 建议使用`python`命令; 如果是python3.x, 则建议使用`python3`命令

* 如果您需要使用清华源，可以通过以下命令

  ```
   python3 -m pip install paddlepaddle-gpu==[版本号] -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```



## **三、验证安装**

安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle.fluid as fluid` ，再输入
 `fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。

## **四、如何卸载**

请使用以下命令卸载PaddlePaddle：

* **CPU版本的PaddlePaddle**: `python -m pip uninstall paddlepaddle` 或 `python3 -m pip uninstall paddlepaddle`

* **GPU版本的PaddlePaddle**: `python -m pip uninstall paddlepaddle-gpu` 或 `python3 -m pip uninstall paddlepaddle-gpu`
