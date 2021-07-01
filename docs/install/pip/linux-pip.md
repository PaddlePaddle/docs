# Linux下的PIP安装

## 一、环境准备

### 1.1目前飞桨支持的环境

* **Linux 版本 (64 bit)**

  * **CentOS 7 (GPU版本支持CUDA 10.1/10.2/11.2)**
  * **Ubuntu 16.04 (GPU 版本支持 CUDA 10.1/10.2/11.2)**
  * **Ubuntu 18.04 (GPU 版本支持 CUDA 10.1/10.2/11.2)**

* **Python 版本 3.6/3.7/3.8/3.9 (64 bit)**

* **pip 或 pip3 版本 20.2.2或更高版本 (64 bit)**

### 1.2如何查看您的环境

* 可以使用以下命令查看本机的操作系统和位数信息：

  ```
  uname -m && cat /etc/*release
  ```



* 确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python

  * 根据您的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径

    ```
    which python
    ```


* 需要确认python的版本是否满足要求

  * 使用以下命令确认是 3.6/3.7/3.8/3.9

        python --version

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

* 如果您对机器环境不了解，请下载使用[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)，配套说明请参考[这里](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。



## 二、开始安装

本文档为您介绍pip安装方式

### 首先请您选择您的版本

* 如果您的计算机没有 NVIDIA® GPU，请安装[CPU版的PaddlePaddle](#cpu)

* 如果您的计算机有NVIDIA® GPU，请确保满足以下条件并且安装[GPU版PaddlePaddle](#gpu)

  * **CUDA 工具包10.1/10.2配合cuDNN v7.6.5(cuDNN版本>=7.6.5 <8, 如需多卡支持，需配合NCCL2.7及更高)**

  * **CUDA 工具包11.2配合cuDNN v8.1.1(如需多卡支持，需配合NCCL2.7及更高)**

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


#### 2.1 CPU版的PaddlePaddle


  ```
  python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
  ```



#### 2.2 GPU版的PaddlePaddle



2.2.1 CUDA10.1的PaddlePaddle

  ```
  python -m pip install paddlepaddle-gpu==2.1.1.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
  ```



2.2.2 CUDA10.2的PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
  ```

2.2.3 CUDA11.0的PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==2.1.1.post110 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
  ```


2.2.4 CUDA11.2的PaddlePaddle


  ```
  python -m pip install paddlepaddle-gpu==2.1.1.post112 -f https://paddlepaddle.org.cn/whl/mkl/stable.html
  ```



注：

* 如果你使用的是安培架构的GPU，推荐使用CUDA11.2。如果你使用的是非安培架构的GPU，推荐使用CUDA10.2，性能更优。

* 请确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python。根据您的环境您可能需要将说明中所有命令行中的 python 替换为 python3 或者替换为具体的 Python 路径。

* 如果您需要使用清华源，可以通过以下命令

  ```
   python -m pip install paddlepaddle-gpu==[版本号] -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

* 上述命令默认安装`avx`的包。如果你的机器不支持`avx`，需要安装`noavx`的Paddle包，可以通过以下命令安装，仅支持python3.8：

  * cpu、mkl版本noavx机器安装：

  ```
  python -m pip install paddlepaddle==2.1.1 -f http://www.paddlepaddle.org.cn/whl/mkl/stable/noavx/html --no-index
  ```

  * cpu、openblas版本noavx机器安装：

  ```
  python -m pip install paddlepaddle==2.1.1 -f https://www.paddlepaddle.org.cn/whl/openblas/stable/noavx.html --no-index
  ```


  * gpu版本cuda10.1安装：

  ```
  python -m pip install paddlepaddle-gpu==2.1.1.post101 -f https://www.paddlepaddle.org.cn/whl/mkl/stable/noavx.html
  ```

  * gpu版本cuda10.2安装：

  ```
  python -m pip install paddlepaddle-gpu==2.1.1 -f https://www.paddlepaddle.org.cn/whl/mkl/stable/noavx.html --no-index
  ```

  判断你的机器是否支持`avx`，可以输入以下命令，如果输出中包含`avx`，则表示机器支持`avx`
  ```
  cat /proc/cpuinfo | grep -i avx
  ```

* 如果你想安装联编`tensorrt`的Paddle包，可以通过以下命令

  ```
  python -m pip install paddlepaddle-gpu==[版本号] -f https://paddlepaddle.org.cn/whl/stable/tensorrt.html
  ```

* 如果你想安装`openblas`的Paddle包，可以通过以下命令

  ```
  python -m pip install paddlepaddle-gpu==[版本号] -f https://paddlepaddle.org.cn/whl/openblas/stable.html
  ```




## **三、验证安装**

安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。


## **四、如何卸载**

请使用以下命令卸载PaddlePaddle：

* **CPU版本的PaddlePaddle**: `python -m pip uninstall paddlepaddle`

* **GPU版本的PaddlePaddle**: `python -m pip uninstall paddlepaddle-gpu`
