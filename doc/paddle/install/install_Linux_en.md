# **Install on Linux**


## Environmental preparation

* **Linux Version(64 bit)**
    * **CentOS 6 (GPU version supports CUDA 9.0/9.1/9.2/10.0/10.1/10.2, only supports single card)**
    * **CentOS 7 (GPU version supports CUDA 9.0/9.1/9.2/10.0/10.1/10.2, CUDA 9.1 only supports single card)**
    * **Ubuntu 14.04 (GPU version supports CUDA 10.0/10.1)**
    * **Ubuntu 16.04 (GPU version supports CUDA 9.0/9.1/9.2/10.0/10.1/10.2)**
    * **Ubuntu 18.04 (GPU version supports CUDA 10.0/10.1/10.2)**
* **Python version 2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**
* **pip or pip3 version 20.2.2+ (64 bit)**

### Note

* You can use`uname -m && cat /etc/*release` to view the local operating system and bit information
* Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python

    * If you are using Python 2, use the following command to output Python path. Depending on the environment, you may need to replace Python in all command lines in the description with specific Python path

        which python

    * If you are using Python 3, use the following command to output Python path. Depending on your environment, you may need to replace Python 3 in all command lines in the instructions with Python or specific Python path

        which python3

* You need to confirm whether the version of Python meets the requirements

    * If you are using Python 2, use the following command to confirm that it is 2.7.15+

        python --version

    * If you are using Python 3, use the following command to confirm that it is 3.5.1+/3.6/3.7/3.8

        python3 --version

* It is required to confirm whether the version of pip meets the requirements. The version of pip is required to be 20.2.2+

    * If you are using Python 2

        python -m ensurepip

        python -m pip --version

    * If you are using Python 3

        python3 -m ensurepip

        python3 -m pip --version

* You need to confirm that Python and pip are 64bit, and the processor architecture is x86_64(or called x64、Intel 64、AMD64). Currently, paddlepaddle does not support arm64 architecture. The first line below outputs "64bit", and the second line outputs "x86_64", "x64" or "AMD64"：

    * If you are using Python 2

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

    * If you are using Python 2

        python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

* The installation package provided by default requires computer support for MKL
* If you do not know the machine environment, please download and use[Quick install script](https://fast-install.bj.bcebos.com/fast_install.sh), for instructions please refer to[here](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。

## Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install the CPU version of PaddlePaddle

* If your computer has NVIDIA® GPU, please make sure that the following conditions are met and install the GPU version of PaddlePaddle

    * **CUDA toolkit 9.0/10.0/10.1/10.2 with cuDNN v7.6+(for multi card support, NCCL2.3.7 or higher)**
    * **Hardware devices with GPU computing power over 1.0**


        You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

* If you need to use a multi-card environment, please make sure that you have installed nccl2 correctly, or install nccl2 according to the following instructions (here are the installation instructions of nccl2 under CUDA9 and cuDNN7). For more version installation information, please refer to NVIDIA[Official Website] (https://developer.nvidia.com/nccl):

    * **Centos system can refer to the following commands **

        ```shell
        wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        yum update -y
        yum install -y libnccl-2.3.7-2+cuda9.0 libnccl-devel-2.3.7-2+cuda9.0 libnccl-static-2.3.7-2+cuda9.0
        ```

    * **Ubuntu system can refer to the following commands **

        ```shell
        wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
        dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
        sudo apt-get install -y libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0
        ```

## Installation method

There are three installation methods under CentOS system:

* pip installation（recommend）
* [Compile From Source Code](./compile/compile_CentOS_en.html#ct_source)
* [Compile From Docker Source Code](./compile/compile_CentOS_en.html#ct_docker)

Here is pip installation

## Installation steps

* CPU version of PaddlePaddle：
  * For Python 2: `python -m pip install paddlepaddle==2.0.0rc0 -i https://mirror.baidu.com/pypi/simple` or `python -m pip install paddlepaddle==2.0.0rc0 -i https://pypi.tuna.tsinghua.edu.cn/simple`
  * For Python 3： `python3 -m pip install paddlepaddle==2.0.0rc0 -i https://mirror.baidu.com/pypi/simple` or `python3 -m pip install paddlepaddle==2.0.0rc0 -i https://pypi.tuna.tsinghua.edu.cn/simple`
* GPU version of PaddlePaddle：
  * For Python 2： `python -m pip install paddlepaddle-gpu==2.0.0rc0 -i https://mirror.baidu.com/pypi/simple` 或 `python -m pip install paddlepaddle-gpu==2.0.0rc0 -i https://pypi.tuna.tsinghua.edu.cn/simple`
  * For Python 3： `python3 -m pip install paddlepaddle-gpu==2.0.0rc0 -i https://mirror.baidu.com/pypi/simple` 或 `python3 -m pip install paddlepaddle-gpu==2.0.0rc0 -i https://pypi.tuna.tsinghua.edu.cn/simple`

You can[Verify installation succeeded or not](#check)，if you have any questions, you can refer to [FAQ](./FAQ.html)


Note:

* If it is python2.7, it is recommended to use the `python` command; if it is python3.x, it is recommended to use the 'python3' command


* `python -m pip install paddlepaddle-gpu==2.0.0rc0 -i https://pypi.tuna.tsinghua.edu.cn/simple` This command will install the PaddlePaddle that supports CUDA 10.2 cuDNN v7.


* Download the latest stable installation package by default. For development installation package, please refer to [here](./Tables.html#ciwhls)

<a name="check"></a>
## ***Verify installation***

After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle.fluid` and `fluid.install_check.run_check()`

If `Your Paddle Fluid is installed succesfully!` appears, to verify that the installation was successful.


## ***How to uninstall***

Please use the following command to uninstall PaddlePaddle:

* ***CPU version of PaddlePaddle***: `python -m pip uninstall paddlepaddle` or `python3 -m pip uninstall paddlepaddle`

* ***GPU version of PaddlePaddle***: `python -m pip uninstall paddlepaddle-gpu` or `python3 -m pip uninstall paddlepaddle-gpu`
