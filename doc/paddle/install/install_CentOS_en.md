# **Install on CentOS**


## Environmental preparation

* **CentOS Version(64 bit)**
    * **CentOS 6 (GPU version supports CUDA 9.0/9.1/9.2/10.0/10.1, only supports single card)**
    * **CentOS 7 (GPU version supports CUDA 9.0/9.1/9.2/10.0/10.1, CUDA 9.1 only supports single card)**
* **Python version 2.7.15+/3.5.1+/3.6/3.7 (64 bit)**
* **pip or pip3 version 9.0.1+ (64 bit)**

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

    * If you are using Python 3, use the following command to confirm that it is 3.5.1+/3.6/3.7

        python3 --version

* It is required to confirm whether the version of pip meets the requirements. The version of pip is required to be 9.0.1+

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

    * **CUDA toolkit 10.0 with cuDNN v7.6+(for multi card support, NCCL2.3.7 or higher)**
    * **CUDA toolkit 9.0 with cuDNN v7.6+(for multi card support, NCCL2.3.7 or higher)**
    * **Hardware devices with GPU computing power over 1.0**

        You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

* 如果您需要使用多卡环境请确保您已经正确安装nccl2，或者按照以下指令安装nccl2（这里提供的是CentOS 7，CUDA9，cuDNN7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl):


        wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        yum update -y
        yum install -y libnccl-2.3.7-2+cuda9.0 libnccl-devel-2.3.7-2+cuda9.0 libnccl-static-2.3.7-2+cuda9.0

## Installation method

There are three installation methods under CentOS system:

* pip installation（recommend）
* [Compile From Source Code](./compile/compile_CentOS_en.html#ct_source)
* [Compile From Docker Source Code](./compile/compile_CentOS_en.html#ct_docker)

Here is pip installation

## Installation steps

* CPU version of PaddlePaddle：
  * For Python 2: `python -m pip install paddlepaddle==2.0.0a0 -i https://mirror.baidu.com/pypi/simple` or `python -m pip install paddlepaddle==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`
  * For Python 3： `python3 -m pip install paddlepaddle==2.0.0a0 -i https://mirror.baidu.com/pypi/simple` or `python3 -m pip install paddlepaddle==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`
* GPU version of PaddlePaddle：
  * For Python 2： `python -m pip install paddlepaddle-gpu==2.0.0a0 -i https://mirror.baidu.com/pypi/simple` 或 `python -m pip install paddlepaddle-gpu==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`
  * For Python 3： `python3 -m pip install paddlepaddle-gpu==2.0.0a0 -i https://mirror.baidu.com/pypi/simple` 或 `python3 -m pip install paddlepaddle-gpu==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`

You can[Verify installation succeeded or not](#check)，if you have any questions, you can refer to [FAQ](./FAQ.html)


Note:

* If it is python2.7, it is recommended to use the `python` command; if it is python3.x, it is recommended to use the 'python3' command


* `python -m pip install paddlepaddle-gpu==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple` This command will install the PaddlePaddle that supports CUDA 10.0 cuDNN v7.


* Download the latest stable installation package by default. For development installation package, please refer to [here](./Tables.html#ciwhls)

<a name="check"></a>
## ***Verify installation***

After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle.fluid` and `fluid.install_check.run_check()`

If `Your Paddle Fluid is installed succesfully!` appears, to verify that the installation was successful.


## ***How to uninstall***

Please use the following command to uninstall PaddlePaddle:

* ***CPU version of PaddlePaddle***: `python -m pip uninstall paddlepaddle` or `python3 -m pip uninstall paddlepaddle`

* ***GPU version of PaddlePaddle***: `python -m pip uninstall paddlepaddle-gpu` or `python3 -m pip uninstall paddlepaddle-gpu`
