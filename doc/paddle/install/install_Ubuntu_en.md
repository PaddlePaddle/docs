# **Install on Ubuntu**

## Environment preparation

* **Ubuntu version (64 bit)**
    * **Ubuntu 14.04 (GPU version supports CUDA 10.0/10.1)**
    * **Ubuntu 16.04 (GPU version supports CUDA 9.0/9.1/9.2/10.0/10.1)**
    * **Ubuntu 18.04 (GPU version supports CUDA 10.0/10.1)**
* **Python version 2.7.15+/3.5.1+/3.6/3.7 (64 bit)**
* **pip or pip3 version 9.0.1+ (64 bit)**

### Note


* You can use `uname -m && cat /etc/*release` view the operating system and digit information of the machine
* Confirm that the Python where you need to install PaddlePaddle is your expected location, because your computer may have multiple Python

    * If you are using Python 2, use the following command to output Python path. Depending on the environment, you may need to replace Python in all command lines in the description with specific Python path

        which python

    * If you are using Python 3, use the following command to output Python path. Depending on the environment, you may need to replace Python 3 in all command lines in the description with Python or specific Python path

        which python3

* You need to confirm that the version of Python meets the requirements
    * If you are using Python 2，use the following command to confirm it is 2.7.15+

        python --version

    * If you are using Python 3，use the following command to confirm it is 3.5.1+/3.6/3.7

        python3 --version

* You need to confirm that the version of pip meets the requirements, pip version is required 9.0.1+

    * If you are using Python 2

        python -m ensurepip

        python -m pip --version

    * If you are using Python 3

        python3 -m ensurepip

        python3 -m pip --version

* Confirm that Python and pip is 64 bit，and the processor architecture is x86_64(or called x64、Intel 64、AMD64)architecture. Currently, PaddlePaddle doesn't support arm64 architecture. The first line below outputs "64bit", and the second line outputs "x86_64", "x64" or "AMD64" :

    * If you are using Python 2

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

    * If you are using Python 3

        python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

* The installation package provided by default requires computer support for MKL
* If you do not know the machine environment, please download and use[Quick install script](https://fast-install.bj.bcebos.com/fast_install.sh), please refer to[here](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md).

## Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install CPU version of PaddlePaddle

* If your computer has NVIDIA® GPU, and meet the following conditions, we command you to install PaddlePaddle
    * **CUDA toolkit 10.0 with cuDNN v7.6+(for multi card support, NCCL2.3.7 or higher)**
    * **CUDA toolkit 9.0 with cuDNN v7.6+(for multi card support, NCCL2.3.7 or higher)**
    * **Hardware devices with GPU computing power over 1.0**


    You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

* If you need to use multi card environment, please make sure that you have installed nccl2 correctly, or install nccl2 according to the following instructions (here is the installation instructions of nccl2 under ubuntu 16.04, CUDA9 and cuDNN7). For more version of installation information, please refer to NVIDIA[official website](https://developer.nvidia.com/nccl):


    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
    sudo apt-get install -y libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0



## Choose an installation method

Under the Ubuntu system, we offer 3 installation methods:

* Pip installation (recommended)
* [Source code compilation and installation](./compile/compile_Ubuntu.html#ubt_source)
* [Docker source code compilation and installation](./compile/compile_Ubuntu.html#ubt_docker)

We will introduce pip installation here.

## Installation steps

* CPU version of PaddlePaddle：
  * For Python 2： `python -m pip install paddlepaddle==2.0.0a0 -i https://mirror.baidu.com/pypi/simple` or `python -m pip install paddlepaddle==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`
  * For Python 3： `python3 -m pip install paddlepaddle==2.0.0a0 -i https://mirror.baidu.com/pypi/simple` or `python3 -m pip install paddlepaddle==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`

* GPU version PaddlePaddle：
  * For Python 2： `python -m pip install paddlepaddle-gpu==2.0.0a0 -i https://mirror.baidu.com/pypi/simple` or `python -m pip install paddlepaddle-gpu==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`
  * For Python 3： `python3 -m pip install paddlepaddle-gpu==2.0.0a0 -i https://mirror.baidu.com/pypi/simple` or `python3 -m pip install paddlepaddle-gpu==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`

You can [verify whether the installation is successful](#check), if you have any questions please see [FAQ](./FAQ.html)

Note:

* For python2.7, we recommend to use `python` command; For python3.x, we recommend to use `python3` command.

* `python -m pip install paddlepaddle-gpu==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple` This command will install PaddlePaddle supporting CUDA 10.0 cuDNN v7.


* Download the latest stable installation package by default. For development installation package, please refer to[here](./Tables.html#ciwhls)

<a name="check"></a>
<br/><br/>
## ***Verify installation***

    After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle.fluid as fluid` and then  `fluid.install_check.run_check()` to verify that the installation was successful.

    If `Your Paddle Fluid is installed succesfully!` appears, it means the installation was successful.

<br/><br/>
## ***How to uninstall***

Please use the following command to uninstall PaddlePaddle (users who use Docker to install PaddlePaddle should use the following command in the container containing PaddlePaddle. Please use the corresponding version of pip):

* ***CPU version of PaddlePaddle***: `pip uninstall paddlepaddle` or `pip3 uninstall paddlepaddle`

- ***GPU version of PaddlePaddle***: `pip uninstall paddlepaddle-gpu` or `pip3 uninstall paddlepaddle-gpu`
