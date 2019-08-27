# **Ubuntu下安装**

## 环境准备

* *Ubuntu 版本 (64 bit)*
    * *Ubuntu 14.04 (GPU 版本支持 CUDA 8.0/10.0)*
    * *Ubuntu 16.04 (GPU 版本支持 CUDA 8.0/9.0/9.1/9.2/10.0)*
    * *Ubuntu 18.04 (GPU 版本支持 CUDA 10.0)*
* *Python 版本 2.7.15+/3.5.1+/3.6/3.7 (64 bit)*
* *pip或pip3 版本 9.0.1+ (64 bit)*

### 注意事项

* 可以使用`uname -m && cat /etc/*release`查看本机的操作系统和位数信息
* 确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python

    * 如果您是使用 Python 2，使用以下命令输出 Python 路径，根据的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径

        which python

    * 如果您是使用 Python 3，使用以下命令输出 Python 路径，根据您的环境您可能需要将说明中所有命令行中的 python3 替换为 python 或者替换为具体的 Python 路径

        which python3 

* 需要确认python的版本是否满足要求

    * 如果您是使用 Python 2，使用以下命令确认是 2.7.15+

        python --version

    * 如果您是使用 Python 3，使用以下命令确认是 3.5.1+/3.6/3.7

        python3 --version

* 需要确认pip的版本是否满足要求，要求pip版本为9.0.1+

    * 如果您是使用 Python 2 

        python -m ensurepip

        python -m pip --version

    * 如果您是使用 Python 3

        python3 -m ensurepip

        python3 -m pip --version

* 需要确认Python和pip是64bit，并且处理器架构是x86_64（或称作x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构。下面的第一行输出的是"64bit"，第二行输出的是"x86_64"、"x64"或"AMD64"即可：

    * 如果您是使用 Python 2

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

    * 如果您是使用 Python 3
    
        python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

* 默认提供的安装包需要计算机支持MKL
* 如果您对机器环境不了解，请下载使用[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)，配套说明请参考[这里](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/beginners_guide/install/install_script.md)。

## 选择CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请安装CPU版的PaddlePaddle

* 如果您的计算机有 NVIDIA® GPU，并且满足以下条件，推荐安装GPU版的PaddlePaddle
	* *CUDA 工具包10.0配合cuDNN v7.3+(如需多卡支持，需配合NCCL2.3.7及更高)*
	* *CUDA 工具包9.0配合cuDNN v7.3+(如需多卡支持，需配合NCCL2.3.7及更高)*
	* *CUDA 工具包8.0配合cuDNN v7.1+(如需多卡支持，需配合NCCL2.1.15-2.2.13）*
	* *GPU运算能力超过1.0的硬件设备*


	您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)
	
* 如果您需要使用多卡环境请确保您已经正确安装nccl2，或者按照以下指令安装nccl2（这里提供的是ubuntu 16.04，CUDA9，cuDNN7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl):


		wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
		dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb`
		sudo apt-get install -y libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0

	

## 安装方式

Ubuntu系统下有5种安装方式：

* pip安装（推荐）
* [conda安装](./install_Conda.html)
* [Docker安装](./install_Docker.html)
* [源码编译安装](./compile/compile_Ubuntu.html#ubt_source)
* [Docker源码编译安装](./compile/compile_Ubuntu.html#ubt_docker)

这里为您介绍pip安装方式

## 安装步骤

* CPU版PaddlePaddle：`python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple` 或 `python3 -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple`


* GPU版PaddlePaddle：`python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple` 或 `python3 -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple`

您可[验证是否安装成功](#check)，如有问题请查看[FAQ](./FAQ.html)

注：

* 如果是python2.7, 建议使用`python`命令; 如果是python3.x, 则建议使用`python3`命令


* `python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple` 此命令将安装支持CUDA 10.0 cuDNN v7的PaddlePaddle，如您对CUDA或cuDNN版本有不同要求，可用`python -m pip install paddlepaddle-gpu==[版本号] -i https://pypi.tuna.tsinghua.edu.cn/simple`或 `python3 -m pip install paddlepaddle-gpu==[版本号] -i https://pypi.tuna.tsinghua.edu.cn/simple`命令来安装，版本号请见[这里](https://pypi.org/project/paddlepaddle-gpu/#history)，关于paddlepaddle与CUDA, cuDNN版本的对应关系请见[安装包列表](./Tables.html/#whls)


* 默认下载最新稳定版的安装包，如需获取开发版安装包，请参考[这里](./Tables.html/#ciwhls)

<a name="check"></a>
## 验证安装
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle.fluid as fluid` ，再输入
 `fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。

## 如何卸载
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `python -m pip uninstall paddlepaddle` 或 `python3 -m pip uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `python -m pip uninstall paddlepaddle-gpu` 或 `python3 -m pip uninstall paddlepaddle-gpu`
