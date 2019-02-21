# **CentOS下安装**

## 环境准备

* *64位操作系统*
* *CentOS 6 / 7*
* *Python 2.7/3.5/3.6/3.7*
* *pip或pip3 >= 9.0.1*

### 注意事项

* 可以使用`uname -m && cat /etc/*release`查看本机的操作系统和位数信息
* 可以使用`pip -V`(Python版本为2.7)或`pip3 -V`(Python版本为3.5/3.6/3.7)，确认pip/pip3版本是否满足要求
* 默认提供的安装包需要计算机支持AVX指令集。可使用`cat /proc/cpuinfo | grep avx`来检测您的处理器是否支持该指令集，如不支持，请在[这里](./Tables.html/#ciwhls-release)下载`no_avx`版本的安装包

## 选择CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请安装CPU版本的PaddlePaddle

* 如果您的计算机有NVIDIA® GPU，并且满足以下条件，推荐安装GPU版PaddlePaddle
    * *CUDA 工具包9.0配合cuDNN v7*
    * *CUDA 工具包8.0配合cuDNN v7*
    * *CUDA 工具包8.0配合cuDNN v5*
    * *GPU运算能力超过1.0的硬件设备*

您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

## 安装方式

CentOS系统下有4种安装方式：

* pip安装（推荐）
* [Docker安装](./install_Docker.html)
* [源码编译安装](./compile/compile_CentOS.html/#ct_source)
* [Docker源码编译安装](./compile/compile_CentOS.html/#ct_docker)

这里为您介绍pip安装方式

## 安装步骤

* CPU版PaddlePaddle：`pip install paddlepaddle` 或 `pip3 install paddlepaddle`
* GPU版PaddlePaddle：`pip install paddlepaddle-gpu` 或 `pip3 install paddlepaddle-gpu`

您可[验证是否安装成功](#check)，如有问题请查看[FAQ](./FAQ.html)

注：

* pip与python版本对应。如果是python2.7, 建议使用`pip`命令; 如果是python3.x, 则建议使用`pip3`命令
* `pip install paddlepaddle-gpu` 此命令将安装支持CUDA 9.0 cuDNN v7的PaddlePaddle，如您对CUDA或cuDNN版本有不同要求，可用`pip install paddlepaddle==[版本号]`或 `pip3 install paddlepaddle==[版本号]`命令来安装，版本号请见[这里](https://pypi.org/project/paddlepaddle-gpu/#history)
* 默认下载最新稳定版的安装包，如需获取开发版安装包，请参考[这里](./Tables.html/#ciwhls)

<a name="check"></a>
## ***验证安装***
安装完成后您可以使用命令`python` 或 `python3` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

## ***如何卸载***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall paddlepaddle-gpu` 或 `pip3 uninstall paddlepaddle-gpu`
