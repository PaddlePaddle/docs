# **Windows下安装**

## 环境准备

* *Windows 7/8/10 专业版/企业版 (64bit) (GPU版本支持CUDA 8.0/9.0/10.0，且仅支持单卡)*
* *Python 版本 2.7.15+/3.5.1+/3.6/3.7 (64 bit)*
* *pip 或 pip3 版本 9.0.1+ (64 bit)*

### 注意事项

* 确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python

    * 如果您是使用 Python 2，使用以下命令输出 Python 路径，根据的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径

        where python

    * 如果您是使用 Python 3，使用以下命令输出 Python 路径，根据您的环境您可能需要将说明中所有命令行中的 python3 替换为 python 或者替换为具体的 Python 路径

        where python3 

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

* 默认提供的安装包需要计算机支持MKL，如果您的环境不支持，请在[这里](./Tables.html/#ciwhls-release)下载`openblas`版本的安装包
* 当前版本暂不支持NCCL，分布式等相关功能

## 选择CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请安装CPU版的PaddlePaddle

* 如果您的计算机有 NVIDIA® GPU，并且满足以下条件，推荐安装GPU版的PaddlePaddle
    * *CUDA 工具包8.0配合cuDNN v7.1+，9.0/10.0配合cuDNN v7.3+*
    * *GPU运算能力超过1.0的硬件设备*

注: 目前官方发布的windows安装包仅包含 CUDA 8.0/9.0/10.0 的单卡模式，不包含 CUDA 9.1/9.2/10.1，如需使用，请通过源码自行编译。

您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

## 安装方式

Windows系统下有3种安装方式：

* pip安装（推荐）
* [conda安装](./install_Conda.html)
* [源码编译安装](./compile/compile_Windows.html#win_source)

这里为您介绍pip安装方式

## 安装步骤

* CPU版PaddlePaddle：`python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple` 或 `python3 -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple`


* GPU版PaddlePaddle：`python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple` 或 `python3 -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple`

您可[验证是否安装成功](#check)，如有问题请查看[FAQ](./FAQ.html)

注：

* 如果是python2.7, 建议使用`python`命令; 如果是python3.x, 则建议使用`python3`命令


* `python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple` 此命令将安装支持CUDA 8.0(配合cuDNN v7.1+)或者CUDA 9.0/10.0(配合cuDNN v7.3+)的PaddlePaddle，如您对CUDA或cuDNN版本有不同要求，可用`python -m pip install paddlepaddle-gpu==[版本号] -i https://pypi.tuna.tsinghua.edu.cn/simple`或 `python3 -m pip install paddlepaddle-gpu==[版本号] -i https://pypi.tuna.tsinghua.edu.cn/simple`命令来安装，版本号请见[这里](https://pypi.org/project/paddlepaddle-gpu/#history), 关于paddlepaddle与CUDA, cuDNN版本的对应关系请见[安装包列表](./Tables.html/#whls)


<a name="check"></a>
## 验证安装
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle.fluid as fluid` ，再输入
 `fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。

## 如何卸载

* ***CPU版本的PaddlePaddle***: `python -m pip uninstall paddlepaddle` 或 `python3 -m pip uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `python -m pip uninstall paddlepaddle-gpu` 或 `python3 -m pip uninstall paddlepaddle-gpu`
