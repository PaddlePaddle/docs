# **Windows下安装**

## 环境准备

* *Windows 7/8/10 专业版/企业版 (64bit) (GPU版本支持CUDA 8.0/9.0，且仅支持单卡)*
* *Python 版本 2.7.15+/3.5.1+/3.6/3.7 (64 bit)*
* *pip 或 pip3 版本 9.0.1+ (64 bit)*

### 注意事项

* 默认提供的安装包需要计算机支持AVX指令集和MKL，如果您的环境不支持，请在[这里](./Tables.html/#ciwhls-release)下载`openblas`版本的安装包
* 当前版本暂不支持NCCL，分布式等相关功能

## 选择CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请安装CPU版的PaddlePaddle

* 如果您的计算机有 NVIDIA® GPU，并且满足以下条件，推荐安装GPU版的PaddlePaddle
    * *CUDA 工具包8.0配合cuDNN v7.1+， 9.0配合cuDNN v7.3+*
    * *GPU运算能力超过1.0的硬件设备*

注: 目前官方发布的windows安装包仅包含 CUDA 8.0/9.0 的单卡模式，不包含 CUDA 9.1/9.2/10.0/10.1，如需使用，请通过源码自行编译。

您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

## 安装方式

Windows系统下有3种安装方式：

* pip安装（推荐）
* [Docker安装](./install_Docker.html)
* [源码编译安装](./compile/compile_Windows.html#win_source)

这里为您介绍pip安装方式

## 安装步骤

* CPU版PaddlePaddle：`pip install -U paddlepaddle` 或 `pip3 install -U paddlepaddle`


* GPU版PaddlePaddle：`pip install -U paddlepaddle-gpu` 或 `pip3 install -U paddlepaddle-gpu`

您可[验证是否安装成功](#check)，如有问题请查看[FAQ](./FAQ.html)

注：

* pip与python版本对应。如果是python2.7, 建议使用`pip`命令; 如果是python3.x, 则建议使用`pip3`命令


* `pip3 install -U paddlepaddle-gpu` 此命令将安装支持CUDA 8.0/9.0 cuDNN v7.3+的PaddlePaddle，如您对CUDA或cuDNN版本有不同要求，可用`pip install -U paddlepaddle-gpu==[版本号]`或 `pip3 install -U paddlepaddle-gpu==[版本号]`命令来安装，版本号请见[这里](https://pypi.org/project/paddlepaddle-gpu/#history), 关于paddlepaddle与CUDA, cuDNN版本的对应关系请见[安装包列表](./Tables.html/#whls)


<a name="check"></a>
## 验证安装
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle.fluid as fluid` ，再输入
 `fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。

## 如何卸载

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall paddlepaddle-gpu` 或 `pip3 uninstall paddlepaddle-gpu`
