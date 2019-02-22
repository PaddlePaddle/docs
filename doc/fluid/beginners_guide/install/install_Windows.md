# **Windows下安装**

## 环境准备

* *64位操作系统*
* *Windows 7/8 ，Windows 10 专业版/企业版*
* *Python 2.7/3.5/3.6/3.7*
* *pip或pip3 >= 9.0.1*

### 注意事项

<<<<<<< HEAD
* 当前版本暂不支持NCCL，分布式相关功能

## 选择CPU/GPU
=======
* 当前版本暂不支持NCCL，分布式等相关功能
>>>>>>> ba77c0bb077b71ace096ec50671b53c3fd951c5c

* 如果您的计算机没有 NVIDIA® GPU，请安装CPU版的PaddlePaddle

* 如果您的计算机有 NVIDIA® GPU，并且满足以下条件，推荐安装GPU版的PaddlePaddle
    * *CUDA 工具包8.0配合cuDNN v7*
    * *GPU运算能力超过1.0的硬件设备*

您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

## 安装方式

Windows系统下有3种安装方式：

* pip安装（推荐）
* [Docker安装](./install_Docker.html)
* [源码编译安装](./compile/compile_Windows.html/#win_source)

这里为您介绍pip安装方式

## 安装步骤

<<<<<<< HEAD
* CPU版PaddlePaddle：`pip install paddlepaddle` 或 `pip3 install paddlepaddle`
* GPU版PaddlePaddle：`pip install paddlepaddle-gpu` 或 `pip3 install paddlepaddle-gpu`

您可[验证是否安装成功](#check)，如有问题请查看[FAQ](./FAQ.html)
=======
* ***CPU版本的PaddlePaddle***:
执行如下命令：`pip install paddlepaddle`(python2.7) 或 `pip3 install paddlepaddle`(python3.x) 安装PaddlePaddle

* ***GPU版本的PaddlePaddle***:
执行如下命令：`pip install paddlepaddle-gpu`(python2.7) 或 `pip3 install paddlepaddle-gpu`(python3.x) 安装PaddlePaddle

## ***验证安装***
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。
>>>>>>> ba77c0bb077b71ace096ec50671b53c3fd951c5c

注：

<<<<<<< HEAD
* pip与python版本对应。如果是python2.7, 建议使用`pip`命令; 如果是python3.x, 则建议使用`pip3`命令
* `pip install paddlepaddle-gpu` 此命令将安装支持CUDA 8.0 cuDNN v7的PaddlePaddle，目前windows环境下暂不支持其他版本的CUDA和cuDNN。

<a name="check"></a>
## 验证安装
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

## 如何卸载

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`
=======
* ***CPU版本的PaddlePaddle***:
请使用以下命令：`pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`  卸载PaddlePaddle

* ***GPU版本的PaddlePaddle***:
请使用以下命令：`pip uninstall paddlepaddle-gpu` 或 `pip3 uninstall paddlepaddle-gpu`  卸载PaddlePaddle
>>>>>>> ba77c0bb077b71ace096ec50671b53c3fd951c5c

* ***GPU版本的PaddlePaddle***: `pip uninstall paddlepaddle-gpu` 或 `pip3 uninstall paddlepaddle-gpu`
