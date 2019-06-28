# **CentOS下安装**

## 环境准备

* *64位操作系统*
* *CentOS 6 / 7*
* *Python（64 bit） 2.7/3.5.1+/3.6/3.7*
* *pip或pip3 >= 9.0.1*

### 注意事项

* 可以使用`uname -m && cat /etc/*release`查看本机的操作系统和位数信息
* 可以使用`pip -V`(Python版本为2.7)或`pip3 -V`(Python版本为3.5/3.6/3.7)，确认pip/pip3版本是否满足要求
* 如果您对机器环境不了解，请下载使用[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)，配套说明请参考[这里](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/beginners_guide/install/install_script.md)。

## 选择CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请安装CPU版本的PaddlePaddle

* 如果您的计算机有NVIDIA® GPU，请确保满足以下条件并且安装GPU版PaddlePaddle
	
	* *CUDA 工具包10.0配合cuDNN v7.3+(如需多卡支持，需配合NCCL2.3.7及更高)*
	* *CUDA 工具包9.0配合cuDNN v7.3+(如需多卡支持，需配合NCCL2.3.7及更高)*
	* *CUDA 工具包8.0配合cuDNN v7.3+(官方不支持多卡）*
	* *GPU运算能力超过1.0的硬件设备*

		您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

* 如果您需要使用多卡环境请确保您已经正确安装nccl2，或者按照以下指令安装nccl2（这里提供的是CentOS 7，CUDA9，cuDNN7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl/nccl-download):


		wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
		rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
		yum update -y
		yum install -y libnccl-2.3.7-2+cuda9.0 libnccl-devel-2.3.7-2+cuda9.0 libnccl-static-2.3.7-2+cuda9.0


## 安装方式

CentOS系统下有4种安装方式：

* pip安装（推荐）
* [Docker安装](./install_Docker.html)
* [源码编译安装](./compile/compile_CentOS.html/#ct_source)
* [Docker源码编译安装](./compile/compile_CentOS.html/#ct_docker)

这里为您介绍pip安装方式

## 安装步骤

* CPU版PaddlePaddle：`pip install -U paddlepaddle` 或 `pip3 install -U paddlepaddle`
* GPU版PaddlePaddle：`pip install -U paddlepaddle-gpu` 或 `pip3 install  -U paddlepaddle-gpu`

您可[验证是否安装成功](#check)，如有问题请查看[FAQ](./FAQ.html)

注：

* pip与python版本对应。如果是python2.7, 建议使用`pip`命令; 如果是python3.x, 则建议使用`pip3`命令
* `pip install -U paddlepaddle-gpu` 此命令将安装支持CUDA 9.0 cuDNN v7的PaddlePaddle，如您对CUDA或cuDNN版本有不同要求，可用`pip install -U paddlepaddle==[版本号]`或 `pip3 install -U paddlepaddle==[版本号]`命令来安装，版本号请见[这里](https://pypi.org/project/paddlepaddle-gpu/#history)
* 默认下载最新稳定版的安装包，如需获取开发版安装包，请参考[这里](./Tables.html/#ciwhls)

<a name="check"></a>
## ***验证安装***
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle.fluid as fluid` ，再输入
 `fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。

## ***如何卸载***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall paddlepaddle-gpu` 或 `pip3 uninstall paddlepaddle-gpu`
