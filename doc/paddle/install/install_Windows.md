# **Windows下安装**

## 环境准备

* **Windows 7/8/10 专业版/企业版 (64bit)**
* **GPU版本支持CUDA 9.0/10.0/10.1/10.2，且仅支持单卡**
* **Python 版本 2.7.15+/3.5.1+/3.6+/3.7+/3.8+ (64 bit)**
* **pip 版本 20.2.2+ (64 bit)**

### 注意事项

* 确认您安装PaddlePaddle的 Python 是您预期的版本，因为您计算机可能有多个 Python，使用以下命令

    python --version

    * 如果您是使用 Python 2，输出应是 2.7.15+

    * 如果您是使用 Python 3，输出应是 3.5.1+/3.6+/3.7+/3.8+

* 如果不符合您预期的版本，使用以下命令查看python的路径是否是您预期的位置

    where python

    * 如果您是使用 Python 2, python2.7的安装目录应位于第一行

    * 如果您是使用 Python 3, python3.5.1+/3.6+/3.7+的安装目录应位于第一行

    * 您可以通过以下任意方法进行调整：

        * 使用具体的Python路径来执行命令（例如C:\Python36\python.exe 或者 C:\Python27\python.exe)  
        * 在环境变量中，将您预期的安装路径设置在第一顺序位（请在控制面板->系统属性->环境变量->PATH中修改)

* 需要确认pip的版本是否满足要求，要求pip版本为20.2.2+

    python -m ensurepip

    python -m pip --version

* 需要确认Python和pip是64bit，并且处理器架构是x86_64（或称作x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构。下面的第一行输出的是"64bit"，第二行输出的是"x86_64"、"x64"或"AMD64"即可：

    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"


* 默认提供的安装包需要计算机支持MKL，如果您的环境不支持，请在[这里](./Tables.html#ciwhls-release)下载`openblas`版本的安装包
* 当前版本暂不支持NCCL，分布式等相关功能

## 选择CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请安装CPU版的PaddlePaddle

* 如果您的计算机有 NVIDIA® GPU，并且满足以下条件，推荐安装GPU版的PaddlePaddle
    * **CUDA 工具包 9.0/10.0/10.1/10.2 配合 cuDNN v7.6.5+**
    * **GPU运算能力超过3.0的硬件设备**

注: 目前官方发布的windows安装包仅包含 CUDA 9.0/10.0/10.1/10.2，不包含 CUDA 9.1/9.2，如需使用，请通过源码自行编译。

您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

## 安装方式

Windows系统下有2种安装方式：

* pip安装（推荐）
* [源码编译安装](./compile/compile_Windows.html#win_source)

这里为您介绍pip安装方式

## 安装步骤

* CPU版PaddlePaddle：
  * `python -m pip install paddlepaddle==2.0.0rc0 -f https://paddlepaddle.org.cn/whl/stable.html`

* GPU版PaddlePaddle：
  * CUDA10.2: `python -m pip install paddlepaddle-gpu==2.0.0rc0 -f https://paddlepaddle.org.cn/whl/stable.html`
  * CUDA10.1: `python -m pip install paddlepaddle-gpu==2.0.0rc0.post101 -f https://paddlepaddle.org.cn/whl/stable.html`
  * CUDA10.0: `python -m pip install paddlepaddle-gpu==2.0.0rc0.post100 -f https://paddlepaddle.org.cn/whl/stable.html`
  * CUDA9.0: `python -m pip install paddlepaddle-gpu==2.0.0rc0.post90 -f https://paddlepaddle.org.cn/whl/stable.html`

您可[验证是否安装成功](#check)，如有问题请查看[FAQ](./FAQ.html)


<a name="check"></a>
## 验证安装
安装完成后您可以使用 `python` 进入python解释器，输入`import paddle.fluid as fluid` ，再输入
 `fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。

## 如何卸载

* **CPU版本的PaddlePaddle**: `python -m pip uninstall paddlepaddle`

* **GPU版本的PaddlePaddle**: `python -m pip uninstall paddlepaddle-gpu`
