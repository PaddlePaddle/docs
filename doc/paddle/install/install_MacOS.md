# **MacOS下安装**

## 环境准备

* **MacOS 版本 10.11/10.12/10.13/10.14 (64 bit) (不支持GPU版本)**
* **Python 版本 2.7.15+/3.5.1+/3.6/3.7 (64 bit)**
* **pip 或 pip3 版本 9.0.1+ (64 bit)**

### 注意事项

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

## 选择CPU/GPU

* 目前在MacOS环境仅支持CPU版PaddlePaddle

## 安装方式

MacOS系统下有3种安装方式：

* pip安装（推荐）
* [源码编译安装](./compile/compile_MacOS.html#mac_source)
* [Docker源码编译安装](./compile/compile_MacOS.html#mac_docker)


这里为您介绍pip安装方式

## 安装步骤

* CPU版PaddlePaddle：
  * 对于Python 2： `python -m pip install paddlepaddle==2.0.0a0 -i https://mirror.baidu.com/pypi/simple` 或 `python -m pip install paddlepaddle==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`
  * 对于Python 3： `python3 -m pip install paddlepaddle==2.0.0a0 -i https://mirror.baidu.com/pypi/simple` 或 `python3 -m pip install paddlepaddle==2.0.0a0 -i https://pypi.tuna.tsinghua.edu.cn/simple`

您可[验证是否安装成功](#check)，如有问题请查看[FAQ](./FAQ.html)

注：

* MacOS上您需要安装unrar以支持PaddlePaddle，可以使用命令`brew install unrar`
* 如果是python2.7, 建议使用`python`命令; 如果是python3.x, 则建议使用`python3`命令


* 默认下载最新稳定版的安装包，如需获取开发版安装包，请参考[这里](./Tables.html#ciwhls)


* 使用MacOS中自带Python可能会导致安装失败。对于**Python2**，建议您使用[Homebrew](https://brew.sh)或[Python.org](https://www.python.org/ftp/python/2.7.15/python-2.7.15-macosx10.9.pkg)提供的python2.7.15；对于**Python3**，请使用[Python.org](https://www.python.org/downloads/mac-osx/)提供的python3.5.x、python3.6.x或python3.7.x。

<a name="check"></a>
## 验证安装
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle.fluid as fluid` ，再输入
 `fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。

## 如何卸载

请使用以下命令卸载PaddlePaddle：

* `python -m pip uninstall paddlepaddle` 或 `python3 -m pip uninstall paddlepaddle`
