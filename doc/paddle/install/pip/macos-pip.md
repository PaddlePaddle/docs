# MacOS下的PIP安装

## 一、环境准备

### 1.1目前飞桨支持的环境

* **macOS 版本 10.11/10.12/10.13/10.14 (64 bit) (不支持GPU版本)**

* **Python 版本 2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**

* **pip 或 pip3 版本 20.2.2+ (64 bit)**


### 1.2如何查看您的环境

* 可以使用以下命令查看本机的操作系统和位数信息：

  ```
  uname -m && cat /ect/*release
  ```



* 确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python

  * 如果您是使用 Python 2，使用以下命令输出 Python 路径，根据的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径

    ```
    which python
    ```

  * 如果您是使用 Python 3，使用以下命令输出 Python 路径，根据您的环境您可能需要将说明中所有命令行中的 python3 替换为 python 或者替换为具体的 Python 路径

    ```
    which python3
    ```



* 需要确认python的版本是否满足要求

  * 如果您是使用 Python 2，使用以下命令确认是 2.7.15+

    ```
    python --version
    ```

  * 如果您是使用 Python 3，使用以下命令确认是 3.5.1+/3.6/3.7/3.8

    ```
    python3 --version
    ```

* 需要确认pip的版本是否满足要求，要求pip版本为20.2.2+

  * 如果您是使用 Python2

    ```
    python -m ensurepip
    ```

    ```
    python -m pip --version
    ```

  * 如果您是使用 Python 3

    ```
    python3 -m ensurepip
    ```

    ```
    python3 -m pip --version
    ```



* 需要确认Python和pip是64bit，并且处理器架构是x86_64（或称作x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构。下面的第一行输出的是"64bit"，第二行输出的是"x86_64"、"x64"或"AMD64"即可：

  * 如果您是使用 Python 2

    ```
    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```

  * 如果您是使用 Python 3

    ```
    python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```



* 默认提供的安装包需要计算机支持MKL

* 如果您对机器环境不了解，请下载使用[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)，配套说明请参考[这里](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。



## 二、开始安装

本文档为您介绍pip安装方式

### 首先请您选择您的版本

* 目前在MacOS环境仅支持CPU版PaddlePaddle


### 根据版本进行安装

确定您的环境满足条件后可以开始安装了，选择下面您要安装的PaddlePaddle

* 对于Python 2,您可以执行:

  ```
  python -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple
  ```

* 对于Python 3，您可以执行:

  ```
  python3 -m pip install paddlepaddle==2.0.0 -i https://mirror.baidu.com/pypi/simple
  ```

* 注:
* MacOS上您需要安装unrar以支持PaddlePaddle，可以使用命令`brew install unrar`
* 如果是python2.7, 建议使用`python`命令;如果是python3.x, 则建议使用`python3`命令
* 默认下载最新稳定版的安装包，如需获取开发版安装包，请参考[这里](https://www.paddlepaddle.org.cn/install/quick/zh/1.8.5-windows-pip)
* 使用MacOS中自带Python可能会导致安装失败。对于Python2，建议您使用[Homebrew](https://brew.sh)或[Python.org](https://www.paddlepaddle.org.cn/install/quick/zh/2.0rc-macos-pip)提供的python2.7.15；对于Python3，请使用[Python.org](https://www.python.org/downloads/mac-osx/)提供的python3.5.x、python3.6.x、python3.7.x或python3.8.x。

## **三、验证安装**

安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

## **四、如何卸载**

请使用以下命令卸载PaddlePaddle：

* `python -m pip uninstall paddlepaddle` 或 `python3 -m pip uninstall paddlepaddle`
