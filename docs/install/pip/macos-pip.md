# MacOS下的PIP安装

## 一、环境准备

### 1.1目前飞桨支持的环境

* **macOS 版本 10.x/11.x (64 bit) (不支持GPU版本)**

* **Python 版本 3.6/3.7/3.8/3.9 (64 bit)**

* **pip 或 pip3 版本 20.2.2或更高版本 (64 bit)**


### 1.2如何查看您的环境

* 可以使用以下命令查看本机的操作系统和位数信息：

  ```
  uname -m && cat /etc/*release
  ```



* 确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python

  * 使用以下命令输出 Python 路径，根据的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径

    ```
    which python
    ```



* 需要确认python的版本是否满足要求

  * 使用以下命令确认是 3.6/3.7/3.8/3.9

    ```
    python --version
    ```

* 需要确认pip的版本是否满足要求，要求pip版本为20.2.2或更高版本


    ```
    python -m ensurepip
    ```

    ```
    python -m pip --version
    ```



* 需要确认Python和pip是64bit，并且处理器架构是x86_64（或称作x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构（mac M1除外，paddle 已支持Mac M1 芯片）。下面的第一行输出的是"64bit"，第二行输出的是"x86_64"、"x64"或"AMD64"即可：

    ```
    python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```



* 默认提供的安装包需要计算机支持MKL

* 如果您对机器环境不了解，请下载使用[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)，配套说明请参考[这里](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。



## 二、开始安装

本文档为您介绍pip安装方式

### 首先请选择您的版本

* 目前在MacOS环境仅支持CPU版PaddlePaddle


### 根据版本进行安装

确定您的环境满足条件后可以开始安装了，选择下面您要安装的PaddlePaddle


  ```
  python -m pip install paddlepaddle==2.3.0 -i https://mirror.baidu.com/pypi/simple
  ```


注:
* MacOS上您需要安装unrar以支持PaddlePaddle，可以使用命令`brew install unrar`
* 请确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python。根据您的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径。
* 默认下载最新稳定版的安装包，如需获取develop 版本nightly build 的安装包，请参考[这里](https://www.paddlepaddle.org.cn/install/quick/zh/1.8.5-windows-pip)
* 使用MacOS中自带Python可能会导致安装失败。请使用[python官网](https://www.python.org/downloads/mac-osx/)提供的python3.6.x、python3.7.x、python3.8.x 或python3.9.x。
* 支持Mac M1芯片的安装。

## **三、验证安装**

安装完成后您可以使用 `python` 进入python解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

## **四、如何卸载**

请使用以下命令卸载PaddlePaddle：

* `python -m pip uninstall paddlepaddle`
