# MacOS 下的 PIP 安装

[The Python Package Index(PyPI)](https://pypi.org/)是 Python 的包管理器。本文档为你介绍 PyPI 安装方式，飞桨提供的 PyPI 安装包支持 TensorRT 推理功能。

## 一、环境准备

### 1.1 如何查看您的环境

* 可以使用以下命令查看本机的操作系统和位数信息：

  ```
  uname -m && cat /etc/*release
  ```



* 确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python

  * 使用以下命令输出 Python 路径，根据的环境您可能需要将说明中所有命令行中的 python3 替换为具体的 Python 路径

    ```
    which python
    ```



* 需要确认 python 的版本是否满足要求

  * 使用以下命令确认是 3.8/3.9/3.10/3.11/3.12

    ```
    python3 --version
    ```

* 需要确认 pip 的版本是否满足要求，要求 pip 版本为 20.2.2 或更高版本


    ```
    python3 -m ensurepip
    ```

    ```
    python3 -m pip --version
    ```



* 需要确认 Python 和 pip 是 64bit，并且处理器架构是 x86_64（或称作 x64、Intel 64、AMD64）架构 或 arm64 架构（paddle 已原生支持 Mac M1 芯片）：

    ```
    python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
    ```



* 如果您对机器环境不了解，请下载使用[快速安装脚本](https://fast-install.bj.bcebos.com/fast_install.sh)，配套说明请参考[这里](https://github.com/PaddlePaddle/FluidDoc/tree/develop/doc/fluid/install/install_script.md)。



## 二、开始安装

### 首先请选择您的版本

* 目前在 MacOS 环境仅支持 CPU 版 PaddlePaddle


### 根据版本进行安装

确定您的环境满足条件后可以开始安装了，选择下面您要安装的 PaddlePaddle


  ```
  python3 -m pip install paddlepaddle==2.6.0 -i https://mirror.baidu.com/pypi/simple
  ```


注:
* MacOS 上您需要安装 unrar 以支持 PaddlePaddle，可以使用命令`brew install unrar`
* 请确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python。根据您的环境您可能需要将说明中所有命令行中的 python3 替换为具体的 Python 路径。
* 默认下载最新稳定版的安装包，如需获取 develop 版本 nightly build 的安装包，请参考[这里](https://www.paddlepaddle.org.cn/install/quick/zh/1.8.5-windows-pip)
* 使用 MacOS 中自带 Python 可能会导致安装失败。请使用[python 官网](https://www.python.org/downloads/mac-osx/)提供的 python3.8.x、python3.9.x、python3.10.x、python3.11.x、python3.12.x。
* 上述命令默认安装`avx`、`mkl`的包，判断你的机器是否支持`avx`，可以输入以下命令，如果输出中包含`avx`，则表示机器支持`avx`。飞桨不再支持`noavx`指令集的安装包。
  ```
  sysctl machdep.cpu.features | grep -i avx
  ```
  或
  ```
  sysctl machdep.cpu.leaf7_features | grep -i avx
  ```

## **三、验证安装**

安装完成后您可以使用 `python` 进入 python 解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

## **四、如何卸载**

请使用以下命令卸载 PaddlePaddle：

* `python3 -m pip uninstall paddlepaddle`
