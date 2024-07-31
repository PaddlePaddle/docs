..  _install_introduction:

=========
 安装指南
=========


-----------
  重要更新
-----------

* 支持用户安装 paddle 不依赖 cuda 和 cudnn，Paddle 自动处理 CUDA 和 cuDNN 的安装


-----------
  安装说明
-----------

本说明将指导您在 64 位操作系统编译和安装 PaddlePaddle

**1. 操作系统要求：**

* Windows 7 / 8 / 10/ 11，专业版 / 企业版
* Ubuntu 20.04 / 22.04
* CentOS 7
* macOS 10.x/11.x/12.x/13.x/14.x
* 操作系统要求是 64 位版本

**2. 处理器要求**

* 处理器支持 MKL
* 处理器架构是 x86_64（或称作 x64、Intel 64、AMD64）架构，目前 PaddlePaddle 仅提供 arm64 架构下 cpu wheel 包，不提供 arm 64 架构下 gpu wheel 包

**3. Python 和 pip 版本要求：**

* Python 的版本要求 3.8/3.9/3.10/3.11/3.12
* Python 具有 pip, 且 pip 的版本要求 20.2.2+
* Python 和 pip 要求是 64 位版本

**第一种安装方式：使用 pip 安装**

您可以选择“使用 pip 安装”、“使用 conda 安装”、“使用 docker 安装”、“从源码编译安装” 四种方式中的任意一种方式进行安装。

本节将介绍使用 pip 的安装方式。

1. 需要您确认您的 操作系统 满足上方列出的要求

2. 需要您确认您的 处理器 满足上方列出的要求

3. 确认您需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python

    使用以下命令输出 Python 路径，根据您的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径

        在 Windows 环境下，输出 Python 路径的命令为：

        ::

            where python

        在 macOS/Linux 环境下，输出 Python 路径的命令为：

        ::

            which python


4. 检查 Python 的版本

    使用以下命令确认是 3.8/3.9/3.10/3.11/3.12
    ::

        python --version

5. 检查 pip 的版本，确认是 20.2.2+

    ::

        python -m ensurepip
        python -m pip --version


6. 确认 Python 和 pip 是 64 bit，并且处理器架构是 x86_64（或称作 x64、Intel 64、AMD64）架构。下面的第一行输出的是 "64bit" ，第二行输出的是 "x86_64" 、 "x64" 或 "AMD64" 即可：

    ::

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"


7. 如果您希望使用 `pip <https://pypi.org/project/pip/>`_ 进行安装 PaddlePaddle 可以直接使用以下命令:

    (1). **CPU 版本** ：如果您只是想安装 CPU 版本请参考如下命令安装

        安装 CPU 版本的命令为：
        ::

            python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

    (2). **GPU 版本** ：如果您想使用 GPU 版本请参考如下命令安装
        
        注意：

            * 如果您想要安装CUDA 12.3版本，该版本需要libstdc++.so.6的版本大于3.4.30。为了满足此要求，您可以选择安装GCC 12版本，或者单独升级libstdc++库。

            * 如果你想要安装CUDA 11.8版本，该版本要求libstdc++.so.6的版本大于3.4.25。为了满足此要求，您可以选择安装GCC 8或者更高的GCC版本，或者单独升级libstdc++库。

        安装 GPU cuda12.3 版本的命令为：
        ::

            python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu123/

        安装 GPU cuda11.8 版本的命令为：
        ::
            python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/

    请确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python。根据您的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径。

8. 验证安装

    使用 python 进入 python 解释器，输入 import paddle ，再输入 paddle.utils.run_check()。

    如果出现 PaddlePaddle is installed successfully!，说明您已成功安装。

9. 更多帮助信息请参考：

    `Linux 下的 PIP 安装 <pip/linux-pip.html>`_

    `macOS 下的 PIP 安装 <pip/macos-pip.html>`_

    `Windows 下的 PIP 安装 <pip/windows-pip.html>`_


**第二种安装方式：使用源代码编译安装**

- 如果您只是使用 PaddlePaddle ，建议使用 **pip** 安装即可。
- 如果您有开发 PaddlePaddle 的需求，请参考：`从源码编译 <compile/fromsource.html>`_


..  toctree::
    :hidden:

    pip/frompip.rst
    conda/fromconda.rst
    docker/fromdocker.rst
    compile/fromsource.rst
    install_xpu_cn.md
    install_dcu_cn.md
    install_npu_cn.md
    install_mlu_cn.md
    install_NGC_PaddlePaddle_ch.rst
    Tables.md
