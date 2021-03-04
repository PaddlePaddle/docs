..  _install_introduction:

#########
 安装说明
#########
本说明将指导您在64位操作系统编译和安装PaddlePaddle

**1. 操作系统要求：**

* Windows 7 / 8 / 10，专业版 / 企业版
* Ubuntu 14.04 / 16.04 / 18.04
* CentOS 6 / 7
* MacOS 10.11 / 10.12 / 10.13 / 10.14
* 操作系统要求是 64 位版本

**2. 处理器要求**

* 处理器支持 MKL
* 处理器架构是x86_64（或称作 x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构

**3. Python 和 pip 版本要求：**

* Python 2 的版本要求 2.7.15+
* Python 3 的版本要求 3.5.1+/3.6/3.7/3.8
* Python 具有 pip, 且 pip 的版本要求 20.2.2+
* Python 和 pip 要求是 64 位版本

**4. PaddlePaddle 对 GPU 支持情况：**

* 目前 **PaddlePaddle** 仅支持 **NVIDIA** 显卡的 **CUDA** 驱动
* 需要安装 `cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/>`_ ，版本要求 7.6+(For CUDA9/10)
* 如果您需要 GPU 多卡模式，需要安装 `NCCL 2 <https://developer.nvidia.com/nccl/>`_

    * 仅 Ubuntu/CentOS 支持 NCCL 2 技术
* 需要安装 `CUDA <https://docs.nvidia.com/cuda/cuda-installation-guide-windows/>`_ ，根据您系统不同，对 CUDA 版本要求不同：

    * Windows 安装 GPU 版本

        * Windows 7/8/10 支持 CUDA 9.0/10.0 单卡模式，不支持 CUDA 9.1/9.2/10.1	 
        * 不支持 **nvidia-docker** 方式安装
    * Ubuntu 安装 GPU 版本

        * Ubuntu 14.04 支持 CUDA 10.0/10.1，不支持CUDA 9.0/9.1/9.2
        * Ubuntu 16.04 支持 CUDA 9.0/9.1/9.2/10.0/10.1/10.2
        * Ubuntu 18.04 支持 CUDA 10.0/10.1/10.2/11.0，不支持CUDA 9.0/9.1/9.2
        * 如果您是使用 **nvidia-docker** 安装，支持 CUDA 9.0/9.1/9.2/10.0/10.1/10.2/11.0
    * CentOS 安装 GPU 版本

        * 如果您是使用本机 **pip** 安装：

            * CentOS 7 支持 CUDA 9.0/9.1/9.2/10.0/10.1/10.2/11.0，CUDA 9.1 仅支持单卡模式
            * CentOS 6 支持 CUDA 9.0/9.1/9.2/10.0/10.1/10.2 单卡模式
        * 如果您是使用本机源码编译安装：

            * CentOS 7 支持 CUDA 9.0/9.1/9.2/10.0/10.1/10.2/11.0，CUDA 9.1 仅支持单卡模式
            * CentOS 6 不推荐，不提供编译出现问题时的官方支持
        * 如果您是使用 **nvidia-docker** 安装，在CentOS 7 下支持 CUDA 9.0/9.1/9.2/10.0/10.1/10.2/11.0
    * MacOS 不支持：PaddlePaddle 在 MacOS 平台没有 GPU 支持

请确保您的环境满足以上条件。如您有其他需求，请参考 `多版本whl包安装列表 <Tables.html#ciwhls>`_ .

**5. PaddlePaddle 对 NCCL 支持情况：**

* Windows 支持情况

    * 不支持NCCL
* Ubuntu 支持情况

    * Ubuntu 14.04：

        * CUDA10.1 下支持NCCL v2.4.2-v2.4.8
        * CUDA10.0 下支持NCCL v2.3.7-v2.4.8
    * Ubuntu 16.04:

        * CUDA10.1 下支持NCCL v2.4.2-v2.4.8
        * CUDA10.0/9.2/9.0 下支持NCCL v2.3.7-v2.4.8        
        * CUDA9.1 下支持NCCL v2.1.15
    * Ubuntu 18.04：

        * CUDA10.1 下支持NCCL v2.4.2-v2.4.8
        * CUDA10.0 下支持NCCL v2.3.7-v2.4.8
* CentOS 支持情况

    * CentOS 6：不支持NCCL
    * CentOS 7：

        * CUDA10.1 下支持NCCL v2.4.2-v2.4.8
        * CUDA10.0/9.2/9.0 下支持NCCL v2.3.7-v2.4.8
* MacOS 支持情况

    * 不支持NCCL

**第一种安装方式：使用 pip 安装**

您可以选择“使用pip安装”、“使用conda安装”、“使用docker安装”、“从源码编译安装” 四种方式中的任意一种方式进行安装。

本节将介绍使用 pip 的安装方式。

1. 需要您确认您的 操作系统 满足上方列出的要求

2. 需要您确认您的 处理器 满足上方列出的要求

3. 确认您需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python

    如果您是使用 Python 2，使用以下命令输出 Python 路径，根据您的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径
    
        在 Windows 环境下，输出 Python 路径的命令为：
        
        ::

            where python

        在 MacOS/Linux 环境下，输出 Python 路径的命令为：

        ::

            which python

    如果您是使用 Python 3，使用以下命令输出 Python 路径，根据您的环境您可能需要将说明中所有命令行中的 python3 替换为 python 或者替换为具体的 Python 路径

        在 Windows 环境下，输出 Python 路径的命令为：

        ::

            where python3

        在 MacOS/Linux 环境下，输出 Python 路径的命令为：

        ::

            which python3

4. 检查 Python 的版本

    如果您是使用 Python 2，使用以下命令确认是 2.7.15+
    ::
    
        python --version

    如果您是使用 Python 3，使用以下命令确认是 3.5.1+/3.6/3.7/3.8
    ::
    
        python3 --version
    
5. 检查 pip 的版本，确认是 20.2.2+  

    如果您是使用 Python 2
    ::
    
        python -m ensurepip 
        python -m pip --version

    如果您是使用 Python 3
    ::
    
        python3 -m ensurepip
        python3 -m pip --version

6. 确认 Python 和 pip 是 64 bit，并且处理器架构是x86_64（或称作 x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构。下面的第一行输出的是 "64bit" ，第二行输出的是 "x86_64" 、 "x64" 或 "AMD64" 即可：

    如果您是使用 Python 2
    ::

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

    如果您是使用 Python 3
    ::
    
        python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

7. 如果您希望使用 `pip <https://pypi.org/project/pip/>`_ 进行安装PaddlePaddle可以直接使用以下命令:

    (1). **CPU版本** ：如果您只是想安装CPU版本请参考如下命令安装 

        安装CPU版本的命令为：
        ::
    
            python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

            或

            python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

        如果您是使用Windows系统请使用以下指令:

            python -m pip install paddlepaddle==2.0.0 -f https://paddlepaddle.org.cn/whl/stable.html


    (2). **GPU版本** ：如果您想使用GPU版本请参考如下命令安装 

        注意：

            * 需要您确认您的 GPU 满足上方列出的要求

        请注意用以下指令安装的PaddlePaddle在Windows、Ubuntu、CentOS下只支持CUDA10.2：
        ::

            python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple

            或

            python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

        如果您是使用Windows系统请使用以下指令:

            python -m pip install paddlepaddle_gpu==2.0.0 -f https://paddlepaddle.org.cn/whl/stable.html 
        
    请确认需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python。根据您的环境您可能需要将说明中所有命令行中的 python 替换为 python3 或者替换为具体的 Python 路径。

8. 验证安装

    使用 python 或 python3 进入python解释器，输入import paddle ，再输入 paddle.utils.run_check()。

    如果出现 PaddlePaddle is installed successfully!，说明您已成功安装。

9. 更多帮助信息请参考：

    `Linux下的PIP安装 <pip/linux-pip.html>`_

    `MacOS下的PIP安装 <pip/macos-pip.html>`_

    `Windows下的PIP安装 <pip/windows-pip.html>`_


**第二种安装方式：使用源代码编译安装**

- 如果您只是使用 PaddlePaddle ，建议使用 **pip** 安装即可。
- 如果您有开发PaddlePaddle的需求，请参考：`从源码编译 <compile/fromsource.html>`_


..	toctree::
	:hidden:

	pip/frompip.rst
	conda/fromconda.rst
	docker/fromdocker.rst
	compile/fromsource.rst
	install_Kunlun_zh.md
	Tables.md
