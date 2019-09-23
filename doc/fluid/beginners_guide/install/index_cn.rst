==========
 安装说明
==========
本说明将指导您在64位操作系统编译和安装PaddlePaddle

1. 操作系统要求：
============================

* Windows 7 / 8 / 10，专业版 / 企业版

* Ubuntu 14.04 / 16.04 / 18.04

* CentOS 6 / 7

* MacOS 10.11 / 10.12 / 10.13 / 10.14

* 操作系统要求是 64 位版本

2. 处理器要求
============================

* 处理器支持 MKL

* 处理器架构是x86_64（或称作 x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构

3. Python 和 pip 版本要求：
============================

* Python 2 的版本要求 2.7.15+

* Python 3 的版本要求 3.5.1+/3.6/3.7

* Python 具有 pip, 且 pip 的版本要求 9.0.1+

* Python 和 pip 要求是 64 位版本

4. PaddlePaddle 对 GPU 支持情况：
=================================

* 目前 `PaddlePaddle` 仅支持 `NVIDIA` 显卡的 `CUDA` 驱动

* 需要安装 `cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/>`_ ，版本要求 7.3+(For CUDA9/10), 7.1+(For CUDA 8)

* 如果您需要 GPU 多卡模式，需要安装 `NCCL 2 <https://developer.nvidia.com/nccl/>`_
    * 仅 Ubuntu/CentOS 支持 NCCL 2 技术

* 需要安装 `CUDA <https://docs.nvidia.com/cuda/cuda-installation-guide-windows/>`_，根据您系统不同，对 CUDA 版本要求不同：
    * Windows 安装 GPU 版本
        * Windows 7/8/10 支持 CUDA 8.0/9.0/10.0 单卡模式，不支持 CUDA 9.1/9.2/10.1
		
        * 不支持 `nvidia-docker` 方式安装

    * Ubuntu 安装 GPU 版本
        * Ubuntu 14.04 支持 CUDA 8.0/10.0，不支持CUDA 9.0/9.1/9.2/10.1

        * Ubuntu 16.04 支持 CUDA 8.0/9.0/9.1/9.2/10.0，不支持10.1

        * Ubuntu 18.04 支持 CUDA 10.0，不支持CUDA 8.0/9.0/9.1/9.2/10.1

        * 如果您是使用 `nvidia-docker` 安装，支持 CUDA 8.0/9.0/9.1/9.2/10.0，不支持10.1

    * CentOS 安装 GPU 版本
        * 如果您是使用本机 `pip` 安装：
            * CentOS 7 支持 CUDA 9.0/9.2/10.0，不支持10.1，支持 CUDA 8.0/9.1 但仅支持单卡模式

            * CentOS 6 支持 CUDA 8.0/9.0/9.1/9.2/10.0 单卡模式，不支持10.1

        * 如果您是使用本机源码编译安装：
            * CentOS 7 支持 CUDA 9.0/9.2/10.0

            * CentOS 6 不推荐，不提供编译出现问题时的官方支持
		
        * 如果您是使用 `nvidia-docker` 安装，在CentOS 7 下支持 CUDA 8.0/9.0/9.1/9.2/10.0，不支持10.1

    * MacOS 不支持：PaddlePaddle 在 MacOS 平台没有 GPU 支持

请确保您的环境满足以上条件。如您有其他需求，请参考 `多版本whl包安装列表 <Tables.html/#ciwhls>`_

5. PaddlePaddle 对 NCCL 支持情况：
=================================

* Windows 支持情况

    * 不支持NCCL

* Ubuntu 支持情况

    * Ubuntu 14.04：

        * CUDA10.0 下支持NCCL v2.3.7-v2.4.8
        
        * CUDA8.0 下支持NCCL v2.1.15-v2.2.13

    * Ubuntu 16.04:

        * CUDA10.0/9.2/9.0 下支持NCCL v2.3.7-v2.4.8
        
        * CUDA9.1 下支持NCCL v2.1.15

        * CUDA8.0 下支持NCCL v2.1.15-v2.2.13

    * Ubuntu 18.04：

        * CUDA10.0 下支持NCCL v2.3.7-v2.4.8

* CentOS 支持情况

    * CentOS 6：不支持NCCL

    * CentOS 7：

        * CUDA10.0/9.2/9.0 下支持NCCL v2.3.7-v2.4.8

* MacOS 支持情况
    * 不支持NCCL

第一种安装方式：使用 pip 安装
================================

您可以选择“使用pip安装”、“使用conda安装”、“使用docker安装”、“从源码编译安装” 四种方式中的任意一种方式进行安装。

本节将介绍使用 `pip` 的安装方式。

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

    如果您是使用 Python 3，使用以下命令确认是 3.5.1+/3.6/3.7
    ::
    
        python3 --version
    
5. 检查 pip 的版本，确认是 9.0.1+  

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

    (1). **CPU版本**：如果您只是想安装CPU版本请参考如下命令安装（使用清华源） 

        如果您是使用 Python 2，安装CPU版本的命令为：
        ::
    
            python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
        
        如果您是使用 Python 3，安装CPU版本的命令为：
        ::
    
            python3 -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

    (2). **GPU版本**：如果您想使用GPU版本请参考如下命令安装（使用清华源） 

        注意：
            * 需要您确认您的 GPU 满足上方列出的要求

        如果您是使用 Python2，请注意用以下指令安装的PaddlePaddle在Windows、Ubuntu、CentOS下默认支持CUDA10.0：
        ::

            python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple

        如果您是使用 Python 2，CUDA 8，cuDNN 7.1+，安装GPU版本的命令为：
        ::
    
            python -m pip install paddlepaddle-gpu==1.5.2.post87 -i https://pypi.tuna.tsinghua.edu.cn/simple

        如果您是使用 Python 2，CUDA 9，cuDNN 7.3+，安装GPU版本的命令为：
        ::
    
            python -m pip install paddlepaddle-gpu==1.5.2.post97 -i https://pypi.tuna.tsinghua.edu.cn/simple

        如果您是使用 Python 2，CUDA 10.0，cuDNN 7.3+，安装GPU版本的命令为：
        ::
    
            python -m pip install paddlepaddle-gpu==1.5.2.post107 -i https://pypi.tuna.tsinghua.edu.cn/simple
        
        如果您是使用 Python 3，请将上述命令中的 `python` 更换为 `python3` 进行安装。

8. 验证安装

    使用 python 或 python3 进入python解释器，输入import paddle.fluid ，再输入 paddle.fluid.install_check.run_check()。

    如果出现 Your Paddle Fluid is installed succesfully!，说明您已成功安装。

9. 更多帮助信息请参考：
    `Ubuntu下安装 <install_Ubuntu.html>`_

    `CentOS下安装 <install_CentOS.html>`_

    `MacOS下安装 <install_MacOS.html>`_

    `Windows下安装 <install_Windows.html>`_


第二种安装方式：使用 conda 安装
================================

您可以选择“使用pip安装”、“使用conda安装”、“使用docker安装”、“从源码编译安装” 四种方式中的任意一种方式进行安装。

本节将介绍使用 `conda` 的安装方式。

1. 需要您确认您的 操作系统 满足上方列出的要求

2. 需要您确认您的 处理器 满足上方列出的要求

3. 对于国内用户无法连接到Anaconda官方源的可以按照以下命令添加清华源进行安装。

    ::

        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
        conda config --set show_channel_urls yes

4. 如果您需要新建 conda 的虚拟环境专门给 Paddle 使用（--name后边的环境名称，您可以自己选择）：

    如果您是使用 Python2 并且在 Window 环境下
    
    ::

        conda create --name paddle python=2.7
        activate paddle

    如果您是使用 Python2 并且在 MacOS/Linux 环境下

    ::

        conda create --name paddle python=2.7
        conda activate paddle

    如果您是使用 Python3 并且在 Window 环境下，注意：python3版本可以是3.5.1+/3.6/3.7

    ::

        conda create --name paddle python=3.7
        activate paddle

    如果您是使用 Python3 并且在 MacOS/Linux 环境下，注意：python3版本可以是3.5.1+/3.6/3.7

    ::

        conda create --name paddle python=3.7
        conda activate paddle

5. 确认您需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python，进入 Anaconda 的命令行终端，输入以下指令确认 Python 位置

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

6. 检查 Python 的版本

    如果您是使用 Python 2，使用以下命令确认是 2.7.15+
    ::
    
        python --version

    如果您是使用 Python 3，使用以下命令确认是 3.5.1+/3.6/3.7
    ::
    
        python3 --version
    
7. 检查 pip 的版本，确认是 9.0.1+  

    如果您是使用 Python 2
    ::
    
        python -m ensurepip 
        python -m pip --version

    如果您是使用 Python 3
    ::
    
        python3 -m ensurepip
        python3 -m pip --version

8. 确认 Python 和 pip 是 64 bit，并且处理器架构是x86_64（或称作 x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构。下面的第一行输出的是 "64bit" ，第二行输出的是 "x86_64" 、 "x64" 或 "AMD64" 即可：

    如果您是使用 Python 2
    ::

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

    如果您是使用 Python 3
    ::
    
        python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

9. 如果您希望使用 conda 进行安装PaddlePaddle可以直接使用以下命令:

    (1). **CPU版本**：如果您只是想安装CPU版本请参考如下命令安装  

    ::

        conda install paddlepaddle
    

    (2). **GPU版本**：如果您想使用GPU版本请参考如下命令安装 

        注意：
            * 需要您确认您的 GPU 满足上方列出的要求

        如果您是使用 CUDA 8，cuDNN 7.1+，安装GPU版本的命令为：
        ::
    
            conda install paddlepaddle-gpu cudatoolkit=8.0

        如果您是使用 CUDA 9，cuDNN 7.3+，安装GPU版本的命令为：
        ::
    
            conda install paddlepaddle-gpu cudatoolkit=9.0
        

        如果您是使用 CUDA 10.0，cuDNN 7.3+，安装GPU版本的命令为：
        ::
    
            conda install paddlepaddle-gpu cudatoolkit=10.0

10. 验证安装

    使用 python 或 python3 进入python解释器，输入import paddle.fluid ，再输入 paddle.fluid.install_check.run_check()。

    如果出现 Your Paddle Fluid is installed succesfully!，说明您已成功安装。

11. 更多帮助信息请参考：
    `conda下安装 <install_Conda.html>`_


第三种安装方式：使用 docker 安装
================================

您可以选择“使用pip安装”、“使用conda安装”、“使用docker安装”、“从源码编译安装” 四种方式中的任意一种方式进行安装。

本节将介绍使用 `docker` 的安装方式。

如果您希望使用 `docker <https://www.docker.com>`_ 安装PaddlePaddle，可以使用以下命令:

1. **CPU 版本**

    (1). 首先需要安装 `docker <https://www.docker.com>`_

    注意：
        * CentOS 6 不支持 `docker` 方式安装

        * 处理器需要支持 MKL

    (2). 拉取预安装 PaddlePaddle 的镜像：
    ::

        docker pull hub.baidubce.com/paddlepaddle/paddle:1.5.2

    (3). 用镜像构建并进入Docker容器：
    ::

        docker run --name paddle -it -v dir1:dir2 hub.baidubce.com/paddlepaddle/paddle:1.5.2 /bin/bash

        > --name [Name of container] 设定Docker的名称；

        > -it 参数说明容器已和本机交互式运行；

        > -v 参数用于宿主机与容器里文件共享；其中dir1为宿主机目录，dir2为挂载到容器内部的目录，用户可以通过设定dir1和dir2自定义自己的挂载目录；例如：$PWD:/paddle 指定将宿主机的当前路径（Linux中PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录；

        > hub.baidubce.com/paddlepaddle/paddle:1.5.2 是需要使用的image名称；/bin/bash是在Docker中要执行的命令

2. **GPU 版本**

    (1). 首先需要安装 `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_

    注意：
        * 处理器需要支持 MKL

        * 您的计算机需要具有支持 `CUDA` 驱动的 `NVIDIA` 显卡

        * 需要安装 `cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/>`_ ，版本要求 7.3+(For CUDA9/10), 7.1+(For CUDA 8)

        * 如果您需要 GPU 多卡模式，需要安装 `NCCL 2 <https://developer.nvidia.com/nccl/>`_
            * 仅 Ubuntu/CentOS 支持 NCCL 2 技术

        * 需要安装 `CUDA <https://docs.nvidia.com/cuda/cuda-installation-guide-windows/>`_，根据您系统不同，对 CUDA 版本要求不同：

            * Ubuntu/CentOS 7 ，如果您是使用 `nvidia-docker` 安装，支持 CUDA 8.0/9.0/9.1/9.2/10.0

            * Windows/MacOS/CentOS 6 不支持 `nvidia-docker` 方式安装


    (2). 拉取支持`CUDA 10.0`, `cuDNN 7.3+` 预安装 PaddlePaddle 的镜像：
    ::

        nvidia-docker pull hub.baidubce.com/paddlepaddle/paddle:1.5.2-gpu-cuda10.0-cudnn7

    (3). 用镜像构建并进入Docker容器：
    ::

        nvidia-docker run --name paddle -it -v dir1:dir2 hub.baidubce.com/paddlepaddle/paddle:1.5.2-gpu-cuda10.0-cudnn7 /bin/bash

        > --name [Name of container] 设定Docker的名称；

        > -it 参数说明容器已和本机交互式运行；

        > -v 参数用于宿主机与容器里文件共享；其中dir1为宿主机目录，dir2为挂载到容器内部的目录，用户可以通过设定dir1和dir2自定义自己的挂载目录；例如：$PWD:/paddle 指定将宿主机的当前路径（Linux中PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录；

        > hub.baidubce.com/paddlepaddle/paddle:1.5.2-gpu-cuda10.0-cudnn7 是需要使用的image名称；/bin/bash是在Docker中要执行的命令  

    或如果您需要支持 `CUDA 8` 或者 `CUDA 9` 的版本，将上述命令的 `cuda10.0` 替换成 `cuda8.0` 或者 `cuda9.0` 即可

3. 如果您的机器不在中国大陆地区，可以直接从DockerHub拉取镜像：
    ::

        docker run --name paddle -it -v dir1:dir2 paddlepaddle/paddle:1.5.2 /bin/bash

        > --name [Name of container] 设定Docker的名称；

        > -it 参数说明容器已和本机交互式运行；

        > -v 参数用于宿主机与容器里文件共享；其中dir1为宿主机目录，dir2为挂载到容器内部的目录，用户可以通过设定dir1和dir2自定义自己的挂载目录；例如：$PWD:/paddle 指定将宿主机的当前路径（Linux中PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录；

        > paddlepaddle/paddle:1.5.2 是需要使用的image名称；/bin/bash是在Docker中要执行的命令

4. 验证安装

    使用 python 或 python3 进入python解释器，输入import paddle.fluid ，再输入 paddle.fluid.install_check.run_check()。

    如果出现 Your Paddle Fluid is installed succesfully!，说明您已成功安装。

5. 更多帮助信息请参考：`使用Docker安装 <install_Docker.html>`_。
	
第四种安装方式：使用源代码编译安装
====================================

- 如果您只是使用 `PaddlePaddle` ，建议从 `pip` 和 `conda` 、 `docker` 三种安装方式中选取一种进行安装即可。
- 如果您有开发PaddlePaddle的需求，请参考：`从源码编译 <compile/fromsource.html>`_

..	toctree::
	:hidden:

	install_Ubuntu.md
	install_CentOS.md
	install_MacOS.md
	install_Windows.md
	install_Conda.md
	install_Docker.md
	compile/fromsource.rst
	Tables.md
 
