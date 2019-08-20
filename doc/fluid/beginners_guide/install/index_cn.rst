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

2. Python 和 pip 版本要求：
============================

* Python 2 的版本要求 2.7.15+

* Python 3 的版本要求 3.5.1+/3.6/3.7

* Python 具有 pip, 且 pip 的版本要求 9.0.1+

* Python 和 pip 要求是 64 位版本

3. PaddlePaddle 对 GPU 支持情况：
=================================

* 目前 `PaddlePaddle` 仅支持 `NVIDIA` 显卡的 `CUDA` 驱动

* 需要安装 `cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/>`_ ，版本要求 7.3+(For CUDA9/10), 7.1+(For CUDA 8)

* 如果您需要 GPU 多卡模式，需要安装 `NCCL 2 <https://developer.nvidia.com/nccl/>`_
    * 仅 Ubuntu/CentOS 支持 NCCL 2 技术

* 需要安装 `CUDA <https://docs.nvidia.com/cuda/cuda-installation-guide-windows/>`_，根据您系统不同，对 CUDA 版本要求不同：
    * Windows 安装 GPU 版本
        * Windows 7/8/10.0 支持 CUDA 8/9 单卡模式
		
        * 不支持 `nvidia-docker` 方式安装

    * Ubuntu 安装 GPU 版本
        * Ubuntu 14.04 支持 CUDA 8/10.0

        * Ubuntu 16.04 支持 CUDA 8/9/10.0

        * Ubuntu 18.04 支持 CUDA 10.0

        * 如果您是使用 `nvidia-docker` 安装，支持 CUDA 8/9/10.0

    * CentOS 安装 GPU 版本
        * 如果您是使用本机 `pip` 安装：
            * CentOS 7 支持 CUDA 9/10.0，支持 CUDA 8 但仅支持单卡模式

            * CentOS 6 支持 CUDA 8/9 单卡模式

        * 如果您是使用本机源码编译安装：
            * CentOS 7 支持 CUDA 9/10.0

            * CentOS 6 不推荐，不提供编译出现问题时的官方支持
		
        * 如果您是使用 `nvidia-docker` 安装，在CentOS 7 下支持 CUDA 8/9/10.0。

    * MacOS 不支持：PaddlePaddle 在 MacOS 平台没有 GPU 支持

请确保您的环境满足以上条件。如您有其他需求，请参考 `多版本whl包安装列表 <Tables.html/#ciwhls>`_

第一种安装方式：使用 pip 安装
================================

您可以选择“使用pip安装”、“使用docker安装”、“从源码编译安装” 三种方式中的任意一种方式进行安装。

本节将介绍使用 `pip` 的安装方式。

1. 需要您确认您的 操作系统 满足上方列出的要求

2. 处理器支持 MKL

3. 检查 Python 的版本

    如果您是使用 Python 2，使用以下命令确认是 2.7.15+
    ::
    
        python --version

    如果您是使用 Python 3，使用以下命令确认是 3.5.1+/3.6/3.7
    ::
    
        python3 --version
    
4. 检查 pip 的版本，确认是 9.0.1+  

    如果您是使用 Python 2
    ::
    
        pip --version

    如果您是使用 Python 3
    ::
    
        pip3 --version

5. 确认 Python 和 pip 是 64 bit，下面的命令输出的是 "64bit" 即可：

    如果您是使用 Python 2
    ::
    
        python -c "import platform;print(platform.architecture()[0])"

    如果您是使用 Python 3
    ::
    
        python3 -c "import platform;print(platform.architecture()[0])"

6. 如果您希望使用 `pip <https://www.docker.com>`_ 进行安装PaddlePaddle可以直接使用以下命令:

- 注意：目前官方没有对 `conda` 和 `anaconda` 进行支持，使用他们所附带的 `pip` 安装 `paddlepaddle` 也可能会带来冲突。所以建议使用纯净的 Python 环境的 `pip` 进行安装。


    (1). **CPU版本**：如果您只是想安装CPU版本请参考如下命令安装  

        如果您是使用 Python 2，安装CPU版本的命令为：
        ::
    
            pip install paddlepaddle
        
        如果您是使用 Python 3，安装CPU版本的命令为：
        ::
    
            pip3 install paddlepaddle

    (2). **GPU版本**：如果您想使用GPU版本请参考如下命令安装 

        注意：
            * 您的计算机需要具有支持 `CUDA` 驱动的 `NVIDIA` 显卡

            * 需要安装 `cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/>`_ ，版本要求 7.3+

            * 如果您需要 GPU 多卡模式，需要安装 `NCCL 2 <https://developer.nvidia.com/nccl/>`_
                * 仅 Ubuntu/CentOS 支持 NCCL 2 技术

            * 需要安装 `CUDA <https://docs.nvidia.com/cuda/cuda-installation-guide-windows/>`_，根据您系统不同，对 CUDA 版本要求不同：
                * Windows 7/8/10.0 支持 CUDA 8/9 单卡模式

                * Ubuntu
                    * Ubuntu 14.04 支持 CUDA 8/10.0

                    * Ubuntu 16.04 支持 CUDA 8/9/10.0

                    * Ubuntu 18.04 支持 CUDA 10.0

             	* CentOS 
                    * CentOS 7 支持 CUDA 9/10.0 ，CUDA 8 仅具有单卡模式支持

                    * CentOS 6 支持 CUDA 8/9 单卡模式

                * MacOS 不支持：PaddlePaddle 在 MacOS 平台没有 GPU 支持


        如果您是使用 Python2，请注意用以下指令安装的PaddlePaddle在Windows下默认支持CUDA9，Ubuntu、CentOS下默认支持CUDA10.0：
        ::

            pip install paddlepaddle-gpu 

        如果您是使用 Python 2，CUDA 8，cuDNN 7.3+，安装GPU版本的命令为：
        ::
    
            pip install paddlepaddle-gpu==1.5.1.post87

        如果您是使用 Python 2，CUDA 9，cuDNN 7.3+，安装GPU版本的命令为：
        ::
    
            pip install paddlepaddle-gpu==1.5.1.post97
        

        如果您是使用 Python 2，CUDA 10.0，cuDNN 7.3+，安装GPU版本的命令为：
        ::
    
            pip install paddlepaddle-gpu==1.5.1.post107
        
        如果您是使用 Python 3，请将上述命令中的 `pip` 更换为 `pip3` 进行安装。

7. 更多帮助信息请参考：
    `Ubuntu下安装 <install_Ubuntu.html>`_

    `CentOS下安装 <install_Ubuntu.html>`_

    `MacOS下安装 <install_Ubuntu.html>`_

    `Windows下安装 <install_Ubuntu.html>`_


第二种安装方式：使用 docker 安装
================================

您可以选择“使用pip安装”、“使用docker安装”、“从源码编译安装” 三种方式中的任意一种方式进行安装。

本节将介绍使用 `docker` 的安装方式。

如果您希望使用 `docker <https://www.docker.com>`_ 安装PaddlePaddle，可以使用以下命令:

1. **CPU 版本**

    (1). 首先需要安装 `docker <https://www.docker.com>`_

    注意：
        * CentOS 6 不支持 `docker` 方式安装

        * 处理器需要支持 MKL

    (2). 拉取预安装 PaddlePaddle 的镜像：
    ::

        docker pull hub.baidubce.com/paddlepaddle/paddle:1.5.1

    (3). 用镜像构建并进入Docker容器：
    ::

        docker run --name paddle -it -v dir1:dir2 hub.baidubce.com/paddlepaddle/paddle:1.5.1 /bin/bash

        > --name [Name of container] 设定Docker的名称；

        > -it 参数说明容器已和本机交互式运行；

        > -v 参数用于宿主机与容器里文件共享；其中dir1为宿主机目录，dir2为挂载到容器内部的目录，用户可以通过设定dir1和dir2自定义自己的挂载目录；例如：$PWD:/paddle 指定将宿主机的当前路径（Linux中PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录；

        > hub.baidubce.com/paddlepaddle/paddle:1.5.1 是需要使用的image名称；/bin/bash是在Docker中要执行的命令

2. **GPU 版本**

    (1). 首先需要安装 `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_

    注意：
        * 处理器需要支持 MKL

        * 您的计算机需要具有支持 `CUDA` 驱动的 `NVIDIA` 显卡

        * 需要安装 `cuDNN <https://docs.nvidia.com/deeplearning/sdk/cudnn-install/>`_ ，版本要求 7.3+

        * 如果您需要 GPU 多卡模式，需要安装 `NCCL 2 <https://developer.nvidia.com/nccl/>`_
            * 仅 Ubuntu/CentOS 支持 NCCL 2 技术

        * 需要安装 `CUDA <https://docs.nvidia.com/cuda/cuda-installation-guide-windows/>`_，根据您系统不同，对 CUDA 版本要求不同：

            * Ubuntu/CentOS 7 ，如果您是使用 `nvidia-docker` 安装，支持 CUDA 8/9/10.0

            * Windows/MacOS/CentOS 6 不支持 `nvidia-docker` 方式安装


    (2). 拉取支持`CUDA 10.0`, `cuDNN 7.3+` 预安装 PaddlePaddle 的镜像：
    ::

        nvidia-docker pull hub.baidubce.com/paddlepaddle/paddle:1.5.1-gpu-cuda10.0-cudnn7

    (3). 用镜像构建并进入Docker容器：
    ::

        nvidia-docker run --name paddle -it -v dir1:dir2 hub.baidubce.com/paddlepaddle/paddle:1.5.1-gpu-cuda10.0-cudnn7 /bin/bash

        > --name [Name of container] 设定Docker的名称；

        > -it 参数说明容器已和本机交互式运行；

        > -v 参数用于宿主机与容器里文件共享；其中dir1为宿主机目录，dir2为挂载到容器内部的目录，用户可以通过设定dir1和dir2自定义自己的挂载目录；例如：$PWD:/paddle 指定将宿主机的当前路径（Linux中PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录；

        > hub.baidubce.com/paddlepaddle/paddle:1.5.1-gpu-cuda10.0-cudnn7 是需要使用的image名称；/bin/bash是在Docker中要执行的命令  

    或如果您需要支持 `CUDA 8` 或者 `CUDA 9` 的版本，将上述命令的 `cuda10.0` 替换成 `cuda8.0` 或者 `cuda9.0` 即可，cuDNN 仅支持 `cuDNN 7.3+`

3. 如果您的机器不在中国大陆地区，可以直接从DockerHub拉取镜像：
    ::

        docker run --name paddle -it -v dir1:dir2 paddlepaddle/paddle:1.5.1 /bin/bash

        > --name [Name of container] 设定Docker的名称；

        > -it 参数说明容器已和本机交互式运行；

        > -v 参数用于宿主机与容器里文件共享；其中dir1为宿主机目录，dir2为挂载到容器内部的目录，用户可以通过设定dir1和dir2自定义自己的挂载目录；例如：$PWD:/paddle 指定将宿主机的当前路径（Linux中PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录；

        > paddlepaddle/paddle:1.5.1 是需要使用的image名称；/bin/bash是在Docker中要执行的命令

4. 更多帮助信息请参考：`使用Docker安装 <install_Docker.html>`_。
	
第三种安装方式：使用源代码编译安装
====================================

- 如果您只是使用 `PaddlePaddle` ，建议从 `pip` 和 `docker` 两种安装方式中选取一种进行安装即可。
- 如果您有开发PaddlePaddle的需求，请参考：`从源码编译 <compile/fromsource.html>`_

..	toctree::
	:hidden:

	install_Ubuntu.md
	install_CentOS.md
	install_MacOS.md
	install_Windows.md
	install_Docker.md
	compile/fromsource.rst
	Tables.md
 
