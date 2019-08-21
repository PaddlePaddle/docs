# **使用conda安装**

[Anaconda](https://www.anaconda.com/)是一个免费开源的Python和R语言的发行版本，用于计算科学，Anaconda致力于简化包管理和部署。Anaconda的包使用软件包管理系统Conda进行管理。Conda是一个开源包管理系统和环境管理系统，可在Windows、macOS和Linux上运行。

## 环境准备

在进行PaddlePaddle安装之前请确保您的Anaconda软件环境已经正确安装。软件下载和安装参见Anaconda官网(https://www.anaconda.com/)。在您已经正确安装Anaconda的情况下请按照下列步骤安装PaddlePaddle。

## 安装步骤

1. 创建虚拟环境

    首先根据具体的Python版本创建Anaconda虚拟环境，前PaddlePaddle的Anaconda安装支持以下四种Python安装环境。

    例如您想使用的python版本为2.7:

        conda create -n paddle_env python=2.7

    如果您想使用的python版本为3.5:

        conda create -n paddle_env python=3.5

    如果您想使用的python版本为3.6:

        conda create -n paddle_env python=3.6

    如果您想使用的python版本为3.7:

        conda create -n paddle_env python=3.7

    使用conda activate paddle_env命令进入Anaconda虚拟环境。

2. 确认您的conda虚拟环境和需要安装PaddlePaddle的Python是您预期的位置，因为您计算机可能有多个Python。进入Anaconda的命令行终端，输入以下指令确认Python位置。

    如果您是使用 Python 2，使用以下命令输出 Python 路径，根据您的环境您可能需要将说明中所有命令行中的 python 替换为具体的 Python 路径

        where python (for Windows) or which python (for MacOS/Linux)

    如果您是使用 Python 3，使用以下命令输出 Python 路径，根据您的环境您可能需要将说明中所有命令行中的 python3 替换为 python 或者替换为具体的 Python 路径

        where python3 (for Windows) or which python3 (for MacOS/Linux)

3. 检查Python的版本

    如果您是使用 Python 2，使用以下命令确认是 2.7.15+
    
        python --version

    如果您是使用 Python 3，使用以下命令确认是 3.5.1+/3.6/3.7
    
        python3 --version

4. 确认Python和pip是64bit，并且处理器架构是x86_64（或称作x64、Intel 64、AMD64）架构，目前PaddlePaddle不支持arm64架构。下面的第一行输出的是"64bit"，第二行输出的是"x86_64（或x64、AMD64）"即可：

    如果您是使用 Python 2

        python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

    如果您是使用 Python 3
    
        python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"

5. 安装PaddlePaddle

    (1). **CPU版本**：如果您只是想安装CPU版本请参考如下命令安装

        conda install paddlepaddle

    (2). **GPU版本**：如果您想使用GPU版本请参考如下命令安装 

        如果您是使用 CUDA 8，cuDNN 7.1+，安装GPU版本的命令为：
    
            conda install paddlepaddle-gpu cudatoolkit=8.0

        如果您是使用 CUDA 9，cuDNN 7.3+，安装GPU版本的命令为：
    
            conda install paddlepaddle-gpu cudatoolkit=9.0
        
        如果您是使用 CUDA 10.0，cuDNN 7.3+，安装GPU版本的命令为：
    
            conda install paddlepaddle-gpu cudatoolkit=10.0

6. 安装环境验证

    使用python进入python解释器，输入import paddle.fluid，再输入 paddle.fluid.install_check.run_check()。如果出现“Your Paddle Fluid is installed succesfully!”，说明您已成功安装。

## 注意

对于国内用户无法连接到Anaconda官方源的可以按照以下命令添加清华源进行安装。

    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
    conda config --set show_channel_urls yes
