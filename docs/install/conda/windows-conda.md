# Windows 下的 Conda 安装

[Anaconda](https://www.anaconda.com/)是一个免费开源的 Python 和 R 语言的发行版本，用于计算科学，Anaconda 致力于简化包管理和部署。Anaconda 的包使用软件包管理系统 Conda 进行管理。Conda 是一个开源包管理系统和环境管理系统，可在 Windows、macOS 和 Linux 上运行。

## 一、环境准备

在进行 PaddlePaddle 安装之前请确保您的 Anaconda 软件环境已经正确安装。软件下载和安装参见 Anaconda 官网(https://www.anaconda.com/)。在您已经正确安装 Anaconda 的情况下请按照下列步骤安装 PaddlePaddle。

* Windows 7/8/10 专业版/企业版 (64bit)
* GPU 版本支持 CUDA 10.1/10.2/11.2, 且只支持单卡
* conda 版本 4.8.3+ (64 bit)
* Windows 原生暂不支持 NCCL，分布式等相关功能
* 如果在 WSL2 环境下，推荐根据 Linux 方法安装使用 Paddle

### 1.1 创建虚拟环境

#### 1.1.1 安装环境

首先根据具体的 Python 版本创建 Anaconda 虚拟环境，PaddlePaddle 的 Anaconda 安装支持以下五种 Python 安装环境。


如果您想使用的 python 版本为 3.6:

```
conda create -n paddle_env python=3.6
```

如果您想使用的 python 版本为 3.7:

```
conda create -n paddle_env python=3.7
```

如果您想使用的 python 版本为 3.8:

```
conda create -n paddle_env python=3.8
```

如果您想使用的 python 版本为 3.9:

```
conda create -n paddle_env python=3.9
```


#### 1.1.2 进入 Anaconda 虚拟环境

```
conda activate paddle_env
```


## 1.2 其他环境检查

确认 Python 和 pip 是 64bit，并且处理器架构是 x86_64（或称作 x64、Intel 64、AMD64）架构，目前 PaddlePaddle 不支持 arm64 架构。下面的第一行输出的是"64bit"，第二行输出的是"x86_64（或 x64、AMD64）"即可：

```
python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
```


## 二、开始安装

本文档为您介绍 conda 安装方式


### 添加清华源（可选）

对于国内用户无法连接到 Anaconda 官方源的可以按照以下命令添加清华源。

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```


### 根据版本进行安装

确定您的环境满足条件后可以开始安装了，选择下面您要安装的 PaddlePaddle


#### CPU 版的 PaddlePaddle

如果您的计算机没有 NVIDIA® GPU 设备，请安装 CPU 版的 PaddlePaddle

```
conda install paddlepaddle --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```


#### GPU 版的 PaddlePaddle

如果您的计算机有 NVIDIA® GPU 设备

*  如果您是使用 CUDA 10.1，cuDNN 7.6+，安装 GPU 版本的命令为:

  ```
  conda install paddlepaddle-gpu==2.1.0 cudatoolkit=10.1 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
  ```

*  如果您是使用 CUDA 10.2，cuDNN 7.6+，安装 GPU 版本的命令为:

  ```
  conda install paddlepaddle-gpu==2.1.0 cudatoolkit=10.2 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
  ```

*  如果您是使用 CUDA 11.2，cuDNN 8.1.1+，安装 GPU 版本的命令为:

  ```
  conda install paddlepaddle-gpu==2.1.0 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
  ```


## **三、验证安装**

安装完成后您可以使用 `python` 进入 python 解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。
