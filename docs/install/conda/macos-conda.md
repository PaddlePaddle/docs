# macOS 下的 Conda 安装

[Anaconda](https://www.anaconda.com/)是一个免费开源的 Python 和 R 语言的发行版本，用于计算科学，Anaconda 致力于简化包管理和部署。Anaconda 的包使用软件包管理系统 Conda 进行管理。Conda 是一个开源包管理系统和环境管理系统，可在 Windows、macOS 和 Linux 上运行。

## 一、环境准备

### 1.1 创建虚拟环境

#### 1.1.1 安装环境

首先根据具体的 Python 版本创建 Anaconda 虚拟环境，PaddlePaddle 的 Anaconda 安装支持 3.8 - 3.12 版本的 Python 安装环境。

```
conda create -n paddle_env python=YOUR_PY_VER
```


#### 1.1.2 进入 Anaconda 虚拟环境


```
conda activate paddle_env
```



### 1.2 其他环境检查

#### 1.2.1 确认 Python 安装路径

确认您的 conda 虚拟环境和需要安装 PaddlePaddle 的 Python 是您预期的位置，因为您计算机可能有多个 Python。进入 Anaconda 的命令行终端，输入以下指令确认 Python 位置。

输出 Python 路径的命令为:

```
which python3
```

根据您的环境，您可能需要将说明中所有命令行中的 python3 替换为具体的 Python 路径



#### 1.2.2 检查 Python 版本

使用以下命令确认版本

```
python3 --version
```



#### 1.2.3 检查系统环境

确认 Python 和 pip 是 64bit，并且处理器架构是 x86_64（或称作 x64、Intel 64、AMD64）架构 或 arm64 架构（paddle 已原生支持 Mac M1 芯片）：


```
python3 -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
```


## 二、开始安装

本文档为您介绍 conda 安装方式

### 添加清华源（可选）

* 对于国内用户无法连接到 Anaconda 官方源的可以按照以下命令添加清华源:

  ```
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  ```
  ```
  conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  ```
  ```
  conda config --set show_channel_urls yes
  ```

### 安装 CPU 版 PaddlePaddle

* 目前在 macOS 环境仅支持 CPU 版 PaddlePaddle，请参考如下命令安装 Paddle:

  ```
  conda install paddlepaddle==3.0.0b0 -c paddle
  ```

## **三、验证安装**

安装完成后您可以使用 `python3` 进入 python 解释器，输入`import paddle` ，再输入
 `paddle.utils.run_check()`

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。
