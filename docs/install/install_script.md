# 辅助安装脚本

## 使用方法

下载脚本至本地后，使用命令`/bin/bash fast_install.sh`启动脚本

### Ubuntu 和 CentOS

脚本会执行以下几步：

1.  GPU 检测

    检测您的机器是否含有我们支持的 GPU，如果有，会安装 GPU 版本的 PaddlePaddle，否则会安装 CPU 版本。
    （PaddlePaddle 目前支持 NVIDIA[官网](https://developer.nvidia.com/cuda-gpus#collapseOne)列出的，算力 7.0 以下的 GPU 和 v100 系列的 GPU）

2. CUDA，cuDNN 检测

    检测您的机器是否安装我们支持的 CUDA，cuDNN，具体地：

    1. 在`/usr/local/` 及其子目录下寻找 `cuda10.1/cuda10.2/cuda11.0/cuda11.2` 目录下的`version.txt`文件（通常如果您以默认方式安装了 CUDA）。 如果提示未找到 CUDA 请使用命令`find / -name version.txt`找到您所需要的 CUDA 目录下的“version.txt”路径，然后按照提示输入。
    2.  在`/usr` 及其子目录下寻找文件 `cudnn.h`  , 如果您的 cuDNN 未安装在默认路径请使用命令`find / -name cudnn.h`寻找您希望使用的 cuDNN 版本的`cudnn.h`路径并按提示输入

   如果未找到相应文件，则会安装 CPU 版本的 PaddlePaddle

3. 选择数学库
脚本默认会为您安装支持[MKL](https://software.intel.com/en-us/mkl)数学库的 PaddlePaddle，如果您的机器不支持`MKL`，请选择安装支持[OPENBLAS](https://www.openblas.net)的 PaddlePaddle

4. 选择 PaddlePaddle 版本
我们为您提供 2 种版本：开发版和稳定版，推荐您选择测试验证过的稳定版

5. 选择 Python 版本
脚本默认会使用您机器中的 Python，您也可以输入您希望使用的 Python 的路径

6. 检查[AVX](https://zh.wikipedia.org/zh-hans/AVX 指令集)指令集

7. 使用[Python virtualenv](https://virtualenv.pypa.io/en/latest/)
脚本也支持按您的需求创建 Python 的虚拟环境

以上检查完成后就会为您安装对应您系统的 PaddlePaddle 了，安装一般需要 1~2 分钟会根据您的网络来决定，请您耐心等待。


### macOS

脚本会执行以下几步：

1. 选择 PaddlePaddle 版本
我们为您提供 2 种版本：开发版和稳定版，推荐您选择测试验证过的稳定版

2.  检查 Python 版本
由于 macOS 自带的 Python 通常依赖于系统环境，因此我们不支持 macOS 自带的 Python 环境，请重新从 Python.org 安装 Python，然后根据提示输入您希望使用的 Python 的路径

3. 检查是否支持[AVX](https://zh.wikipedia.org/zh-hans/AVX 指令集)指令集
