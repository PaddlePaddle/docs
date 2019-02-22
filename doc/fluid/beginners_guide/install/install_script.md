# 辅助安装脚本

## 使用方法

下载脚本至本地后，使用命令`/bin/bash fast_install.sh`启动脚本

### Ubuntu和CentOS

脚本会执行以下几步：

1.	GPU检测

	检测您的机器是否含有我们支持的GPU，如果有，会安装GPU版本的PaddlePaddle，否则会安装CPU版本。
	（PaddlePaddle目前支持NVIDIA[官网](https://developer.nvidia.com/cuda-gpus#collapseOne)列出的，算力7.0以下的GPU和v100系列的GPU）

2. CUDA，cuDNN检测

	检测您的机器是否安装我们支持的CUDA，cuDNN，具体地：

	1. 在`/usr/local/` 及其子目录下寻找 `cuda/cuda8/cuda9` 目录下的`version.txt`文件（通常如果您以默认方式安装了CUDA）。 如果提示未找到CUDA请使用命令`find / -name version.txt`找到您所需要的CUDA目录下的“version.txt”路径，然后按照提示输入。
	2.  在`/usr` 及其子目录下寻找文件 `cudnn.h`  , 如果您的cuDNN未安装在默认路径请使用命令`find / -name cudnn.h`寻找您希望使用的cuDNN版本的`cudnn.h`路径并按提示输入

   如果未找到相应文件，则会安装CPU版本的PaddlePaddle

3. 选择数学库
脚本默认会为您安装支持[MKL](https://software.intel.com/en-us/mkl)数学库的PaddlePaddle，如果您的机器不支持`MKL`，请选择安装支持[OPENBLAS](https://www.openblas.net)的PaddlePaddle

4. 选择PaddlePaddle版本
我们为您提供2种版本：开发版和稳定版，推荐您选择测试验证过的稳定版

5. 选择Python版本
脚本默认会使用您机器中的Python，您也可以输入您希望使用的Python的路径

6. 检查[AVX](https://zh.wikipedia.org/zh-hans/AVX指令集)指令集

7. 使用[Python virtualenv](https://virtualenv.pypa.io/en/latest/)
脚本也支持按您的需求创建Python的虚拟环境

以上检查完成后就会为您安装对应您系统的PaddlePaddle了，安装一般需要1~2分钟会根据您的网络来决定，请您耐心等待。


### MacOS

脚本会执行以下几步：

1. 选择PaddlePaddle版本
我们为您提供2种版本：开发版和稳定版，推荐您选择测试验证过的稳定版

2.	检查Python版本
由于MacOS自带的Python通常依赖于系统环境，因此我们不支持MacOS自带的Python环境，请重新从Python.org安装Python，然后根据提示输入您希望使用的Python的路径

3. 检查是否支持[AVX](https://zh.wikipedia.org/zh-hans/AVX指令集)指令集

