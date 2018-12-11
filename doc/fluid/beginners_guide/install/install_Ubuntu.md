***

# **Ubuntu下安装**

本说明将介绍如何在*64位台式机或笔记本电脑*以及Ubuntu系统下安装PaddlePaddle，我们支持的Ubuntu系统需满足以下要求：



请注意：在其他系统上的尝试可能会导致安装失败。请确保您的环境满足以上条件，我们默认提供的安装同时需要您的计算机处理器支持AVX指令集，否则请选择[多版本whl包安装列表](Tables.html/#ciwhls)中`no_avx`的版本。

Ubuntu系统下您可以使用`cat /proc/cpuinfo | grep avx`来检测您的处理器是否支持AVX指令集

* *Ubuntu 14.04 /16.04 /18.04*

## 确定要安装的版本

* 仅支持CPU的PaddlePaddle。如果您的计算机没有 NVIDIA® GPU，则只能安装此版本。如果您的计算机有GPU，
也推荐您先安装CPU版本的PaddlePaddle，来检测您本地的环境是否适合。

* 支持GPU的PaddlePaddle。为了使PaddlePaddle程序运行更加迅速，我们通过GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *CUDA 工具包9.0配合cuDNN v7*
	* *CUDA 工具包8.0配合cuDNN v7*
	* *GPU运算能力超过1.0的硬件设备*



## 选择如何安装
在Ubuntu的系统下我们提供4种安装方式：

* pip安装
* Docker安装(镜像中python的版本为2.7)
* 源码编译安装
* Docker源码编译安装(镜像中的python版本为2.7，3.5，3.6，3.7)



**使用pip安装**（最便捷的安装方式），我们为您提供pip安装方法，但它更依赖您的本机环境，可能会出现和您本机环境相关的一些问题。

**使用Docker进行安装**（最保险的安装方式），因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。



从[**源码编译安装**](#ubt_source)以及[**使用Docker进行源码编译安装**](#ubt_docker)，这是一种通过将PaddlePaddle源代码编译成为二进制文件，然后在安装这个二进制文件的过程，相比使用我们为您编译过的已经通过测试的二进制文件形式的PaddlePaddle，手动编译更为复杂，我们将在说明的最后详细为您解答。



<br/><br/>
### ***使用pip安装***

#### ****直接安装****

您可以直接粘贴以下命令到命令行来安装PaddlePaddle(适用于ubuntu16.04及以上安装CPU-ONLY的版本)，如果出现问题，您可以参照后面的解释对命令作出适应您系统的更改：

Python2.7：

	apt update && apt install -y python-dev python-pip && pip install paddlepaddle

Python3.5（该指令适用于本机未安装python2的用户，否则，请卸载python2之后再使用本指令）：

  apt-get udpate && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get install -y curl python3.5 python3.5-dev wget vim git && curl https://bootstrap.pypa.io/get-pip.py -o - | python3.5 && easy_install pip && pip3 install paddlepaddle

Python3.6、Python3.7：（由于版本相对较新，在不同Ubuntu版本上安装差异较大，不一一描述其安装过程，执行以下命令前，我们认为您已经准备好python3.6或3.7的环境，并安装了对应版本的python3-dev以及pip3）

    apt update && pip3 install paddlepaddle

<br/>

#### ****分步安装****
首先，我们使用以下指令来**检测本机的环境**是否适合安装PaddlePaddle：

    uname -m && cat /etc/*release

> 上面的命令将会显示本机的操作系统和位数信息，请确保您的计算机和本教程的要求一致。


其次，您的电脑需要满足以下任一要求：

*	Python2.7.x (dev)，Pip >= 9.0.1
*	Python3.5+.x (dev)，Pip3 >= 9.0.1

> 您的Ubuntu上可能已经安装pip请使用pip -V或pip3 -V来确认我们建议使用pip 9.0.1或更高版本来安装

	更新apt的源：   `apt update`

使用以下命令安装或升级Python和pip到需要的版本：（python3.6、python3.7安装pip和dev在不同Ubuntu版本下差别较大，不一一描述）

	- For python2： `sudo apt install python-dev python-pip`
	- For python3.5：`sudo apt install python3.5-dev` and `curl https://bootstrap.pypa.io/get-pip.py -o - | python3.5 && easy_install pip`
	- For python3.6、python3.7： 我们默认您应准备好python3.6（3.7）以及对应版本的dev和pip3
> 即使您的环境中已经有Python2或Python3也需要安装Python-dev或Python3.5（3.6、3.7）-dev。

现在，让我们来安装PaddlePaddle：

1. 使用pip install来安装PaddlePaddle

	* 对于需要**CPU版本PaddlePaddle**的用户：`pip install paddlepaddle` 或 `pip3 install paddlepaddle`

	* 对于需要**GPU版本PaddlePaddle**的用户：`pip install paddlepaddle-gpu` 或 `pip3 install paddlepaddle-gpu`

	> 1. 为防止出现nccl.h找不到的问题请首先按照以下命令安装nccl2（这里提供的是ubuntu 16.04，CUDA9，cuDNN v7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl/nccl-download):
			i. `wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb`
			ii.  `dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb`
			iii. `sudo apt-get install -y libnccl2=2.2.13-1+cuda9.0 libnccl-dev=2.2.13-1+cuda9.0`

	> 2. 如果您不规定pypi包版本号，我们默认为您提供支持Cuda 9/cuDNN v7的PaddlePaddle版本。

	* 对于出现`Cannot uninstall 'six'.`问题的用户，可是由于您的系统中已有的Python安装问题造成的，请使用`pip install paddlepaddle --ignore-installed six`（CPU）或`pip 	install paddlepaddle --ignore-installed six`（GPU）解决。

	* 对于有**其他要求**的用户：`pip install paddlepaddle==[版本号]` 或 `pip3 install paddlepaddle==[版本号]`

	> `版本号`参见[安装包列表](./Tables.html/#whls)或者您如果需要获取并安装**最新的PaddlePaddle开发分支**，可以从[多版本whl包列表](./Tables.html/#ciwhls)或者我们的[CI系统](https://paddleci.ngrok.io/project.html?projectId=Manylinux1&tab=projectOverview) 中下载最新的whl安装包和c-api开发包并安装。如需登录，请点击“Log in as guest”。





现在您已经完成使用`pip install` 来安装的PaddlePaddle的过程。


<br/><br/>
### ***使用Docker安装***

<!-- TODO: uncomment it when the offical website can split it to different pages我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。-->

为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**，关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)。



> 请注意，要安装和使用支持 GPU 的PaddlePaddle版本，您必须先安装[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)



如果已经**正确安装Docker**，即可以开始**使用Docker安装PaddlePaddle**

1. 使用以下指令拉取我们为您预安装好PaddlePaddle的镜像：

	* 对于需要**CPU版本的PaddlePaddle**的用户请使用以下指令拉取我们为您预安装好*PaddlePaddle For CPU*的镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:1.2`

	* 对于需要**GPU版本的PaddlePaddle**的用户请使用以下指令拉取我们为您预安装好*PaddlePaddle For GPU*的镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:1.2-gpu-cuda9.0-cudnn7`

	* 您也可以通过以下指令拉取任意的我们提供的Docker镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:[tag]`

		> （请把[tag]替换为[镜像表](./Tables.html/#dockers)中的内容）

2. 使用以下指令用已经拉取的镜像构建并进入Docker容器：

	`docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash`

	> 上述命令中，--name [Name of container] 设定Docker的名称；-it 参数说明容器已和本机交互式运行； -v $PWD:/paddle 指定将当前路径（Linux中PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录； `<imagename>` 指定需要使用的image名称，如果您需要使用我们的镜像请使用`hub.baidubce.com/paddlepaddle/paddle:[tag]` 注：tag的意义同第二步；/bin/bash是在Docker中要执行的命令。

3. （可选：当您需要第二次进入Docker容器中）使用如下命令使用PaddlePaddle：

	`docker start [Name of container]`

	> 启动之前创建的容器。

	`docker attach [Name of container]`

	> 进入启动的容器。

至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)。

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。




<br/><br/>
## ***验证安装***
安装完成后您可以使用：`python` 或 `python3` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
## ***如何卸载***
请使用以下命令卸载PaddlePaddle（使用docker安装PaddlePaddle的用户请进入包含PaddlePaddle的容器中使用以下命令，请使用相应版本的pip）：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall paddlepaddle-gpu` 或 `pip3 uninstall paddlepaddle-gpu`

