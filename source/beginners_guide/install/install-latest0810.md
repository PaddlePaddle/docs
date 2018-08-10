# **安装说明**
本说明将指导您编译和安装PaddlePaddle，目前PaddlePaddle支持以下环境：

* *64位台式机或笔记本电脑*
* *Ubuntu 14.04 /16.04 /18.04*
* *CentOS 7 / 6*

请确保您的环境满足以上条件

## **安装PaddlePaddle**

* Ubuntu下安装PaddlePaddle
* CentOS下安装PaddlePaddle

***
### **Ubuntu下安装PaddlePaddle**

本说明将介绍如何在Ubuntu下安装PaddlePaddle,我们支持的Ubuntu系统需满足以下要求

请注意：在其他系统上的尝试可能会导致安装失败

* *64位台式机或笔记本电脑*
* *Ubuntu 14.04 /16.04 /18.04*

#### 确定要安装的PaddlePaddle版本

* 仅支持CPU的PaddlePaddle。如果您的计算机没有 NVIDIA® GPU，则只能安装此版本。如果您的计算机有GPU，
也推荐您先安装CPU版本的PaddlePaddle，来检测您本地的环境是否适合。

* 支持GPU的PaddlePaddle，为了使PaddlePaddle程序运行的更加迅速，我们通过GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For CUDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *Cuda 工具包9.0配合cudnn v7*
	* *Cuda 工具包8.0配合cudnn v7*
	* *GPU运算能力超过1.0的硬件设备*



#### 选择如何安装PaddlePaddle
在Ubuntu的系统下我们提供4种不同的安装方式：

* Docker安装
* 原生pypi安装
* 源码编译安装
* Docker源码编译安装


我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样


**使用原生pypi安装**，我们为您提供原生pypi安装方法，但它更依赖您的本机环境，可能会出现和您本机环境相关的一些问题

从**源码编译安装**，这是一种通过将PaddlePaddle源代码编译成为二进制文件，然后在安装这个二进制文件的过程，相比使用我们为您编译过的已经通过测试的二进制文件形式的PaddlePaddle，手动编译更为复杂，我们将在说明的最后详细为您解答
<br/><br/>
##### ***使用Docker进行安装***

我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样

为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**, 关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)

如果已经**正确安装Docker**，即可以开始**使用Docker安装PaddlePaddle**

1. 使用以下指令拉取我们为您预安装好PaddlePaddle的镜像：


	* 对于需要**CPU版本的PaddlePaddle**的用户请使用以下指令拉取我们为您预安装好*PaddlePaddle For CPU*的镜像

		`docker pull paddlepaddle/paddle:latest`
		

	* 对于需要**GPU版本的PaddlePaddle**的用户请使用以下指令拉取我们为您预安装好*PaddlePaddle For GPU*的镜像

		`docker pull paddlepaddle/paddle:latest-gpu`
		

	* 您也可以通过以下指令拉取任意的我们提供的Docker镜像

		`docker pull paddlepaddle/paddle:[tag]`
		> （请把[tag]替换为[镜像表](https://hub.docker.com/r/paddlepaddle/paddle_manylinux_devel/tags/)中的内容）
		
2. 使用以下指令用已经拉取的镜像构建并进入Docker容器：

	`docker run --name [Name of container] -it -v $PWD:/work <imagename> /bin/bash`

	> 上述命令中，--name [Name of container] 设定Docker的名称；-it 参数说明容器已和本机交互式运行； -v $PWD:/work 指定将当前路径（Linux中$PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /work 目录； paddlepaddle/paddle 指定需要使用的image名称；/bin/bash是在Docker中要执行的命令

3. 在Docker中 使用如下命令使用PaddlePaddle

	`docker start [Name of container]`
	> 启动之前创建的容器

	`docker attach [Name of container]`
	> 进入启动的容器

至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。


<br/><br/>
##### ***使用原生的pip安装PaddlePaddle***

首先, 我们使用以下指令来**检测本机的环境**是否适合安装PaddlePaddle

`uname -m && cat /etc/*release`
> 上面的命令将会显示本机的操作系统和位数信息，请确保您的计算机和本教程的要求一致


其次, 您的电脑需要满足以下要求：

*	Python2.7.x (dev)
*	Pip >= 9.0.1
	> 您的Ubuntu上可能已经安装pip请使用pip -V来确认我们建议使用pip 9.0.1或更高版本来安装

	更新apt的源：   `apt update`

	使用以下命令安装或升级Python和pip到需要的版本： `sudo apt install python-dev python-pip`
	> 即使您的环境中已经有Python2.7也需要安装Python dev

现在，让我们来安装PaddlePaddle

1. 使用pip install来安装PaddlePaddle

	* 对于需要**CPU版本PaddlePaddle**的用户：`pip install paddlepaddle`
	

	* 对于需要**GPU版本PaddlePaddle**的用户：(*仅支持ubuntu16.04/14.04*) `pip install paddlepaddle-gpu`
	> 1. 为防止出现nccl.h找不到的问题请首先按照NVIDIA[官方文档](https://developer.nvidia.com/nccl/nccl-download)安装nccl2. 			
	> 2. 如果您不规定pypi包版本号，我们默认为您提供支持Cuda 8/Cudnn v7的PaddlePaddle版本


	对于出现`Cannot uninstall 'six'.`问题的用户，可是由于您的系统中已有的Python安装问题造成的，请使用`pip install paddlepaddle --ignore-installed six`（CPU）或`pip 	install paddlepaddle --ignore-installed six`（GPU）解决。

2. 使用以下指令将默认装在`/usr/local/lib`下的`libmkldnn`放在`LD_LIBRARY_PATH中`:

	`export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`
	> 如果您的`libmkldnn`没有装在`/usr/local/lib`下，请使用`find / -name libmkldnn.so.0`从根目录开始找到`libmkldnn.so.0`之后将路径填到以下命令[dir]的的位置：`export LD_LIBRARY_PATH=[dir]:$LD_LIBRARY_PATH`

3. 使用以下指令将numpy的版本降至1.12.0 - 1.14.0之间：
	> 由于numpy支持造成numpy 1.15.0 及以上版本引发`shape warning`

	`pip install -U numpy==1.14.0`
	> 如果遇到`Python.h: No such file or directory`请设置`python.h`路径到`C_INCLUDE_PATH/CPLUS_INCLUDE_PATH`
	如果遇到其他问题请参见[常见问题表]()<!--TODO: Link 常见问题表到这里-->

现在您已经完成使用`pip install` 来安装的PaddlePaddle的过程

<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall PaddlePaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall PaddlePaddle-gpu`

<br/><br/>
### **CentOS下安装PaddlePaddle**

本说明将介绍如何在CentOS下安装PaddlePaddle,我们支持的CentOS系统需满足以下要求

请注意：在其他系统上的尝试可能会导致安装失败

* *64位台式机或笔记本电脑*
* *CentOS 6 / 7*

#### 确定要安装的PaddlePaddle版本
* 仅支持CPU的PaddlePaddle。如果您的计算机没有 NVIDIA® GPU，则只能安装此版本。如果您的计算机有GPU，
推荐您先安装CPU版本的PaddlePaddle，来检测您本地的环境是否适合。

* 支持GPU的PaddlePaddle，为了使PaddlePaddle程序运行的更加迅速，我们通过GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For CUDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *Cuda 工具包9.0配合cudnn v7*
	* *Cuda 工具包8.0配合cudnn v7*
	* *GPU运算能力超过1.0的硬件设备*



#### 选择如何安装PaddlePaddle
在CentOS的系统下我们提供3种不同的安装方式：

* Docker安装
* 原生pypi安装
* Docker源码编译安装


我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样


**使用原生pypi安装**，我们为您提供原生pypi安装方法，但它更依赖您的本机环境，可能会出现和您本机环境相关的一些问题

从**源码编译安装**，这是一种通过将PaddlePaddle源代码编译成为二进制文件，然后在安装这个二进制文件的过程，相比使用我们为您编译过的已经通过测试的二进制文件形式的PaddlePaddle，手动编译更为复杂，我们将在说明的最后详细为您解答
<br/><br/>
##### ***使用Docker进行安装***

我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样

为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**, 关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)

当您已经**正确安装Docker**后你就可以开始**使用Docker安装PaddlePaddle**啦

1. 使用以下指令拉取我们为您预安装好PaddlePaddle的镜像：


	* 对于需要**CPU版本的PaddlePaddle**的用户请使用以下指令拉取我们为您预安装好*PaddlePaddle For CPU*的镜像

		`docker pull paddlepaddle/paddle:latest`


	* 对于需要**GPU版本的PaddlePaddle**的用户请使用以下指令拉取我们为您预安装好*PaddlePaddle For GPU*的镜像

		`docker pull paddlepaddle/paddle:latest-gpu`


	* 您也可以通过以下指令拉取任意的我们提供的Docker镜像

		`docker pull paddlepaddle/paddle:[tag]`
		> （请把[tag]替换为[镜像表](https://hub.docker.com/r/paddlepaddle/paddle_manylinux_devel/tags/)中的内容）
2. 使用以下指令用已经拉取的镜像构建并进入Docker容器：

	`docker run --name [Name of container] -it -v $PWD:/work <imagename> /bin/bash`
	
	> 上述命令中，--name [Name of container] 设定Docker的名称；-it 参数说明容器已和本机交互式运行； -v $PWD:/work 指定将当前路径（Linux中$PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /work 目录； paddlepaddle/paddle 指定需要使用的image名称，/bin/bash是在Docker中要执行的命令.

3. 在Docker中 使用如下命令使用PaddlePaddle

	`docker start [Name of container]`
	> 启动之前创建的容器

	`docker attach [Name of container]`
	> 进入启动的容器

至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)
> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。


<br/><br/>
##### ***使用原生的pip安装PaddlePaddle***

首先, 我们使用以下指令来**检测本机的环境**是否适合安装PaddlePaddle

`uname -m && cat /etc/*release`
> 上面的命令将会显示本机的操作系统和位数信息，请确保您的计算机和本教程的要求一致


其次, 您的计算机需要满足以下要求：

*	Python2.7.x (devel)   
	
	> CentOS6需要编译Python成[共享库](https://blog.csdn.net/huzhenwei/article/details/7339548)
*	Pip >= 9.0.1
	> 您的CentOS上可能已经安装pip请使用pip -V来确认我们建议使用pip 9.0.1或更高版本来安装

	更新yum的源：   `yum update`

	使用以下命令安装或升级Python和pip到需要的版本： `sudo yum install python-devel python-pip`
	> 即使您的环境中已经有`Python2.7`也需要安装`python devel`

下面将说明如何安装PaddlePaddle

1. 使用pip install来安装PaddlePaddle
	
	* 对于需要**CPU版本PaddlePaddle**的用户：`pip install paddlepaddle`


	* 对于需要**GPU版本PaddlePaddle**的用户: `pip install paddlepaddle-gpu`
	> 1. 为防止出现nccl.h找不到的问题请首先按照NVIDIA[官方文档](https://developer.nvidia.com/nccl/nccl-download)安装nccl2.
	> 2. 如果您不规定pypi包版本号，我们默认为您提供支持Cuda 8/Cudnn v7的PaddlePaddle版本

	对于出现`Cannot uninstall 'six'.`问题的用户，可是由于您的系统中已有的Python安装问题造	成的，请使用`pip install paddlepaddle --ignore-installed six`（CPU）或`pip 	install paddlepaddle-gpu --ignore-installed six`（GPU）解决。

2. 使用以下指令将默认装在`/usr/local/lib`下的`libmkldnn`放在`LD_LIBRARY_PATH中`:

	`export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`
	> 如果您的`libmkldnn`没有装在`/usr/local/lib`下，请使用`find / -name libmkldnn.so.0`从根目录开始找到`libmkldnn.so.0`之后将路径填到以下命令[dir]的的位置：`export LD_LIBRARY_PATH=[dir]:$LD_LIBRARY_PATH`

3. 使用以下指令将numpy的版本降至1.12.0-1.14.0之间：
	> 由于numpy支持造成numpy 1.15.0 及以上版本引发`shape warning`

	`pip install -U numpy==1.14.0`
	> 如果遇到`Python.h: No such file or directory`请设置`python.h`路径到`C_INCLUDE_PATH/CPLUS_INCLUDE_PATH`
	如果遇到其他问题请参见[常见问题表]()<!--TODO：Link 常见问题表到这里-->

现在您已经完成通过`pip install` 来安装的PaddlePaddle的过程


<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入Python解释器，然后使用`import paddle.fluid` 验证是否安装成功

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall PaddlePaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall PaddlePaddle-gpu`

<br/><br/>
## **从源码编译PaddlePaddle**
我们也为您提供了从源码编译的方式，但不推荐您使用这种方式，这是因为您的本机环境多种多样，在编译源码时易出现复杂的问题而造成安装失败
      
***       
### **Ubuntu下从源码编译PaddlePaddle**
本说明将介绍如何在Ubuntu下编译PaddlePaddle，我们支持的Ubuntu系统需满足以下要求

* 64位台式机或笔记本电脑
* Ubuntu 14.04/16.04/18.04（这涉及到相关工具是否能被正常安装）

#### 确定要编译的PaddlePaddle版本
* **仅支持CPU的PaddlePaddle**，如果您的系统没有 NVIDIA® GPU，则必须安装此版本。而此版本较GPU版本更加容易安
因此即使您的计算机上拥有GPU我们也推荐您先安装CPU版本的PaddlePaddle来检测您本地的环境是否适合。

* **支持GPU的PaddlePaddle**，为了使得PaddlePaddle程序运行的更加迅速，我们通常使用GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For CUDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *Cuda 工具包9.0配合cudnn v7*
	* *Cuda 工具包8.0配合cudnn v7*
	* *GPU运算能力超过1.0的硬件设备*

#### 选择如何编译PaddlePaddle
在Ubuntu的系统下我们提供两种不同的编译方式：

* 直接本机源码编译
* Docker源码编译

我们更加推荐**使用Docker进行编译**，因为我们在把工具和配置都安装在一个 Docker image 里。这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。有人用虚拟机来类比 Docker。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样

我们也提供了可以从**本机直接源码编译**的方法，但是由于在本机上的情况更加复杂，我们只对特定系统提供了支持。
<br/><br/>
##### ***使用Docker进行编译***
为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**, 关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)

当您已经**正确安装Docker**后你就可以开始**使用Docker编译PaddlePaddle**啦

1. 请首先选择您希望储存PaddlePaddle的路径，然后在该路径下使用以下命令将PaddlePaddle的源码从github克隆到本地当前目录下名为Paddle的文件夹中：

	`git clone https://github.com/PaddlePaddle/Paddle.git`

2. 进入Paddle目录下： `cd Paddle`

3. 利用我们提供的镜像（使用该命令您可以不必提前下载镜像）：

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it paddlepaddle/paddle:latest-dev /bin/bash`
	> --name paddle-test为您创建的Docker容器命名为paddle-test，-v $PWD:/paddle 将当前目录挂载到Docker容器中的/paddle目录下，-it 与宿主机保持交互状态，paddlepaddle/paddle:latest-dev 使用名为paddlepaddle/paddle:latest-dev的镜像创建Docker容器，/bin/bash 进入容器后启动/bin/bash命令

4. 进入Docker后进入paddle目录下：`cd paddle`

5. 切换到较稳定release分支下进行编译：

	`git checkout release/0.14.0`

6. 创建并进入/paddle/build路径下：

	`mkdir -p /paddle/build && cd /paddle/build`

7. 使用以下命令安装相关依赖：

	`pip install numpy==1.14.0`
	> 安装numpy 1.14.0，由于目前numpy1.15.0会引起大量warning，因此在numpy修复该问题前我们先使用numpy 1.14.0

	`pip install protobuf==3.1.0`
	> 安装protobuf 3.1.0

	`apt install patchelf`
	> 安装patchelf, PatchELF is a small utility to modify the dynamic linker and RPATH of ELF executables

8. 执行cmake：
	>具体编译选项含义请参见[编译选项表]()<!--TODO: Link 编译选项表到这里-->


	*  对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF`


	* 对于需要编译**GPU版本PaddlePaddle**的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF`


9. 执行编译：

	`make -j$(nproc)`
	> 使用多核编译

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install （whl包的名字）`

至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。

恭喜您，现在您已经完成使用Docker编译PaddlePaddle的过程

<br/><br/>
##### ***本机编译***

1. 检查您的计算机和操作系统是否符合我们支持的编译标准： `uname -m && cat /etc/*release`

2. 更新`apt`的源： `apt update`

2. 我们支持使用virtualenv进行编译安装，首先请使用以下命令创建一个名为`paddle-venv`的虚环境：

	* 安装Python-dev: `apt install python-dev`

	* 安装pip: `apt install python-pip`

	* 安装虚环境`virtualenv`以及`virtualenvwrapper`并创建名为`paddle-venv`的虚环境：

		1.  `apt install virtualenv` 或 `pip install virtualenv`
		2.  `apt install virtualenvwrapper` 或 `pip install virtualenvwrapper`
		3.  找到`virtualenvwrapper.sh`： `find / -name virtualenvwrapper.sh`
		4.  查看`virtualenvwrapper.sh`中的安装方法： `cat vitualenvwrapper.sh`
		5.  安装`virtualwrapper`
		6.  创建名为`paddle-venv`的虚环境： `mkvirtualenv paddle-venv`


3. 进入虚环境：`workon paddle-venv`         


4. **执行编译前**请您确认在虚环境中安装有[安装依赖表]()中提到的相关依赖：<!--TODO：Link 安装依赖表到这里-->

	* 这里特别提供`patchELF`的安装方法，其他的依赖可以使用`apt install`或者`pip install` 后跟依赖名称和版本安装

		`apt install patchelf`
		> 不能使用apt安装的用户请参见patchElF github[官方文档](https://gist.github.com/ruario/80fefd174b3395d34c14)

5. 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下：

	- `git clone https://github.com/PaddlePaddle/Paddle.git`

	- `cd Paddle`

6. 切换到较稳定release分支下进行编译：

	`git checkout release/0.14.0`

7. 并且请创建并进入一个叫build的目录下：

	`mkdir build && cd build`

8. 执行cmake：
	>具体编译选项含义请参见[编译选项表]()<!--TODO：Link 安装选项表到这里-->


	*  对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF`.


	* 对于需要编译**GPU版本PaddlePaddle**的用户：(*仅支持ubuntu16.04/14.04*)

		1. 请确保您已经从[nvidia官网](https://developer.nvidia.com/nccl)下载对应版本的`nccl2`并按照官网流程安装。
		2. 如果您已经正确安装了`nccl2`，就可以开始cmake了：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF`

9. 使用以下命令来编译：

	`make -j$(nproc)`

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install （whl包的名字）`

恭喜您，现在您已经完成使本机编译PaddlePaddle的过程啦

<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入Python解释器，然后使用`import paddle.fluid` 验证是否安装成功

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall PaddlePaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall PaddlePaddle-gpu`


<br/><br/>
### **CentOS下从源码编译PaddlePaddle**
本说明将介绍如何在CentOS下编译PaddlePaddle,我们支持的Ubuntu系统需满足以下要求：

* 64位台式机或笔记本电脑
* CentOS 7 / 6（这涉及到相关工具是否能被正常安装）

#### 确定要编译的PaddlePaddle版本
* 仅支持CPU的PaddlePaddle，如果您的计算机没有 NVIDIA® GPU，则只能安装此版本。如果您的计算机有GPU， 推荐您先安装CPU版本的PaddlePaddle，来检测您本地的环境是否适合。

* 支持GPU的PaddlePaddle，为了使得PaddlePaddle程序运行的更加迅速，我们通常使用GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For CUDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *Cuda 工具包9.0配合cudnn v7*
	* *Cuda 工具包8.0配合cudnn v7*
	* *GPU运算能力超过1.0的硬件设备*

#### 选择如何编译PaddlePaddle
在CentOS 7的系统下我们提供1种的编译方式：

<!--* 直接本机源码编译-->
* Docker源码编译

我们更加推荐**使用Docker进行编译**，因为我们在把工具和配置都安装在一个 Docker image 里。这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样

<!--同样对于那些出于各种原因不能够安装Docker的用户我们也提供了可以从**本机直接源码编译**的方法，但是由于在本机上的情况更加复杂，因此我们只支持特定的系统-->
<br/><br/>
##### ***使用Docker进行编译***

为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**, 关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)

当您已经**正确安装Docker**后你就可以开始**使用Docker编译PaddlePaddle**啦

1. 请首先选择您希望储存PaddlePaddle的路径，然后在该路径下使用以下命令将PaddlePaddle的源码从github克隆到本地当前目录下名为Paddle的文件夹中：

	`git clone https://github.com/PaddlePaddle/Paddle.git`

2. 进入Paddle目录下： `cd Paddle`

3. 利用我们提供的镜像（使用该命令您可以不必提前下载镜像）：

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it paddlepaddle/paddle:latest-dev /bin/bash`
	> --name paddle-test为您创建的Docker容器命名为paddle-test，-v $PWD:/paddle 将当前目录挂载到Docker容器中的/paddle目录下，-it 与宿主机保持交互状态，paddlepaddle/paddle:latest-dev 使用名为paddlepaddle/paddle:latest-dev的镜像创建Docker容器，/bin/bash 进入容器后启动/bin/bash命令

4. 进入Docker后进入paddle目录下：`cd paddle`

5. 切换到较稳定release分支下进行编译：

	`git checkout release/0.14.0`

6. 创建并进入/paddle/build路径下：

	`mkdir -p /paddle/build && cd /paddle/build`

7. 使用以下命令安装相关依赖：

	`pip install numpy==1.14.0`
	> 安装numpy 1.14.0，由于目前numpy1.15.0会引起大量warning，因此在numpy修复该问题前我们先使用numpy 1.14.0

	`pip install protobuf==3.1.0`
	> 安装protobuf 3.1.0

	`apt install patchelf`
	> 安装patchelf, PatchELF is a small utility to modify the dynamic linker and RPATH of ELF executables

8. 执行cmake：
	>具体编译选项含义请参见[编译选项表]()<!--TODO： Link 编译选项表到这里-->


	*  对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF`
		> 我们目前不支持CentOS下GPU版本PaddlePaddle的编译

<!--	* 对于需要编译***GPU版本PaddlePaddle***的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF`-->

9. 执行编译：

	`make -j$(nproc)`
	> 使用多核编译

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install （whl包的名字）`

至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。

恭喜您，现在您已经完成使用Docker编译PaddlePaddle的过程

<br/><br/>
##### ***验证安装***
安装完成后您可以使用：`python` 进入Python解释器，然后使用`import paddle.fluid` 验证是否安装成功

<br/><br/>
##### ***如何卸载PaddlePaddle***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall PaddlePaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall PaddlePaddle-gpu`

</br></br>
### 附录：

## **编译依赖表**

| 依赖包名称 | 版本.               | 说明                    |  安装命令  |
| :--      | :-----:             |:------                 |:----:|
| CMake    | 3.4                 |                        | |
| GCC      | 4.9 或更高           |推荐使用CentOS的devtools2 | |
| Python   | 2.7.x.              | 依赖libpython2.7.so     | |
| SWIG     | 最低 2.0              |    N/A                 | `apt install swig` 或 `yum install swig`|
| wget     | any                 |    N/A                 | `apt install wget` 或 `yum install wget` |
| openblas | any                 |    N/A                 | |
| pip      | 最低9.0.1             |    N/A                 | `apt install python-pip` 或 `yum install Python-pip` |
| numpy    | 最低1.12.0，最高1.14.0 |    N/A                 | `pip install numpy==1.14.0` |
| protobuf | 3.1.0               |    N/A                 | `pip install protobuf==3.1.0`|
| wheel    | any                 |    N/A                 | `pip install wheel`|
| patchELF | any                 |    N/A                 | `apt install patchelf` 或参见github [patchelf官方教程](https://gist.github.com/ruario/80fefd174b3395d34c14) |
| go       | >=1.8               |  可选                   | |

***
</br></br>
## **编译选项表**
| 选项             | 说明                                               |  默认值        |
| :--              | :-----                                            |:------:        |
| WITH_GPU         | 是否支持GPU	                                     |ON
| WITH_C_API       |	是否仅编译CAPI                                   |OFF
| WITH_DOUBLE	      | 是否使用双精度浮点数                                | OFF
| WITH_DSO         | 是否运行时动态加载CUDA动态库，而非静态加载CUDA动态库。   | ON.
| WITH_AVX	      |是否编译含有AVX指令集的PaddlePaddle二进制文件.         |ON
| WITH_PYTHON	      |是否内嵌PYTHON解释器	                              |ON
| WITH_STYLE_CHECK |	是否编译时进行代码风格检查                         |ON
| WITH_TESTING     | 	是否开启单元测试	                              |OFF
| WITH_DOC	      |是否编译中英文文档	                                 |OFF
| WITH_SWIG_PY	  | 是否编译PYTHON的SWIG接口，该接口可用于预测和定制化训练  |Auto
| WITH_GOLANG	      | 是否编译go语言的可容错parameter server             |OFF
| WITH_MKL         |	是否使用MKL数学库，如果为否则是用OpenBLAS          |ON

**BLAS**

PaddlePaddle支持 [MKL](https://software.intel.com/en-us/mkl) 和 [OpenBlAS](http://www.openblas.net) 两种BLAS库。默认使用MKL。如果使用MKL并且机器含有AVX2指令集， 还会下载MKL-DNN数学库，详细参考[这里](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/mkldnn#cmake) 。

如果关闭MKL，则会使用OpenBLAS作为BLAS库。

**CUDA/cuDNN**

PaddlePaddle在编译时/运行时会自动找到系统中安装的CUDA和cuDNN库进行编译和执行。 使用参数 `-DCUDA_ARCH_NAME=Auto` 可以指定开启自动检测SM架构，加速编译。

PaddlePaddle可以使用cuDNN v5.1之后的任何一个版本来编译运行，但尽量请保持编译和运行使用的cuDNN是同一个版本。 我们推荐使用最新版本的cuDNN。

**编译选项的设置**

PaddePaddle通过编译时指定路径来实现引用各种BLAS/CUDA/cuDNN库。cmake编译时，首先在系统路径（ `/usr/liby` 和 `/usr/local/lib` ）中搜索这几个库，同时也会读取相关路径变量来进行搜索。 通过使用`-D`命令可以设置，例如：

> `cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCUDNN_ROOT=/opt/cudnnv5`

**注意**：这几个编译选项的设置，只在第一次cmake的时候有效。如果之后想要重新设置，推荐清理整个编译目录（ rm -rf ）后，再指定。


***
</br></br>
## **安装包列表**
| 版本号             | 版本说明                                       |
| :--               | :-----                                        |
|paddlepaddle-gpu==0.14.0       |	使用CUDA 9.0和cuDNN 7编译的0.14.0版本|
|paddlepaddle-gpu==0.14.0.post87|	使用CUDA 8.0和cuDNN 7编译的0.14.0版本|
|paddlepaddle-gpu==0.14.0.post85|	使用CUDA 8.0和cuDNN 5编译的0.14.0版本|
|paddlepaddle-gpu==0.13.0.      |	使用CUDA 9.0和cuDNN 7编译的0.13.0版本|
|paddlepaddle-gpu==0.12.0.      |	使用CUDA 8.0和cuDNN 5编译的0.12.0版本|
|paddlepaddle-gpu==0.11.0.post87|	使用CUDA 8.0和cuDNN 7编译的0.11.0版本|
|paddlepaddle-gpu==0.11.0.post8.|	使用CUDA 8.0和cuDNN 5编译的0.11.0版本|
|paddlepaddle-gpu==0.11.0.      |	使用CUDA 7.5和cuDNN 5编译的0.11.0版本|

您可以在 [Release History](https://pypi.org/project/paddlepaddle-gpu/#history) 中找到paddlepaddle-gpu的各个发行版本。
