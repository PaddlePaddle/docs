***

# **Ubuntu下从源码编译**

本说明将介绍如何在*64位台式机或笔记本电脑*以及Ubuntu系统下编译PaddlePaddle，我们支持的Ubuntu系统需满足以下要求：

* Ubuntu 14.04/16.04/18.04（这涉及到相关工具是否能被正常安装）

## 确定要编译的版本
* **仅支持CPU的PaddlePaddle**，如果您的系统没有 NVIDIA® GPU，则必须安装此版本。而此版本较GPU版本更加容易安
因此即使您的计算机上拥有GPU我们也推荐您先安装CPU版本的PaddlePaddle来检测您本地的环境是否适合。

* **支持GPU的PaddlePaddle**，为了使得PaddlePaddle程序运行的更加迅速，我们通常使用GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *CUDA 工具包9.0配合cuDNN v7*
	* *CUDA 工具包8.0配合cuDNN v7*
	* *GPU运算能力超过1.0的硬件设备*

## 选择如何编译
在Ubuntu的系统下我们提供2种编译方式：

* Docker源码编译
* 直接本机源码编译（不支持ubuntu18.04下GPU版本）      

我们更加推荐**使用Docker进行编译**，因为我们在把工具和配置都安装在一个 Docker image 里。这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。有人用虚拟机来类比 Docker。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。



我们也提供了可以从**本机直接源码编译**的方法，但是由于在本机上的情况更加复杂，我们只对特定系统提供了支持。

<a name="ubt_docker"></a>

<br/><br/>
### ***使用Docker编译***
为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**，关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)


> 请注意，要安装和使用支持 GPU 的PaddlePaddle版本，您必须先安装[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)



当您已经**正确安装Docker**后你就可以开始**使用Docker编译PaddlePaddle**：

1. 请首先选择您希望储存PaddlePaddle的路径，然后在该路径下使用以下命令将PaddlePaddle的源码从github克隆到本地当前目录下名为Paddle的文件夹中：

	`git clone https://github.com/PaddlePaddle/Paddle.git`

2. 进入Paddle目录下： `cd Paddle`

3. 利用我们提供的镜像（使用该命令您可以不必提前下载镜像）：

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`

	> --name paddle-test为您创建的Docker容器命名为paddle-test，-v $PWD:/paddle 将当前目录挂载到Docker容器中的/paddle目录下（Linux中PWD变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)），-it 与宿主机保持交互状态，`hub.baidubce.com/paddlepaddle/paddle:latest-dev` 使用名为`hub.baidubce.com/paddlepaddle/paddle:latest-dev`的镜像创建Docker容器，/bin/bash 进入容器后启动/bin/bash命令。

4. 进入Docker后进入paddle目录下：`cd paddle`

5. 切换到较稳定release分支下进行编译：(注意，python3.6、python3.7版本是从1.2.0分支开始支持)

	`git checkout release/1.2.0`

6. 创建并进入/paddle/build路径下：

	`mkdir -p /paddle/build && cd /paddle/build`

7. 使用以下命令安装相关依赖：

		For Python2: pip install protobuf==3.1.0
		For Python3: pip3 install protobuf==3.1.0

	> 安装protobuf 3.1.0。

	`apt install patchelf`

	> 安装patchelf，PatchELF 是一个小而实用的程序，用于修改ELF可执行文件的动态链接器和RPATH。

8. 执行cmake：

	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)<!--TODO: Link 编译选项表到这里-->
	>请注意修改参数`-DPY_VERSION`为您当前环境下使用的python版本

	*  对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

	* 对于需要编译**GPU版本PaddlePaddle**的用户：

		`cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`


9. 执行编译：

	`make -j$(nproc)`

	> 使用多核编译

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

		For Python2: pip install （whl包的名字）
		For Python3: pip3 install （whl包的名字）


至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)。

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。

恭喜您，现在您已经完成使用Docker编译PaddlePaddle的过程。

<a name="ubt_source"></a>

<br/><br/>
### ***本机编译***

**请严格按照以下指令顺序执行**

1. 检查您的计算机和操作系统是否符合我们支持的编译标准： `uname -m && cat /etc/*release`

2. 更新`apt`的源： `apt update`

3. 我们支持使用virtualenv进行编译安装，首先请使用以下命令创建一个名为`paddle-venv`的虚环境：

	* a. 安装Python-dev:（请安装与当前环境python版本匹配的python3.x-dev）

			For Python2: apt install python-dev
			For Python3: apt install python3.5-dev

	* b. 安装pip: (请保证拥有9.0.1及以上版本的pip):（请注意修改对应python3的版本）

			For Python2: apt install python-pip
			For Python3: apt-get udpate && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt install curl && curl https://bootstrap.pypa.io/get-pip.py -o - | python3.5 && easy_install pip

	* c. 安装虚环境`virtualenv`以及`virtualenvwrapper`并创建名为`paddle-venv`的虚环境：(请注意修改python版本)

		1.  `apt install virtualenv` 或 `pip install virtualenv` 或 `pip3 install virtualenv`
		2.  `apt install virtualenvwrapper` 或 `pip install virtualenvwrapper` 或 `pip3 install virtualenvwrapper`
		3.  找到`virtualenvwrapper.sh`： `find / -name virtualenvwrapper.sh`
		4.  (Only for Python3) 设置虚环境的解释器路径：`export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.5`
		5.  查看`virtualenvwrapper.sh`中的安装方法： `cat virtualenvwrapper.sh`
		6.  按照`virtualenvwrapper.sh`中的安装方法安装`virtualwrapper`
		7.  创建名为`paddle-venv`的虚环境： `mkvirtualenv paddle-venv`

4. 进入虚环境：`workon paddle-venv`

5. **执行编译前**请您确认在虚环境中安装有[编译依赖表](../Tables.html/#third_party)中提到的相关依赖：<!--TODO：Link 安装依赖表到这里-->

	* 这里特别提供`patchELF`的安装方法，其他的依赖可以使用`apt install`或者`pip install` 后跟依赖名称和版本安装:

		`apt install patchelf`

		> 不能使用apt安装的用户请参见patchElF github[官方文档](https://gist.github.com/ruario/80fefd174b3395d34c14)

5. 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下：

	- `git clone https://github.com/PaddlePaddle/Paddle.git`

	- `cd Paddle`

6. 切换到较稳定release分支下进行编译，将中括号以及其中的内容替换为**目标分支名**：

	`git checkout [name of target branch]`

7. 并且请创建并进入一个叫build的目录下：

	`mkdir build && cd build`

8. 执行cmake：

	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)<!--TODO：Link 安装选项表到这里-->

	*  对于需要编译**CPU版本PaddlePaddle**的用户：(*For Python3: 请给PY_VERSION参数配置正确的python版本*)

			For Python2: cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
			For Python3: cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release

	* 对于需要编译**GPU版本PaddlePaddle**的用户：(*仅支持ubuntu16.04/14.04*)

		1. 请确保您已经正确安装nccl2，或者按照以下指令安装nccl2（这里提供的是ubuntu 16.04，CUDA9，cuDNN7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl/nccl-download):
			i. `wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb`
			ii.  `dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb`
			iii. `sudo apt-get install -y libnccl2=2.2.13-1+cuda9.0 libnccl-dev=2.2.13-1+cuda9.0`

		2. 如果您已经正确安装了`nccl2`，就可以开始cmake了：(*For Python3: 请给PY_VERSION参数配置正确的python版本*)

				For Python2: cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
				For Python3: cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
				
			>`-DPY_VERSION=3.5`请修改为安装环境的Python版本

9. 使用以下命令来编译：

	`make -j$(nproc)`

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install （whl包的名字）`或`pip3 install （whl包的名字）`

恭喜您，现在您已经完成使本机编译PaddlePaddle的过程了。

<br/><br/>
## ***验证安装***
安装完成后您可以使用：`python` 或 `python3` 进入Python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
## ***如何卸载***
请使用以下命令卸载PaddlePaddle（使用docker安装PaddlePaddle的用户请进入包含PaddlePaddle的容器中使用以下命令）：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall paddlepaddle-gpu` 或 `pip3 uninstall paddlepaddle-gpu`

