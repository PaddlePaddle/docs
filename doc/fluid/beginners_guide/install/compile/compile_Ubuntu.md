# **Ubuntu下从源码编译**

## 环境准备

* *64位操作系统*
* *Ubuntu 14.04 (GPU版本只针对CUDA 8, CUDA 10支持)*
* *Ubuntu 16.04*
* *Ubuntu 18.04（GPU版本只针对CUDA10支持)*
* *Python（64 bit） 2.7/3.5.1+/3.6/3.7*
* *pip或pip3（64 bit） >= 9.0.1*

## 选择CPU/GPU

## 选择CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请安装CPU版的PaddlePaddle

* 如果您的计算机有 NVIDIA® GPU，并且满足以下条件，推荐安装GPU版的PaddlePaddle
	* *CUDA 工具包10.0配合cuDNN v7.3+(如需多卡支持，需配合NCCL2.3.7及更高)*
	* *CUDA 工具包9.0配合cuDNN v7.3+(如需多卡支持，需配合NCCL2.3.7及更高)*
	* *CUDA 工具包8.0配合cuDNN v7.3+(如需多卡支持，需配合NCCL2.1.15-2.2.13）*
	* *GPU运算能力超过1.0的硬件设备*

		您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

* 请确保您已经正确安装nccl2，或者按照以下指令安装nccl2（这里提供的是ubuntu 16.04，CUDA9，cuDNN7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl/nccl-download):


		wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
		dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb`
		sudo apt-get install -y libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0

## 安装步骤

在Ubuntu的系统下有2种编译方式：

* 用Docker编译（暂不支持Ubuntu18.04下GPU版本）
* 本机编译

<a name="ubt_docker"></a>
### ***用Docker编译***

[Docker](https://docs.docker.com/install/)是一个开源的应用容器引擎。使用Docker，既可以将PaddlePaddle的安装&使用与系统环境隔离，也可以与主机共享GPU、网络等资源

使用Docker编译PaddlePaddle，您需要：

- 在本地主机上[安装Docker](https://hub.docker.com/search/?type=edition&offering=community)

- 如需在Linux开启GPU支持，请[安装nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

请您按照以下步骤安装：

1. 请首先选择您希望储存PaddlePaddle的路径，然后在该路径下使用以下命令将PaddlePaddle的源码从github克隆到本地当前目录下名为Paddle的文件夹中：

	`git clone https://github.com/PaddlePaddle/Paddle.git`

2. 进入Paddle目录下： `cd Paddle`

3. 创建并进入满足编译环境的Docker容器：

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`

	> --name paddle-test为您创建的Docker容器命名为paddle-test，-v $PWD:/paddle 将当前目录挂载到Docker容器中的/paddle目录下（Linux中PWD变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)），-it 与宿主机保持交互状态，`hub.baidubce.com/paddlepaddle/paddle:latest-dev` 使用名为`hub.baidubce.com/paddlepaddle/paddle:latest-dev`的镜像创建Docker容器，/bin/bash 进入容器后启动/bin/bash命令。

4. 进入Docker后进入paddle目录下：

	`cd paddle`

5. 切换到较稳定release分支下进行编译：

	`git checkout [分支名]`

	例如：

	`git checkout release/1.5`

	注意：python3.6、python3.7版本从release/1.2分支开始支持

6. 创建并进入/paddle/build路径下：

	`mkdir -p /paddle/build && cd /paddle/build`

7. 使用以下命令安装相关依赖：

		For Python2: pip install protobuf
		For Python3: pip3.5 install protobuf

	注意：以上用Python3.5命令来举例，如您的Python版本为3.6/3.7，请将上述命令中的Python3.5改成Python3.6/Python3.7

	> 安装protobuf。

	`apt install patchelf`

	> 安装patchelf
	这是一个小而实用的程序，用于修改ELF可执行文件的动态链接器和RPATH

8. 执行cmake：

	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)

	>请注意修改参数`-DPY_VERSION`为您希望编译使用的python版本, 例如`-DPY_VERSION=3.5`表示python版本为3.5.x

	*  编译**CPU版本PaddlePaddle**：

		`cmake .. -DPY_VERSION=3.5 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

	*  编译**GPU版本PaddlePaddle**：

		`cmake .. -DPY_VERSION=3.5 -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

9. 执行编译：

	`make -j$(nproc)`

	> 使用多核编译

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

		For Python2: pip install -U（whl包的名字）
		For Python3: pip3.5 install -U（whl包的名字）

	注意：以上涉及Python3的命令，用Python3.5来举例，如您的Python版本为3.6/3.7，请将上述命令中的Python3.5改成Python3.6/Python3.7

恭喜，至此您已完成PaddlePaddle的编译安装。您只需要进入Docker容器后运行PaddlePaddle，即可开始使用。更多Docker使用请参见[Docker官方文档](https://docs.docker.com)

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 来安装。

<a name="ubt_source"></a>
### ***本机编译***

1. 检查您的计算机和操作系统是否符合我们支持的编译标准： `uname -m && cat /etc/*release`

2. 更新`apt`的源： `apt update`, 并提前安装[OpenCV](https://opencv.org/releases.html)

3. 我们支持使用virtualenv进行编译安装，首先请使用以下命令创建一个名为`paddle-venv`的虚环境：

	* a. 安装Python-dev（请注意Ubuntu16.04下的python2.7不支持gcc4.8，请使用gcc5.4编译Paddle）:

			For Python2: apt install python-dev
			For Python3: apt install python3.5-dev

	* b. 安装pip: (请保证拥有9.0.1及以上版本的pip):

			For Python2: apt install python-pip
			For Python3: apt-get udpate && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt install curl && curl https://bootstrap.pypa.io/get-pip.py -o - | python3.5 && easy_install pip

	* c. 安装虚环境`virtualenv`以及`virtualenvwrapper`并创建名为`paddle-venv`的虚环境：

		1.  `apt install virtualenv` 或 `pip install virtualenv` 或 `pip3 install virtualenv`
		2.  `apt install virtualenvwrapper` 或 `pip install virtualenvwrapper` 或 `pip3 install virtualenvwrapper`
		3.  找到`virtualenvwrapper.sh`： `find / -name virtualenvwrapper.sh`
		4.  (Only for Python3) 设置虚环境的解释器路径：`export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.5`
		5.  查看`virtualenvwrapper.sh`中的安装方法： `cat virtualenvwrapper.sh`, 该shell文件中描述了步骤及命令
		6.  按照`virtualenvwrapper.sh`中的描述，安装`virtualwrapper`
		7.  设置VIRTUALENVWRAPPER_PYTHON：`export VIRTUALENVWRAPPER_PYTHON=[python-lib-path]:$PATH` （这里将[python-lib-path]的最后两级目录替换为/bin/)
		8.  创建名为`paddle-venv`的虚环境： `mkvirtualenv paddle-venv`

	注意：以上涉及Python3的命令，用Python3.5来举例，如您的Python版本为3.6/3.7，请将上述命令中的Python3.5改成Python3.6/Python3.7

4. 进入虚环境：`workon paddle-venv`

5. **执行编译前**请您确认在虚环境中安装有[编译依赖表](../Tables.html/#third_party)中提到的相关依赖:

	* 这里特别提供`patchELF`的安装方法，其他的依赖可以使用`apt install`或者`pip install` 后跟依赖名称和版本安装:

		`apt install patchelf`

		> 不能使用apt安装的用户请参见patchElF github[官方文档](https://gist.github.com/ruario/80fefd174b3395d34c14)

5. 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下：

	- `git clone https://github.com/PaddlePaddle/Paddle.git`

	- `cd Paddle`

6. 切换到较稳定release分支下进行编译，将中括号以及其中的内容替换为**目标分支名**：

	`git checkout [分支名]`

	例如：

	`git checkout release/1.5`

7. 并且请创建并进入一个叫build的目录下：

	`mkdir build && cd build`

8. 执行cmake：

	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)

	*  对于需要编译**CPU版本PaddlePaddle**的用户：

			For Python2: cmake .. -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
			For Python3: cmake .. -DPY_VERSION=3.5 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release

	* 对于需要编译**GPU版本PaddlePaddle**的用户：(*仅支持ubuntu16.04/14.04*)

		1. 请确保您已经正确安装nccl2，或者按照以下指令安装nccl2（这里提供的是ubuntu 16.04，CUDA9，cuDNN7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl/nccl-download):


			i. `wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb`
			
			ii.  `dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb`


			iii. `sudo apt-get install -y libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0`

		2. 如果您已经正确安装了`nccl2`，就可以开始cmake了：(*For Python3: 请给PY_VERSION参数配置正确的python版本*)

				For Python2: cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
				For Python3: cmake .. -DPY_VERSION=3.5 -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release

	注意：以上涉及Python3的命令，用Python3.5来举例，如您的Python版本为3.6/3.7，请将上述命令中的Python3.5改成Python3.6/Python3.7

9. 使用以下命令来编译：

	`make -j$(nproc)`

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install -U（whl包的名字）`或`pip3 install -U（whl包的名字）`

恭喜，至此您已完成PaddlePaddle的编译安装

## ***验证安装***
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle.fluid as fluid` ，再输入
 `fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。

## ***如何卸载***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall paddlepaddle-gpu` 或 `pip3 uninstall paddlepaddle-gpu`

使用Docker安装PaddlePaddle的用户，请进入包含PaddlePaddle的容器中使用上述命令，注意使用对应版本的pip
