# **CentOS下从源码编译**

## 环境准备

* *64位操作系统*
* *CentOS 6 / 7*
* *Python（64 bit） 2.7/3.5.1+/3.6/3.7*
* *pip或pip3（64 bit） >= 9.0.1*

## 选择CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请安装CPU版本的PaddlePaddle

* 如果您的计算机有NVIDIA® GPU，请确保满足以下条件以编译GPU版PaddlePaddle
	
	* *CUDA 工具包10.0配合cuDNN v7.3+(如需多卡支持，需配合NCCL2.3.7及更高)*
	* *CUDA 工具包9.0配合cuDNN v7.3+(如需多卡支持，需配合NCCL2.3.7及更高)*
	* *CUDA 工具包8.0配合cuDNN v7.3+(官方不支持多卡）*
	* *GPU运算能力超过1.0的硬件设备*

		您可参考NVIDIA官方文档了解CUDA和CUDNN的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

* 请确保您已经正确安装nccl2，或者按照以下指令安装nccl2（这里提供的是CentOS 7，CUDA9，cuDNN7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl/nccl-download):


		wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
		rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
		sudo apt-get install -y libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0
		yum update -y
		yum install -y libnccl-2.3.7-2+cuda9.0 libnccl-devel-2.3.7-2+cuda9.0 libnccl-static-2.3.7-2+cuda9.0


## 安装步骤

在CentOS的系统下有2种编译方式：

* 使用Docker编译
* 本机编译（不提供在CentOS 6下编译中遇到问题的支持）

<a name="ct_docker"></a>
### ***使用Docker编译***

[Docker](https://docs.docker.com/install/)是一个开源的应用容器引擎。使用Docker，既可以将PaddlePaddle的安装&使用与系统环境隔离，也可以与主机共享GPU、网络等资源

使用Docker编译PaddlePaddle，您需要：

- 在本地主机上[安装Docker](https://hub.docker.com/search/?type=edition&offering=community)

- 如需在Linux开启GPU支持，请[安装nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

请您按照以下步骤安装：

1. 请首先选择您希望储存PaddlePaddle的路径，然后在该路径下使用以下命令将PaddlePaddle的源码从github克隆到本地当前目录下名为Paddle的文件夹中：

	`git clone https://github.com/PaddlePaddle/Paddle.git`

2. 进入Paddle目录下： `cd Paddle`

3. 创建并进入已配置好编译环境的Docker容器：

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`

	> --name paddle-test为您创建的Docker容器命名为paddle-test，-v $PWD:/paddle 将当前目录挂载到Docker容器中的/paddle目录下（Linux中PWD变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)），-it 与宿主机保持交互状态，`hub.baidubce.com/paddlepaddle/paddle` 使用名为`hub.baidubce.com/paddlepaddle/paddle:latest-dev`的镜像创建Docker容器，/bin/bash 进入容器后启动/bin/bash命令。

4. 进入Docker后进入paddle目录下：

	`cd paddle`

5. 切换到较稳定版本下进行编译：

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

	> 安装patchelf，PatchELF 是一个小而实用的程序，用于修改ELF可执行文件的动态链接器和RPATH。

8. 执行cmake：

	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)
	>请注意修改参数`-DPY_VERSION`为您希望编译使用的python版本,  例如`-DPY_VERSION=3.5`表示python版本为3.5.x

	* 对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DPY_VERSION=3.5 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

	> 我们目前不支持CentOS下使用Docker编译GPU版本的PaddlePaddle

9. 执行编译：

	`make -j$(nproc)`

	> 使用多核编译

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

		For Python2: pip install -U（whl包的名字）
		For Python3: pip3.5 install -U（whl包的名字）

	注意：以上涉及Python3的命令，用Python3.5来举例，如您的Python版本为3.6/3.7，请将上述命令中的Python3.5改成Python3.6/Python3.7

恭喜，至此您已完成PaddlePaddle的编译安装。您只需要进入Docker容器后运行PaddlePaddle，即可开始使用。更多Docker使用请参见[Docker官方文档](https://docs.docker.com)

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 来安装

<a name="ct_source"></a>
### ***本机编译***

1. 检查您的计算机和操作系统是否符合我们支持的编译标准： `uname -m && cat /etc/*release`

2. 更新`yum`的源： `yum update`, 并添加必要的yum源：`yum install -y epel-release`, 并提前安装[OpenCV](https://opencv.org/releases.html)

3. 安装必要的工具`bzip2`以及`make`： `yum install -y bzip2` ， `yum install -y make`

4. 我们支持使用virtualenv进行编译安装，首先请使用以下命令创建一个名为`paddle-venv`的虚环境：

	* a. 安装Python-dev:

			For Python2: yum install python-devel
			For Python3: (请参照Python官方流程安装）

	* b. 安装pip:

			For Python2: yum install python-pip (请保证拥有9.0.1及以上的pip版本)
			For Python3: (请参照Python官方流程安装, 并保证拥有9.0.1及以上的pip3版本，请注意，python3.6及以上版本环境下，pip3并不一定对应python版本，如python3.7下默认只有pip3.7）

	* c.（Only For Python3）设置Python3相关的环境变量，这里以python3.5版本示例，请替换成您使用的版本（3.6、3.7）：

		1. 首先使用``` find `dirname $(dirname
			$(which python3))` -name "libpython3.so"```找到Python lib的路径，如果是3.6或3.7，请将`python3`改成`python3.6`或`python3.7`，然后将下面[python-lib-path]替换为找到文件路径

		2. 设置PYTHON_LIBRARIES：`export PYTHON_LIBRARY=[python-lib-path]`

		3. 其次使用```find `dirname $(dirname
			$(which python3))`/include -name "python3.5m"```找到Python Include的路径，请注意python版本，然后将下面[python-include-path]替换为找到文件路径
		4. 设置PYTHON_INCLUDE_DIR: `export PYTHON_INCLUDE_DIRS=[python-include-path]`

		5. 设置系统环境变量路径：`export PATH=[python-lib-path]:$PATH` （这里将[python-lib-path]的最后两级目录替换为/bin/)

	* d. 安装虚环境`virtualenv`以及`virtualenvwrapper`并创建名为`paddle-venv`的虚环境：(请注意对应python版本的pip3的命令，如pip3.6、pip3.7)

		1.  `pip install virtualenv` 或 `pip3 install virtualenv`
		2.  `pip install virtualenvwrapper` 或 `pip3 install virtualenvwrapper`
		3.  找到`virtualenvwrapper.sh`： `find / -name virtualenvwrapper.sh`（请找到对应Python版本的`virtualenvwrapper.sh`）
		4.  查看`virtualenvwrapper.sh`中的安装方法： `cat vitualenvwrapper.sh`, 该shell文件中描述了步骤及命令
		5.  按照`virtualenvwrapper.sh`中的描述，安装`virtualwrapper`
		6.  设置VIRTUALENVWRAPPER_PYTHON：`export VIRTUALENVWRAPPER_PYTHON=[python-lib-path]:$PATH` （这里将[python-lib-path]的最后两级目录替换为/bin/)
		7.  创建名为`paddle-venv`的虚环境： `mkvirtualenv paddle-venv`

5. 进入虚环境：`workon paddle-venv`

6. **执行编译前**请您确认在虚环境中安装有[编译依赖表](../Tables.html/#third_party)中提到的相关依赖：

	* 这里特别提供`patchELF`的安装方法，其他的依赖可以使用`yum install`或者`pip install`/`pip3 install` 后跟依赖名称和版本安装:

        `yum install patchelf`
		> 不能使用apt安装的用户请参见patchElF github[官方文档](https://gist.github.com/ruario/80fefd174b3395d34c14)

7. 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下：

	- `git clone https://github.com/PaddlePaddle/Paddle.git`

	- `cd Paddle`

8. 切换到较稳定release分支下进行编译：

	`git checkout [分支名]`

	例如：

	`git checkout release/1.5`

9. 并且请创建并进入一个叫build的目录下：

	`mkdir build && cd build`

10. 执行cmake：

	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)

	*  对于需要编译**CPU版本PaddlePaddle**的用户：

			For Python2: cmake .. -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
			For Python3: cmake .. -DPY_VERSION=3.5 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
			-DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release

		> 如果遇到`Could NOT find PROTOBUF (missing:  PROTOBUF_LIBRARY PROTOBUF_INCLUDE_DIR)`可以重新执行一次cmake指令。
		> 请注意PY_VERSION参数更换为您需要的python版本


	* 对于需要编译**GPU版本PaddlePaddle**的用户：(*仅支持CentOS7（CUDA10/CUDA9）*)

		1. 请确保您已经正确安装nccl2，或者按照以下指令安装nccl2（这里提供的是ubuntu 16.04，CUDA9，cuDNN7下nccl2的安装指令），更多版本的安装信息请参考NVIDIA[官方网站](https://developer.nvidia.com/nccl/nccl-download):


			i. `wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm`


			ii.  `rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm`


			iii. `yum install -y libnccl-2.3.7-2+cuda9.0 libnccl-devel-2.3.7-2+cuda9.0 libnccl-static-2.3.7-2+cuda9.0`

		2. 如果您已经正确安装了`nccl2`，就可以开始cmake了：(*For Python3: 请给PY_VERSION参数配置正确的python版本*)

				For Python2: cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
				For Python3: cmake .. -DPYTHON_EXECUTABLE:FILEPATH=[您可执行的Python3的路径] -DPYTHON_INCLUDE_DIR:PATH=[之前的PYTHON_INCLUDE_DIRS] -DPYTHON_LIBRARY:FILEPATH=[之前的PYTHON_LIBRARY] -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release

	注意：以上涉及Python3的命令，用Python3.5来举例，如您的Python版本为3.6/3.7，请将上述命令中的Python3.5改成Python3.6/Python3.7




11. 使用以下命令来编译：

	`make -j$(nproc)`

12. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

13. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install -U（whl包的名字）`或`pip3 install -U（whl包的名字）`

恭喜，至此您已完成PaddlePaddle的编译安装

## ***验证安装***
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle.fluid as fluid` ，再输入
 `fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。

## ***如何卸载***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`

使用Docker安装PaddlePaddle的用户，请进入包含PaddlePaddle的容器中使用上述命令，注意使用对应版本的pip
