***
# **MacOS下从源码编译**

本说明将介绍如何在*64位台式机或笔记本电脑*以及MacOS系统下编译PaddlePaddle，我们支持的MacOS系统需满足#以下要求：

* MacOS 10.12/10.13（这涉及到相关工具是否能被正常安装）

## 确定要编译的版本
* **仅支持CPU的PaddlePaddle**。

<!--* 支持GPU的PaddlePaddle，为了使得PaddlePaddle程序运行的更加迅速，我们通常使用GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA® GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *Cuda 工具包9.0配合cuDNN v7*
	* *Cuda 工具包8.0配合cuDNN v7*
	* *GPU运算能力超过1.0的硬件设备*-->

## 选择如何编译
在MacOS 10.12/10.13的系统下我们提供1种编译方式：


* Docker源码编译
* 直接本机源码编译





我们更加推荐**使用Docker进行编译**，因为我们在把工具和配置都安装在一个 Docker image 里。这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。        

同样对于那些出于各种原因不能够安装Docker的用户我们也提供了可以从**本机直接源码编译**的方法，但是由于在本机上的情况更加复杂，因此我们只支持特定的系统。        




       

<a name="mac_docker"></a>



<br/><br/>
### ***使用Docker编译***

为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**，关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)。

> 请注意，在MacOS系统下登陆docker需要使用您的dockerID进行登录，否则将出现`Authenticate Failed`错误。       


当您已经**正确安装Docker**后你就可以开始**使用Docker编译PaddlePaddle**啦：

1. 进入Mac的终端

2. 请选择您希望储存PaddlePaddle的路径，然后在该路径下使用以下命令将PaddlePaddle的源码从github克隆到本地当前目录下名为Paddle的文件夹中：

	`git clone https://github.com/PaddlePaddle/Paddle.git`

3. 进入Paddle目录下： `cd Paddle`

4. 利用我们提供的镜像（使用该命令您可以不必提前下载镜像）：

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`
	
	> --name paddle-test为您创建的Docker容器命名为paddle-test，-v $PWD:/paddle 将当前目录挂载到Docker容器中的/paddle目录下（Linux中PWD变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)），-it 与宿主机保持交互状态，`hub.baidubce.com/paddlepaddle/paddle:latest-dev` 使用名为`hub.baidubce.com/paddlepaddle/paddle:latest-dev`的镜像创建Docker容器，/bin/bash 进入容器后启动/bin/bash命令。

5. 进入Docker后进入paddle目录下：`cd paddle`

6. 切换到较稳定release分支下进行编译：

	`git checkout release/1.0.0`

7. 创建并进入/paddle/build路径下：

	`mkdir -p /paddle/build && cd /paddle/build`

8. 使用以下命令安装相关依赖：

		For Python2: pip install protobuf==3.1.0
		For Python3: pip install protobuf==3.1.0
		
	
	> 安装protobuf 3.1.0。

	`apt install patchelf`
	
	> 安装patchelf，PatchELF 是一个小而实用的程序，用于修改ELF可执行文件的动态链接器和RPATH。

9. 执行cmake：      
	
	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)<!--TODO： Link 编译选项表到这里-->


	*  对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF`
		
		> 我们目前不支持CentOS下GPU版本PaddlePaddle的编译            




10. 执行编译：

	`make -j$(nproc)`
	
	> 使用多核编译

11. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

12. 在当前机器或目标机器安装编译好的`.whl`包：

		For Python2: pip install （whl包的名字）
		For Python3: pip3 install （whl包的名字)
		

至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)。

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。

恭喜您，现在您已经完成使用Docker编译PaddlePaddle的过程。


<br/><br/>
### ***本机编译***


1. 检查您的计算机和操作系统是否符合我们支持的编译标准： `uname -m` 并且在`关于本机`中查看系统版本。

2. 安装python以及pip：    

	> MacOS中自带python但是并没有pip套件，我们强烈建议您使用[Homebrew](https://brew.sh)安装python, pip以及其他的依赖，这会大大降低您安装编译的难度。
	
		For python2: brew install python@2
		For python3: brew install python3
	
	> 请注意，当您的mac上安装有多个python时请保证您正在使用的python是您希望使用的python。

3. (Only For Python3)设置Python相关的环境变量：         
	
	- a. 首先使用 
			```find `dirname $(dirname             
			$(which python3))` -name "libpython3.*.dylib"```  
			找到Pythonlib的路径，然后（下面[python-lib-path]替换为找到文件路径）  
		
	- b. 设置PYTHON_LIBRARIES：`export PYTHON_LIBRARY=[python-lib-path]`
		
	- c. 其次使用找到PythonInclude的路径（通常是找到[python-lib-path]的上一级目录为同级目录的include,然后找到该目录下python3.x或者python2.x的路径），然后（下面[python-include-path]替换为找到路径）		
	- d. 设置PYTHON_INCLUDE_DIR: `export PYTHON_INCLUDE_DIRS=[python-include-path]`
		
	- e. 设置系统环境变量路径：`export PATH=[python-lib-path]:$PATH` （这里将[python-lib-path]的最后两级目录替换为/bin/) 
	
	

4. **执行编译前**请您确认您的环境中安装有[编译依赖表](../Tables.html/#third_party)中提到的相关依赖，否则我们强烈推荐使用`Homebrew`安装相关依赖。
	
	> MacOS下的依赖可以使用`pip install [依赖名]` 或 `brew install [依赖名]` 来安装
	
	- a. 这里特别说明一下**CMake**的安装：
		
		由于我们使用的是CMake3.4请根据以下步骤：
		
		1. 从CMake[官方网站](https://cmake.org/files/v3.4/cmake-3.4.3-Darwin-x86_64.dmg)下载CMake镜像并安装
		2. 在控制台输入`sudo "/Applications/CMake.app/Contents/bin/cmake-gui" –install`


5. 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下：

	- `git clone https://github.com/PaddlePaddle/Paddle.git`

	- `cd Paddle`

6. 切换到较稳定release分支下进行编译：

	`git checkout release/1.0.0`

7. 并且请创建并进入一个叫build的目录下：

	`mkdir build && cd build`

8. 执行cmake：
	
	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)<!--TODO：Link 安装选项表到这里-->


	*  对于需要编译**CPU版本PaddlePaddle**的用户：

			For Python2: cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF.
			For Python3: cmake .. -DPY_VERSION=3.5 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF
	

9. 使用以下命令来编译：

	`make -j4`

10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

11. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install （whl包的名字）`或`pip3 install （whl包的名字）`     
	
	> 如果您的电脑上安装有多个python环境以及pip请参见[FAQ](../Tables.html/#MACPRO)

恭喜您，现在您已经完成使本机编译PaddlePaddle的过程了。




<br/><br/>
## ***验证安装***
安装完成后您可以使用：`python` 进入Python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
## ***如何卸载***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`       
