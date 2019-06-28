# **MacOS下从源码编译**

## 环境准备

* *64位操作系统*
* *MacOS 10.12/10.13/10.14*
* *Python（64 bit） 2.7/3.5.1+/3.6/3.7*
* *pip或pip3（64 bit） >= 9.0.1*

## 选择CPU/GPU

* 目前仅支持在MacOS环境下编译安装CPU版本的PaddlePaddle

## 安装步骤
在MacOS系统下有2种编译方式：

* Docker源码编译
* 本机源码编译

<a name="mac_docker"></a>
### ***使用Docker编译***

[Docker](https://docs.docker.com/install/)是一个开源的应用容器引擎。使用Docker，既可以将PaddlePaddle的安装&使用与系统环境隔离，也可以与主机共享GPU、网络等资源

使用Docker编译PaddlePaddle，您需要：

- 在本地主机上[安装Docker](https://hub.docker.com/search/?type=edition&offering=community)

- 使用Docker ID登陆Docker，以避免出现`Authenticate Failed`错误

请您按照以下步骤安装：

1. 进入Mac的终端

2. 请选择您希望储存PaddlePaddle的路径，然后在该路径下使用以下命令将PaddlePaddle的源码从github克隆到本地当前目录下名为Paddle的文件夹中：

	`git clone https://github.com/PaddlePaddle/Paddle.git`

3. 进入Paddle目录下： `cd Paddle`

4. 创建并进入满足编译环境的Docker容器：

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`

	> --name paddle-test为您创建的Docker容器命名为paddle-test，-v $PWD:/paddle 将当前目录挂载到Docker容器中的/paddle目录下（Linux中PWD变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)），-it 与宿主机保持交互状态，`hub.baidubce.com/paddlepaddle/paddle:latest-dev` 使用名为`hub.baidubce.com/paddlepaddle/paddle:latest-dev`的镜像创建Docker容器，/bin/bash 进入容器后启动/bin/bash命令。

5. 进入Docker后进入paddle目录下：

	`cd paddle`

6. 切换到较稳定版本下进行编译：

	`git checkout [分支名]`

	例如：

	`git checkout release/1.2`

	注意：python3.6、python3.7版本从release/1.2分支开始支持

7. 创建并进入/paddle/build路径下：

	`mkdir -p /paddle/build && cd /paddle/build`

8. 使用以下命令安装相关依赖：

		For Python2: pip install protobuf==3.1.0
		For Python3: pip3.5 install protobuf==3.1.0

	注意：以上用Python3.5命令来举例，如您的Python版本为3.6/3.7，请将上述命令中的Python3.5改成Python3.6/Python3.7

	> 安装protobuf 3.1.0。

	`apt install patchelf`

	> 安装patchelf，PatchELF 是一个小而实用的程序，用于修改ELF可执行文件的动态链接器和RPATH。

9. 执行cmake：

	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)
	>请注意修改参数`-DPY_VERSION`为您希望编译使用的python版本,  例如`-DPY_VERSION=3.5`表示python版本为3.5.x

	*  对于需要编译**CPU版本PaddlePaddle**的用户：

		`cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DWITH_AVX=OFF -DCMAKE_BUILD_TYPE=Release`

		> 我们目前不支持MacOS下GPU版本PaddlePaddle的编译

10. 执行编译：

	`make -j$(nproc)`

	> 使用多核编译

11. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

12. 在当前机器或目标机器安装编译好的`.whl`包：

		For Python2: pip install -U（whl包的名字）
		For Python3: pip3.5 install -U（whl包的名字)

		注意：以上涉及Python3的命令，用Python3.5来举例，如您的Python版本为3.6/3.7，请将上述命令中的Python3.5改成Python3.6/Python3.7

恭喜，至此您已完成PaddlePaddle的编译安装。您只需要进入Docker容器后运行PaddlePaddle，即可开始使用。更多Docker使用请参见[Docker官方文档](https://docs.docker.com)

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 来安装

<a name="mac_source"></a>
<br/><br/>
### ***本机编译***

**请严格按照以下指令顺序执行**

1. 检查您的计算机和操作系统是否符合我们支持的编译标准： `uname -m` 并且在`关于本机`中查看系统版本。并提前安装[OpenCV](https://opencv.org/releases.html)

2. 安装Python以及pip：

	> **请不要使用MacOS中自带Python**，我们强烈建议您使用[Homebrew](https://brew.sh)安装python(对于**Python3**请使用python[官方下载](https://www.python.org/downloads/mac-osx/)python3.5.x、python3.6.x、python3.7.x), pip以及其他的依赖，这将会使您高效编译。

		For python2: brew install python@2
		For python3: 使用Python官网安装

	> 请注意，当您的mac上安装有多个python时请保证您正在使用的python是您希望使用的python。

3. (Only For Python2)设置Python相关的环境变量：

	- 请使用`find / -name libpython2.7.dylib`找到您当前使用python的`libpython2.7.dylib`路径，并使用`export LD_LIBRARY_PATH=[libpython2.7.dylib的路径] && export DYLD_LIBRARY_PATH=[libpython2.7.dylib所在的目录的上两级目录]`

4. (Only For Python3)设置Python相关的环境变量：

	- a. 首先使用
			```find `dirname $(dirname
			  $(which python3))` -name "libpython3.*.dylib"```
			找到Pythonlib的路径（弹出的第一个对应您需要使用的python的dylib路径），然后（下面[python-lib-path]替换为找到文件路径）

	- b. 设置PYTHON_LIBRARIES：`export PYTHON_LIBRARY=[python-lib-path]`

	- c. 其次使用找到PythonInclude的路径（通常是找到[python-lib-path]的上一级目录为同级目录的include,然后找到该目录下python3.x或者python2.x的路径），然后（下面[python-include-path]替换为找到路径）
	- d. 设置PYTHON_INCLUDE_DIR: `export PYTHON_INCLUDE_DIRS=[python-include-path]`

	- e. 设置系统环境变量路径：`export PATH=[python-bin-path]:$PATH` （这里[python-bin-path]为将[python-lib-path]的最后两级目录替换为/bin/后的目录)

	- f. 设置动态库链接： `export LD_LIBRARY_PATH=[python-ld-path]` 以及 `export DYLD_LIBRARY_PATH=[python-ld-path]` （这里[python-ld-path]为[python-bin-path]的上一级目录)

	- g. (可选）如果您是在MacOS 10.14上编译PaddlePaddle，请保证您已经安装了[对应版本](http://developer.apple.com/download)的Xcode。

5. **执行编译前**请您确认您的环境中安装有[编译依赖表](../Tables.html/#third_party)中提到的相关依赖，否则我们强烈推荐使用`Homebrew`安装相关依赖。

	> MacOS下如果您未自行修改或安装过“编译依赖表”中提到的依赖，则仅需要使用`pip`安装`numpy，protobuf，wheel`，使用`homebrew`安装`wget，swig`，另外安装`cmake`即可

	- a. 这里特别说明一下**CMake**的安装：

		由于我们使用的是CMake3.4请根据以下步骤：

		1. 从CMake[官方网站](https://cmake.org/files/v3.4/cmake-3.4.3-Darwin-x86_64.dmg)下载CMake镜像并安装
		2. 在控制台输入`sudo "/Applications/CMake.app/Contents/bin/cmake-gui" –install`

	- b. 如果您不想使用系统默认的blas而希望使用自己安装的OPENBLAS请参见[FAQ](../FAQ.html/#OPENBLAS)

6. 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下：

	- `git clone https://github.com/PaddlePaddle/Paddle.git`

	- `cd Paddle`

7. 切换到较稳定release分支下进行编译：

	`git checkout [分支名]`

	例如：

	`git checkout release/1.2`

	注意：python3.6、python3.7版本从release/1.2分支开始支持

8. 并且请创建并进入一个叫build的目录下：

	`mkdir build && cd build`

9. 执行cmake：

	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)

	*  对于需要编译**CPU版本PaddlePaddle**的用户：

			For Python2: cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF  -DCMAKE_BUILD_TYPE=Release
			For Python3: cmake .. -DPY_VERSION=3.5 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
			 -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF  -DCMAKE_BUILD_TYPE=Release

	>`-DPY_VERSION=3.5`请修改为安装环境的Python版本

10. 使用以下命令来编译：

	`make -j4`

11. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包： `cd /paddle/build/python/dist`

12. 在当前机器或目标机器安装编译好的`.whl`包：

	`pip install -U（whl包的名字）`或`pip3 install -U（whl包的名字）`

	> 如果您的电脑上安装有多个python环境以及pip请参见[FAQ](../Tables.html/#MACPRO)

恭喜，至此您已完成PaddlePaddle的编译安装

## ***验证安装***
安装完成后您可以使用 `python` 或 `python3` 进入python解释器，输入`import paddle.fluid as fluid` ，再输入
 `fluid.install_check.run_check()`

如果出现`Your Paddle Fluid is installed succesfully!`，说明您已成功安装。

## ***如何卸载***
请使用以下命令卸载PaddlePaddle

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`

使用Docker安装PaddlePaddle的用户，请进入包含PaddlePaddle的容器中使用上述命令，注意使用对应版本的pip
