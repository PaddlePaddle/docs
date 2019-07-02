# **Windows下从源码编译**

## 环境准备

* *64位操作系统*
* *Windows 10 家庭版/专业版/企业版*
* *Python（64 bit） 2.7/3.5.1+/3.6/3.7*
* *pip或pip3（64 bit） >= 9.0.1*
* *Visual Studio 2015 Update3*

## 选择CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请编译CPU版的PaddlePaddle

* 如果您的计算机有NVIDIA® GPU，并且满足以下条件，推荐编译GPU版的PaddlePaddle
    * *CUDA 工具包8.0配合cuDNN v7*
    * *GPU运算能力超过1.0的硬件设备*

## 安装步骤

在Windows的系统下提供1种编译方式：

* 本机编译（暂不支持NCCL，分布式等相关功能）

<a name="win_source"></a>
### ***本机编译***

1. 安装必要的工具 cmake，git 以及 python ：

    > cmake 需要3.5 及以上版本, 可在官网[下载](https://cmake.org/download/)，并添加到环境变量中。

    > python 需要2.7 及以上版本, 可在官网[下载](https://www.python.org/download/releases/2.7/)。

    > 需要安装`numpy, protobuf, wheel` 。python2.7下, 请使用`pip`命令; 如果是python3.x, 请使用`pip3`命令。

        * 安装 numpy 包可以通过命令 `pip install numpy` 或 `pip3 install numpy`

        * 安装 protobuf 包可以通过命令 `pip install protobuf` 或 `pip3 install protobuf`

        * 安装 wheel 包可以通过命令 `pip install wheel` 或 `pip3 install wheel`

    > git可以在官网[下载](https://gitforwindows.org/)，并添加到环境变量中。

2. 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下：

	- `git clone https://github.com/PaddlePaddle/Paddle.git`
	- `cd Paddle`

3. 切换到较稳定release分支下进行编译：

	`git checkout [分支名]`

	例如：

	`git checkout release/1.2`

	注意：python3.6、python3.7版本从release/1.2分支开始支持

4. 创建名为build的目录并进入：

	- `mkdir build`
	- `cd build`

5. 执行cmake：

	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)

	*  编译**CPU版本PaddlePaddle**：

		For Python2: `cmake .. -G "Visual Studio 14 2015 Win64" -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS}
			 -DPYTHON_LIBRARY=${PYTHON_LIBRARY}
			 -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

		For Python3: `cmake .. -G "Visual Studio 14 2015 Win64" -DPY_VERSION=3.5 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS}
			 -DPYTHON_LIBRARY=${PYTHON_LIBRARY}
			 -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

	*  编译**GPU版本PaddlePaddle**：

		For Python2: `cmake .. -G "Visual Studio 14 2015 Win64" -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS}
			 -DPYTHON_LIBRARY=${PYTHON_LIBRARY}
			 -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
			 -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}`

		For Python3: `cmake .. -G "Visual Studio 14 2015 Win64" -DPY_VERSION=3.5 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS}
			 -DPYTHON_LIBRARY=${PYTHON_LIBRARY}
			 -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
			 -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}`

6. 部分第三方依赖包（openblas，snappystream）目前需要用户自己提供预编译版本，也可以到 `https://github.com/wopeizl/Paddle_deps` 下载预编译好的文件， 将整个 `third_party` 文件夹放到 `build` 目录下.

7. 使用Blend for Visual Studio 2015 打开 `paddle.sln` 文件，选择平台为 `x64`，配置为 `Release`，先编译third_party模块，然后编译其他模块

8. 编译成功后进入 `\paddle\build\python\dist` 目录下找到生成的 `.whl` 包：

	`cd \paddle\build\python\dist`

9. 在当前机器或目标机器安装编译好的 `.whl` 包：

	`pip install -U（whl包的名字）` 或 `pip3 install -U（whl包的名字）`

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
