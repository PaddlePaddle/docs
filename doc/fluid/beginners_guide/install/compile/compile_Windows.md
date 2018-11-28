***
# **Windows下从源码编译**

本说明将介绍如何在*64位台式机或笔记本电脑*以及Windows 10系统下编译PaddlePaddle，我们支持的Windows系统需满足以下要求：

* Windows 10
* Visual Stuido 2015 Update3

## 确定要编译的版本
* **仅支持CPU的PaddlePaddle**。

<!--* 支持GPU的PaddlePaddle，为了使得PaddlePaddle程序运行的更加迅速，我们通常使用GPU对PaddlePaddle程序进行加速，但安装GPU版本的PaddlePaddle需要先拥有满足以下条件的NVIDIA? GPU（具体安装流程和配置请务必参见NVIDIA官方文档：[For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)）
	* *Cuda 工具包9.0配合cuDNN v7*
	* *Cuda 工具包8.0配合cuDNN v7*
	* *GPU运算能力超过1.0的硬件设备*-->

## 选择如何编译
我们在Windows的系统下提供1种编译方式：

* 直接本机源码编译

由于在本机上的情况更加复杂，因此我们只支持特定的系统。


<a name="ct_source"></a>

### ***本机编译***

**请严格按照以下指令顺序执行**

1. 检查您的计算机和操作系统是否符合我们支持的编译标准

    * Windows 10
    
    * Visual Stuido 2015 Update3

2. 安装必要的工具 cmake，git 以及 python ：

    > cmake 需要3.0 及以上版本, 可以在官网进行下载，并添加到环境变量中。 [下载地址](https://cmake.org/download/) **
    
    > git可以在官网进行下载，并添加到环境变量中。 [下载地址](https://gitforwindows.org/) **
    
    > python 需要2.7 及以上版本, 同时确保 `numpy, protobuf, wheel` 等模块得到安装 [下载地址](https://www.python.org/download/releases/2.7/)**
    
        * 安装 numpy 包可以通过命令 `pip install numpy` 或 `pip3 install numpy`
        
        * 安装 protobuf 包可以通过命令 `pip install protobuf` 或 `pip3 install protobuf`
        
        * 安装 wheel 包可以通过命令 `pip install wheel` 或 `pip3 install wheel`

3. 将PaddlePaddle的源码clone在当下目录下的Paddle的文件夹中，并进入Padde目录下：

	- `git clone https://github.com/PaddlePaddle/Paddle.git`
	- `cd Paddle`

4. 切换到较稳定release分支下进行编译：

	- `git checkout release/1.2.0`

5. 并且请创建并进入一个叫build的目录下：

	- `mkdir build`
	- `cd build`

6. 执行cmake：

	>具体编译选项含义请参见[编译选项表](../Tables.html/#Compile)<!--TODO：Link 安装选项表到这里-->


	*  对于需要编译**CPU版本PaddlePaddle**的用户：

			For Python2: cmake .. -G "Visual Studio 14 2015 Win64" -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
			For Python3: cmake .. -G "Visual Studio 14 2015 Win64" -DPY_VERSION=3.5 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release

		> 如果遇到`Could NOT find PROTOBUF (missing:  PROTOBUF_LIBRARY PROTOBUF_INCLUDE_DIR)`可以重新执行一次cmake指令

7. 部分第三方依赖包（openblas，snappystream）目前需要用户自己提供预编译版本，也可以到 `https://github.com/wopeizl/Paddle_deps` 下载预编译好的文件， 将整个 `third_party` 文件夹放到 `build` 目录下.

8. 使用Blend for Visual Studio 2015 打开 `paddle.sln` 文件，选择平台为 `x64`，配置为 `Release`，开始编译

9. 编译成功后进入 `\paddle\build\python\dist` 目录下找到生成的 `.whl` 包：
  
	`cd \paddle\build\python\dist`

10. 在当前机器或目标机器安装编译好的 `.whl` 包：

	`pip install （whl包的名字）` 或 `pip3 install （whl包的名字）`

恭喜您，现在您已经完成使本机编译PaddlePaddle的过程了。


## ***验证安装***
安装完成后您可以使用：`python` 进入Python解释器，然后使用 `import paddle.fluid` 验证是否安装成功。

## ***如何卸载***
请使用以下命令卸载PaddlePaddle：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`
