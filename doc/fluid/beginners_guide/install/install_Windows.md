***

# **Windows下安装**

本说明将介绍如何在*64位台式机或笔记本电脑*以及Windows系统下安装PaddlePaddle，我们支持的Windows系统需满足以下要求。



请注意：在其他系统上的尝试可能会导致安装失败。 请确保您的环境满足以上条件，我们默认提供的安装同时需要您的计算机处理器支持AVX指令集，否则请选择[多版本whl包安装列表](Tables.html/#ciwhls) 中`no_avx`的版本:

Windows系统下可使用`cpu-z`这类软件来检测您的处理器是否支持AVX指令集

当前版本不支持NCCL，分布式，AVX，warpctc和MKL相关功能。

* *Windows 7/8 and Windows 10 专业版/企业版*

## 确定要安装的版本

* Windows下我们目前仅提供支持CPU的PaddlePaddle。

## 选择如何安装

### ***使用pip安装***

我们暂不提供快速安装的命令，请您按照以下步骤进行安装

* 首先，**检查您的计算机和操作系统**是否满足以下要求：
	
		For python2: 使用Python官方下载的python2.7.15
		For python3: 使用Python官方下载的python3.5.x, python3.6.x 或 python3.7.x

*  Python2.7.x ：pip >= 9.0.1
*  Python3.5.x, python3.6.x 或 python3.7.x ：pip3 >= 9.0.1
	    
下面将说明如何安装PaddlePaddle：

* 使用pip install来安装PaddlePaddle：
	
    ** paddlepaddle 的依赖包 `recordio` 有可能用 `pip` 的默认源无法安装，可以使用 `easy_install recordio` 来安装 **

	** 对于需要**CPU版本PaddlePaddle**的用户：`pip install paddlepaddle` 或 `pip3 install paddlepaddle` **

现在您已经完成通过`pip install` 来安装的PaddlePaddle的过程。

## ***验证安装***
安装完成后您可以使用：`python` 或 `python3` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

## ***如何卸载***
请使用以下命令卸载PaddlePaddle（使用docker安装PaddlePaddle的用户请进入包含PaddlePaddle的容器中使用以下命令）：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`  



