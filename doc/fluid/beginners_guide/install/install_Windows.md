***

# **Windows下安装**

本说明将介绍如何在*64位台式机或笔记本电脑*以及Windows系统下安装PaddlePaddle，我们支持的Windows系统需满足以下要求。

请注意：在其他系统上的尝试可能会导致安装失败。

* *Windows 7/8 and Windows 10 专业版/企业版*

## 确定要安装的版本

* Windows下我们目前仅提供支持CPU的PaddlePaddle。


## 选择如何安装
在Windows系统下请使用我们为您提供的[一键安装包](http://paddle-windows.bj.bcebos.com/1.1/PaddlePaddle-windows-1.1.zip)进行安装
	
> 我们提供的一键安装包将基于Docker为您进行便捷的安装流程


我们之所以使用**基于Docker的安装方式**，是因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。        






<br/><br/>
## ***验证安装***
安装完成后您可以使用：`python` 或 `python3` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
## ***如何卸载***
请使用以下命令卸载PaddlePaddle（使用docker安装PaddlePaddle的用户请进入包含PaddlePaddle的容器中使用以下命令）：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`  



