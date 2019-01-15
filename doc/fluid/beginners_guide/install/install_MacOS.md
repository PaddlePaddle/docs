***

# **MacOS下安装**

本说明将介绍如何在*64位台式机或笔记本电脑*以及MacOS系统下安装PaddlePaddle，我们支持的MacOS系统需满足以下要求。

请注意：在其他系统上的尝试可能会导致安装失败。

* MacOS 10.11/10.12/10.13/10.14

## 确定要安装的版本

* 仅支持CPU的PaddlePaddle。



## 选择如何安装
在MacOS的系统下我们提供3种安装方式：

* pip安装（不支持GPU版本）(python3下不支持分布式）
* Docker安装（不支持GPU版本）(镜像中python的版本为2.7)
* Docker源码编译安装（不支持GPU版本）(镜像中的python版本为2.7，3.5，3.6，3.7)


**使用pip安装**（最便捷的安装方式），我们为您提供pip安装方法，但它更依赖您的本机环境，可能会出现和您本机环境相关的一些问题。


**使用Docker进行安装**（最保险的安装方式），因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。



<br/><br/>
### ***使用pip安装***

由于在MacOS中的Python情况差别较大我们暂不提供快速安装的命令，请您按照以下步骤进行安装

首先，**检查您的计算机和操作系统**是否符合我们支持的编译标准： `uname -m` 并且在`关于本机`中查看系统版本。

其次，您的计算机需要满足以下要求：

> **请不要使用MacOS中自带python**，对于**Python2**，建议您使用[Homebrew](https://brew.sh)或[Python.org](https://www.python.org/ftp/python/2.7.15/python-2.7.15-macosx10.9.pkg)提供的python2.7.15；对于**Python3**，请使用[Python.org](https://www.python.org/downloads/mac-osx/)提供的python3.5.x、python3.6.x或python3.7.x。

		For python2: brew install python@2 或 使用Python官方下载的python2.7.15
		For python3: 使用Python官方下载的python3.5.x、python3.6.x或python3.7.x

*  Python2.7.x，Pip >= 9.0.1
*  Python3.5.x，Pip3 >= 9.0.1
*  Python3.6.x，Pip3 >= 9.0.1
*  Python3.7.x，Pip3 >= 9.0.1

	> 注： 您的MacOS上可能已经安装pip请使用pip -V来确认我们建议使用pip 9.0.1或更高版本来安装。

下面将说明如何安装PaddlePaddle：


1. 使用pip install来安装PaddlePaddle：

	* 对于需要**CPU版本PaddlePaddle**的用户：`pip install paddlepaddle` 或 `pip3 install paddlepaddle`

	* 对于有**其他要求**的用户：`pip install paddlepaddle==[版本号]`  或 `pip3 install paddlepaddle==[版本号]`

	> `版本号`参见[最新Release安装包列表](./Tables.html/#ciwhls-release)或者您如果需要获取并安装**最新的PaddlePaddle开发分支**，可以从[CI系统](https://paddleci.ngrok.io/project.html?projectId=Manylinux1&tab=projectOverview) 中下载最新的whl安装包和c-api开发包并安装。如需登录，请点击“Log in as guest”。




现在您已经完成通过`pip install` 来安装的PaddlePaddle的过程。




<br/><br/>
### ***使用Docker安装***

<!-- 我们更加推荐**使用Docker进行安装**，因为我们在把工具和配置都安装在一个 Docker image 里，这样如果遇到问题，其他人可以复现问题以便帮助。另外，对于习惯使用Windows和MacOS的开发者来说，使用Docker就不用配置交叉编译环境了。需要强调的是：Docker 不会虚拟任何硬件，Docker container 里运行的编译工具实际上都是在本机的 CPU 和操作系统上直接运行的，性能和把编译工具安装在本机运行一样。-->

为了更好的使用Docker并避免发生问题，我们推荐使用**最高版本的Docker**，关于**安装和使用Docker**的细节请参阅Docker[官方文档](https://docs.docker.com/install/)。

> 请注意，在MacOS系统下登陆docker需要使用您的dockerID进行登录，否则将出现`Authenticate Failed`错误。

如果已经**正确安装Docker**，即可以开始**使用Docker安装PaddlePaddle**

1. 使用以下指令拉取我们为您预安装好PaddlePaddle的镜像：

	* 对于需要**CPU版本的PaddlePaddle**的用户请使用以下指令拉取我们为您预安装好*PaddlePaddle For CPU*的镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:1.2`

	* 您也可以通过以下指令拉取任意的我们提供的Docker镜像：

		`docker pull hub.baidubce.com/paddlepaddle/paddle:[tag]`

		> （请把[tag]替换为[镜像表](./Tables.html/#dockers)中的内容）

2. 使用以下指令用已经拉取的镜像构建并进入Docker容器：

	`docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash`

	> 上述命令中，--name [Name of container] 设定Docker的名称；-it 参数说明容器已和本机交互式运行； -v $PWD:/paddle 指定将当前路径（Linux中PWD变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)）挂载到容器内部的 /paddle 目录； `<imagename>` 指定需要使用的image名称，如果您需要使用我们的镜像请使用`hub.baidubce.com/paddlepaddle/paddle:[tag]` 注：tag的意义同第二步；/bin/bash是在Docker中要执行的命令。

3. （可选：当您需要第二次进入Docker容器中）使用如下命令使用PaddlePaddle：

	`docker start [Name of container]`

	> 启动之前创建的容器。

	`docker attach [Name of container]`

	> 进入启动的容器。


至此您已经成功使用Docker安装PaddlePaddle，您只需要进入Docker容器后运行PaddlePaddle即可，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)。

> 注：PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码。

<!--TODO: When we support pip install mode on MacOS, we can write on this part -->



<br/><br/>
## ***验证安装***
安装完成后您可以使用：`python` 或 `python3` 进入python解释器，然后使用`import paddle.fluid` 验证是否安装成功。

<br/><br/>
## ***如何卸载***
请使用以下命令卸载PaddlePaddle（使用docker安装PaddlePaddle的用户请进入包含PaddlePaddle的容器中使用以下命令，请使用相应版本的pip）：

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle` 或 `pip3 uninstall paddlepaddle`

