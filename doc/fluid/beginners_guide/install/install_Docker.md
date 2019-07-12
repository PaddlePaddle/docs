# **使用Docker安装**

[Docker](https://docs.docker.com/install/)是一个开源的应用容器引擎。使用Docker，既可以将PaddlePaddle的安装&使用与系统环境隔离，也可以与主机共享GPU、网络等资源

## 环境准备

- 目前支持的系统类型，请见[安装说明](./index_cn.html)，请注意目前暂不支持在CentOS 6使用Docker

- 在本地主机上[安装Docker](https://hub.docker.com/search/?type=edition&offering=community)

- 如需在Linux开启GPU支持，请[安装nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## 安装步骤

1. 拉取PaddlePaddle镜像

	* CPU版的PaddlePaddle： `docker pull hub.baidubce.com/paddlepaddle/paddle:[版本号]`

	* GPU版的PaddlePaddle： `docker pull hub.baidubce.com/paddlepaddle/paddle:[版本号]-gpu-cuda9.0-cudnn7`

    如果您的机器不在中国大陆地区，可以直接从DockerHub拉取镜像：

    * CPU版的PaddlePaddle： `docker pull paddlepaddle/paddle:[版本号]`

    * GPU版的PaddlePaddle： `docker pull paddlepaddle/paddle:[版本号]-gpu-cuda9.0-cudnn7`

    在`:`后请您填写PaddlePaddle版本号，例如当前版本，更多请见[镜像简介](#dockers)，上例中，`cuda9.0-cudnn7` 也仅作示意用，您可以访问[DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/)获取与您机器适配的镜像。

2. 构建、进入Docker容器

	* 使用CPU版本的PaddlePaddle：



		`docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash`
	
		> 上述命令中，各项参数分别指： --name [Name of container] 设定Docker的名称；-it 参数说明容器已和本机交互式运行； -v $PWD:/paddle 指定将当前路径（Linux中PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录； `<imagename>` 指定需要使用的image名称，您可以通过`docker images`命令查看；/bin/bash是在Docker中要执行的命令

	* 使用GPU版本的PaddlePaddle：



		`nvidia-docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash`
	
		> 上述命令中，各项参数分别指： --name [Name of container] 设定Docker的名称；-it 参数说明容器已和本机交互式运行； -v $PWD:/paddle 指定将当前路径（Linux中PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录； `<imagename>` 指定需要使用的image名称，您可以通过`docker images`命令查看；/bin/bash是在Docker中要执行的命令

至此，您已经成功使用Docker安装PaddlePaddle，更多Docker使用请参见[Docker官方文档](https://docs.docker.com)

<a name="dockers"></a>
</br></br>
### **镜像简介**
<p align="center">
<table>
	<thead>
	<tr>
		<th> 镜像源 </th>
		<th> 镜像说明 </th>
	</tr>
	</thead>
	<tbody>
		<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:[Version] </td>
		<td> 安装了指定版本PaddlePaddle </td>
	</tr>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest </td>
		<td> 安装了开发版PaddlePaddle。注意：此版本可能包含尚未发布的特性和不稳定的功能，因此不推荐常规用户或在生产环境中使用。 </td>
	</tr>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest-gpu </td>
		<td> 安装了开发版PaddlePaddle（支持GPU）。注意：此版本可能包含尚未发布的特性和不稳定的功能，因此不推荐常规用户或在生产环境中使用。 </td>
	</tr>
		<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest-dev </td>
		<td> 安装了PaddlePaddle最新的开发环境 </td>
	</tr>
   </tbody>
</table>
</p>

您可以在 [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) 中找到PaddlePaddle的各个发行的版本的docker镜像。

### 注意事项

* 镜像中Python版本为2.7
* PaddlePaddle Docker镜像为了减小体积，默认没有安装`vim`，您可以在容器中执行 `apt-get install -y vim` 安装后，在容器中编辑代码

### 补充说明

* 当您需要第二次进入Docker容器中，使用如下命令：
```
	#启动之前创建的容器
	docker start [Name of container]

	#进入启动的容器
	docker attach [Name of container]
```
* 如您是Docker新手，您可以参考互联网上的资料学习，例如[Docker教程](http://www.runoob.com/docker/docker-hello-world.html)

## 如何卸载

请您进入Docker容器后，执行如下命令

* ***CPU版本的PaddlePaddle***: `pip uninstall paddlepaddle`

* ***GPU版本的PaddlePaddle***: `pip uninstall paddlepaddle-gpu`

或通过`docker rm [Name of container]`来直接删除Docker容器

