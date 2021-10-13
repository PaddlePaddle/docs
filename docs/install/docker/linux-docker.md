# **Linux下的Docker安装**

[Docker](https://docs.docker.com/install/)是一个开源的应用容器引擎。使用Docker，既可以将PaddlePaddle的安装&使用与系统环境隔离，也可以与主机共享GPU、网络等资源。
以下Docker安装与使用流程中，docker里已经安装好了特定版本的PaddlePaddle。

## 环境准备

- 目前支持的系统类型，请见[安装说明](/documentation/docs/zh/install/index_cn.html)，请注意目前暂不支持在CentOS 6使用Docker

- 在本地主机上[安装Docker](https://docs.docker.com/engine/install/)

- 如需在Linux开启GPU支持，请[安装nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## 安装步骤

1. 拉取PaddlePaddle镜像

    对于国内用户，因为网络问题下载docker比较慢时，可使用百度提供的镜像：

    * CPU版的PaddlePaddle：
        ```
        docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.0rc0
        ```

    * CPU版的PaddlePaddle，且镜像中预装好了 jupyter：
        ```
        docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.0rc0-jupyter
        ```

    * GPU版的PaddlePaddle：
        ```
        docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.0rc0-gpu-cuda10.2-cudnn7
        ```
        ```
        docker pull registry.baidubce.com/paddlepaddle/paddle:2.2.0rc0-gpu-cuda11.2-cudnn8
        ```

    如果您的机器不在中国大陆地区，可以直接从DockerHub拉取镜像：

    * CPU版的PaddlePaddle：
        ```
        docker pull paddlepaddle/paddle:2.2.0rc0
        ```

    * CPU版的PaddlePaddle，且镜像中预装好了 jupyter：
        ```
        docker pull paddlepaddle/paddle:2.2.0rc0-jupyter
        ```

    * GPU版的PaddlePaddle：
        ```
        docker pull paddlepaddle/paddle:2.2.0rc0-gpu-cuda10.2-cudnn7
        ```
        ```
        docker pull paddlepaddle/paddle:2.2.0rc0-gpu-cuda11.2-cudnn8
        ```

    您还可以访问[DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/)获取更多镜像。

2. 构建、进入Docker容器

    * 使用CPU版本的PaddlePaddle：



        ```
        docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash
        ```

        > --name [Name of container] 设定Docker的名称；


        > -it 参数说明容器已和本机交互式运行；


        > -v $PWD:/paddle 指定将当前路径（PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录；

        > `<imagename>` 指定需要使用的image名称，您可以通过`docker images`命令查看；/bin/bash是在Docker中要执行的命令


    * 使用CPU版本的PaddlePaddle，且镜像中预装好了 jupyter：

        ```
        mkdir ./jupyter_docker
        ```
        ```
        chmod 777 ./jupyter_docker
        ```
        ```
        cd ./jupyter_docker
        ```
        ```
        docker run -p 80:80 --rm --env USER_PASSWD=[password you set] -v $PWD:/home/paddle <imagename>
        ```

        > --rm 关闭容器后删除容器；


        > --env USER_PASSWD=[password you set] 为 jupyter 设置登录密码，[password you set] 是自己设置的密码；


        > -v $PWD:/home/paddle 指定将当前路径（PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /home/paddle 目录；

        > `<imagename>` 指定需要使用的image名称，您可以通过`docker images`命令查看

    * 使用GPU版本的PaddlePaddle：



        ```
        nvidia-docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash
        ```

        > --name [Name of container] 设定Docker的名称；


        > -it 参数说明容器已和本机交互式运行；


        > -v $PWD:/paddle 指定将当前路径（PWD变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录；

        > `<imagename>` 指定需要使用的image名称，您可以通过`docker images`命令查看；/bin/bash是在Docker中要执行的命令



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
        <td> registry.baidubce.com/paddlepaddle/paddle:2.2.0rc0 </td>
        <td> 安装了2.2.0rc0版本paddle的CPU镜像 </td>
    </tr>
    <tr>
        <td> registry.baidubce.com/paddlepaddle/paddle:2.2.0rc0-jupyter </td>
        <td> 安装了2.2.0rc0版本paddle的CPU镜像，且镜像中预装好了jupyter，启动docker即运行jupyter服务 </td>
    </tr>
    <tr>
        <td> registry.baidubce.com/paddlepaddle/paddle:2.2.0rc0-gpu-cuda11.2-cudnn8 </td>
        <td> 安装了2.2.0rc0版本paddle的GPU镜像，cuda版本为11.2，cudnn版本为8.1 </td>
    </tr>
        <tr>
        <td> registry.baidubce.com/paddlepaddle/paddle:2.2.0rc0-gpu-cuda10.2-cudnn7 </td>
        <td> 安装了2.2.0rc0版本paddle的GPU镜像，cuda版本为10.2，cudnn版本为7 </td>
    </tr>
   </tbody>
</table>
</p>

您可以在 [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) 中找到PaddlePaddle的各个发行的版本的docker镜像。

### 注意事项

* 镜像中Python版本为3.7

### 补充说明

* 当您需要第二次进入Docker容器中，使用如下命令：

    启动之前创建的容器
    ```
    docker start [Name of container]
    ```

    进入启动的容器
    ```
    docker attach [Name of container]
    ```

* 如您是Docker新手，您可以参考互联网上的资料学习，例如[Docker教程](http://www.runoob.com/docker/docker-hello-world.html)

## 如何卸载

请您进入Docker容器后，执行如下命令

* **CPU版本的PaddlePaddle**:
    ```
    pip uninstall paddlepaddle
    ```

* **GPU版本的PaddlePaddle**:
    ```
    pip uninstall paddlepaddle-gpu
    ```

或通过`docker rm [Name of container]`来直接删除Docker容器
