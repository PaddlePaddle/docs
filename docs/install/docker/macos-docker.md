# **macOS 下的 Docker 安装**

[Docker](https://docs.docker.com/install/)是一个开源的应用容器引擎。使用 Docker，既可以将 PaddlePaddle 的安装&使用与系统环境隔离，也可以与主机共享 GPU、网络等资源。
以下 Docker 安装与使用流程中，docker 里已经安装好了特定版本的 PaddlePaddle。

## 环境准备

- macOS 版本 10.x/11.x (64 bit) (不支持 GPU 版本)

- 在本地主机上[安装 Docker](https://docs.docker.com/engine/install/)

- 镜像中 Python 版本为 3.10

## 安装步骤

### 1. 拉取 PaddlePaddle 镜像

对于国内用户，因为网络问题下载 docker 比较慢时，可使用百度提供的镜像：

* CPU 版的 PaddlePaddle：
    ```
    docker pull registry.baidubce.com/paddlepaddle/paddle:3.0.0b2
    ```

* CPU 版的 PaddlePaddle，且镜像中预装好了 jupyter：
    ```
    docker pull registry.baidubce.com/paddlepaddle/paddle:3.0.0b2-jupyter
    ```

如果您的机器不在中国大陆地区，可以直接从 DockerHub 拉取镜像：

* CPU 版的 PaddlePaddle：
    ```
    docker pull paddlepaddle/paddle:3.0.0b2
    ```

* CPU 版的 PaddlePaddle，且镜像中预装好了 jupyter：
    ```
    docker pull paddlepaddle/paddle:3.0.0b2-jupyter
    ```

您还可以访问[DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/)获取更多镜像。

### 2. 构建并进入 docker 容器

* 使用 CPU 版本的 PaddlePaddle：



    ```
    docker run --name paddle_docker -it -v $PWD:/paddle registry.baidubce.com/paddlepaddle/paddle:3.0.0b2 /bin/bash
    ```

    - `--name paddle_docker`：设定 Docker 的名称，`paddle_docker` 是自己设置的名称；


    - `-it`：参数说明容器已和本机交互式运行；


    - `-v $PWD:/paddle`：指定将当前路径（PWD 变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录；

    - `registry.baidubce.com/paddlepaddle/paddle:3.0.0b2`：指定需要使用的 image 名称，您可以通过`docker images`命令查看；/bin/bash 是在 Docker 中要执行的命令

* 使用 CPU 版本的 PaddlePaddle，且镜像中预装好了 jupyter：

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
    docker run -p 80:80 --rm --env USER_PASSWD="password you set" -v $PWD:/home/paddle registry.baidubce.com/paddlepaddle/paddle:3.0.0b2-jupyter
    ```

    - `--rm`：关闭容器后删除容器；


    - `--env USER_PASSWD="password you set"`：为 jupyter 设置登录密码，`password you set` 是自己设置的密码；


    - `-v $PWD:/home/paddle`：指定将当前路径（PWD 变量会展开为当前路径的绝对路径）挂载到容器内部的 /home/paddle 目录；

    - `registry.baidubce.com/paddlepaddle/paddle:3.0.0b2-jupyter`：指定需要使用的 image 名称，您可以通过`docker images`命令查看




至此，您已经成功使用 Docker 安装 PaddlePaddle，更多 Docker 使用请参见[Docker 官方文档](https://docs.docker.com)

<a name="dockers"></a>
</br></br>
## **镜像简介**
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
        <td> registry.baidubce.com/paddlepaddle/paddle:3.0.0b2 </td>
        <td> 安装了 3.0.0b2 版本 paddle 的 CPU 镜像 </td>
    </tr>
    <tr>
        <td> registry.baidubce.com/paddlepaddle/paddle:3.0.0b2-jupyter </td>
        <td> 安装了 3.0.0b2 版本 paddle 的 CPU 镜像，且镜像中预装好了 jupyter，启动 docker 即运行 jupyter 服务 </td>
    </tr>
   </tbody>
</table>
</p>

您可以在 [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) 中找到 PaddlePaddle 的各个发行的版本的 docker 镜像。


## 补充说明

* 当您需要第二次进入 Docker 容器中，使用如下命令：

    启动之前创建的容器
    ```
    docker start <Name of container>
    ```

    进入启动的容器
    ```
    docker attach <Name of container>
    ```

* 如您是 Docker 新手，您可以参考互联网上的资料学习，例如[Docker 教程](http://www.runoob.com/docker/docker-hello-world.html)

## 如何卸载

请您进入 Docker 容器后，执行如下命令

* **CPU 版本的 PaddlePaddle**:
    ```
    pip uninstall paddlepaddle
    ```

或通过`docker rm <Name of container>`来直接删除 Docker 容器
