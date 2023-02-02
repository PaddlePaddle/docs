# **Linux 下的 Docker 安装**

[Docker](https://docs.docker.com/install/)是一个开源的应用容器引擎。使用 Docker，既可以将 PaddlePaddle 的安装&使用与系统环境隔离，也可以与主机共享 GPU、网络等资源

## 环境准备

- 目前支持的系统类型，请见[安装说明](../index_cn.html)，请注意目前暂不支持在 CentOS 6 使用 Docker

- 在本地主机上[安装 Docker](https://hub.docker.com/search/?type=edition&offering=community)

- 如需在 Linux 开启 GPU 支持, 需提前[安装 nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker) 和 [GPU 驱动](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
  * 请通过 docker -v 检查 Docker 版本。对于 19.03 之前的版本，您需要使用 nvidia-docker 和 nvidia-docker 命令；对于 19.03 及之后的版本，您将需要使用 nvidia-container-toolkit 软件包和 --gpus all 命令。这两个选项都记录在上面链接的网页上。

注 nvidia-container-toolkit 安装方法:
  * Ubuntu 系统可以参考以下命令
    * 添加存储库和密钥
    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    ```
    * 安装 nvidia-container-toolkit
    ```bash
    sudo apt update
    sudo apt install nvidia-container-toolkit
    ```
    * 重启 docker
    ```bash
    sudo systemctl restart docker
    ```
  * centos 系统可以参考以下命令
    * 添加存储库和密钥
    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
    ```
    * 安装 nvidia-container-toolkit
    ```bash
    sudo yun update
    sudo yum install -y nvidia-container-toolkit
    ```
    * 重启 docker
    ```bash
    sudo systemctl restart docker
    ```

## 安装步骤

1. 拉取 PaddlePaddle 镜像

    * CPU 版的 PaddlePaddle：
        ```
        docker pull registry.baidubce.com/paddlepaddle/paddle:[版本号]
        ```

    * CPU 版的 PaddlePaddle，且镜像中预装好了 jupyter：
        ```
        docker pull registry.baidubce.com/paddlepaddle/paddle:[版本号]-jupyter
        ```

    * GPU 版的 PaddlePaddle：
        ```
        docker pull registry.baidubce.com/paddlepaddle/paddle:[版本号]-gpu-cuda10.2-cudnn7
        ```

    如果您的机器不在中国大陆地区，可以直接从 DockerHub 拉取镜像：

    * CPU 版的 PaddlePaddle：
        ```
        docker pull paddlepaddle/paddle:[版本号]
        ```

    * CPU 版的 PaddlePaddle，且镜像中预装好了 jupyter：
        ```
        docker pull paddlepaddle/paddle:[版本号]-jupyter
        ```

    * GPU 版的 PaddlePaddle：
        ```
        docker pull paddlepaddle/paddle:[版本号]-gpu-cuda10.2-cudnn7
        ```

    在`:`后请您填写 PaddlePaddle 版本号，例如当前版本`2.1.0`，更多请见[镜像简介](#dockers)。

    上例中，`cuda10.2-cudnn7` 也仅作示意用，表示安装 GPU 版的镜像。如果您还想安装其他 cuda/cudnn 版本的镜像，可以将其替换成`cuda11.2-cudnn8`等。

    您可以访问[DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/)获取与您机器适配的镜像。

2. 构建、进入 Docker 容器

    * 使用 CPU 版本的 PaddlePaddle：

        ```
        docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash
        ```

        > --name [Name of container] 设定 Docker 的名称；


        > -it 参数说明容器已和本机交互式运行；


        > -v $PWD:/paddle 指定将当前路径（PWD 变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录；

        > `<imagename>` 指定需要使用的 image 名称，您可以通过`docker images`命令查看；/bin/bash 是在 Docker 中要执行的命令


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
        docker run -p 80:80 --rm --env USER_PASSWD=[password you set] -v $PWD:/home/paddle <imagename>
        ```

        > --rm 关闭容器后删除容器；


        > --env USER_PASSWD=[password you set] 为 jupyter 设置登录密码，[password you set] 是自己设置的密码；


        > -v $PWD:/home/paddle 指定将当前路径（PWD 变量会展开为当前路径的绝对路径）挂载到容器内部的 /home/paddle 目录；

        > `<imagename>` 指定需要使用的 image 名称，您可以通过`docker images`命令查看

    * 使用 GPU 版本的 PaddlePaddle：


        ```
        docker run --gpus all --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash
        ```

        > --gpus 指定 gpu 设备 ('"device=0,2"':代表使用 0 和 2 号 GPU; all: 代表使用所有 GPU), 可以参考[Docker 官方文档](https://docs.docker.com/engine/reference/commandline/run/#access-an-nvidia-gpu);

        > --name [Name of container] 设定 Docker 的名称；

        > -it 参数说明容器已和本机交互式运行；

        > -v $PWD:/paddle 指定将当前路径（PWD 变量会展开为当前路径的绝对路径）挂载到容器内部的 /paddle 目录；

        > `<imagename>` 指定需要使用的 image 名称，您可以通过`docker images`命令查看；/bin/bash 是在 Docker 中要执行的命令



至此，您已经成功使用 Docker 安装 PaddlePaddle，更多 Docker 使用请参见[Docker 官方文档](https://docs.docker.com)

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
        <td> registry.baidubce.com/paddlepaddle/paddle:2.1.0 </td>
        <td> 安装了 2.1.0 版本 paddle 的 CPU 镜像 </td>
    </tr>
    <tr>
        <td> registry.baidubce.com/paddlepaddle/paddle:2.1.0-jupyter </td>
        <td> 安装了 2.1.0 版本 paddle 的 CPU 镜像，且镜像中预装好了 jupyter，启动 docker 即运行 jupyter 服务 </td>
    </tr>
    <tr>
        <td> registry.baidubce.com/paddlepaddle/paddle:2.1.0-gpu-cuda11.2-cudnn8 </td>
        <td> 安装了 2.1.0 版本 paddle 的 GPU 镜像，cuda 版本为 11.2，cudnn 版本为 8.1 </td>
    </tr>
        <tr>
        <td> registry.baidubce.com/paddlepaddle/paddle:2.1.0-gpu-cuda10.2-cudnn7 </td>
        <td> 安装了 2.1.0 版本 paddle 的 GPU 镜像，cuda 版本为 10.2，cudnn 版本为 7 </td>
    </tr>
   </tbody>
</table>
</p>

您可以在 [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) 中找到 PaddlePaddle 的各个发行的版本的 docker 镜像。

### 注意事项

* 镜像中 Python 版本为 3.7

### 补充说明

* 当您需要第二次进入 Docker 容器中，使用如下命令：

    启动之前创建的容器
    ```
    docker start [Name of container]
    ```

    进入启动的容器
    ```
    docker attach [Name of container]
    ```

* 如您是 Docker 新手，您可以参考互联网上的资料学习，例如[Docker 教程](http://www.runoob.com/docker/docker-hello-world.html)

## 如何卸载

请您进入 Docker 容器后，执行如下命令

* **CPU 版本的 PaddlePaddle**:
    ```
    pip uninstall paddlepaddle
    ```

* **GPU 版本的 PaddlePaddle**:
    ```
    pip uninstall paddlepaddle-gpu
    ```

或通过`docker rm [Name of container]`来直接删除 Docker 容器
