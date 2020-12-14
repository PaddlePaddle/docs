# **Docker Installation**

[Docker](https://docs.docker.com/install/) is an open source application container engine. Using docker, you can not only isolate the installation and use of paddlepaddle from the system environment, but also share GPU, network and other resources with the host

## Environment preparation

- Currently supported system types, please see [Installation instruction](./index_cn.html), please note that Docker is not currently supported in CentOS 6

- On the local host [Install Docker](https://hub.docker.com/search/?type=edition&offering=community)

- To enable GPU support on Linux, please [Install nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## Installation steps

1. Pull PaddlePaddle image

    * CPU version of PaddlePaddle：
        ```
        docker pull hub.baidubce.com/paddlepaddle/paddle:[version number]
        ```

    * GPU version of PaddlePaddle：
        ```
        docker pull hub.baidubce.com/paddlepaddle/paddle:[version number]-gpu-cuda9.0-cudnn7
        ```

    If your machine is not in mainland China, you can pull the image directly from DockerHub:

    * CPU version of PaddlePaddle：
        ```
        docker pull paddlepaddle/paddle:[version number]
        ```

    * GPU version of PaddlePaddle：
        ```
        docker pull paddlepaddle/paddle:[version number]-gpu-cuda9.0-cudnn7
        ```

    After `:', please fill in the PaddlePaddle version number, such as the current version. For more details, please refer to [image profile](#dockers), in the above example, `cuda9.0-cudnn7` is only for illustration. you can see [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) to get the image that matches your machine.

2. Build and enter Docker container

    * Use CPU version of PaddlePaddle：



        ```
        docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash
        ```

        > --name [Name of container] set name of Docker;


        > -it The parameter indicates that the container has been operated interactively with the local machine;


        > -v $PWD:/paddle specifies to mount the current path of the host (PWD variable in Linux will expand to the absolute path of the current path) to the /paddle directory inside the container;

        > `<imagename>` Specify the name of the image to be used. You can view it through the 'docker images' command. /bin/Bash is the command to be executed in Docker


    * Use GPU version of PaddlePaddle：



        ```
        nvidia-docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash
        ```

        > --name [Name of container] set name of Docker;


        > -it The parameter indicates that the container has been operated interactively with the local machine;


        > -v $PWD:/paddle specifies to mount the current path of the host (PWD variable in Linux will expand to the absolute path of the current path) to the /paddle directory inside the container;

        > `<imagename>` Specify the name of the image to be used. You can view it through the 'docker images' command. /bin/Bash is the command to be executed in Docker


Now you have successfully used Docker to install PaddlePaddle. For more information about using Docker, see[Docker official documents](https://docs.docker.com)

<a name="dockers"></a>
</br></br>
### **Introduction to mirror images**
<p align="center">
<table>
    <thead>
    <tr>
        <th> Mirror source </th>
        <th> Mirror description </th>
    </tr>
    </thead>
    <tbody>
        <tr>
        <td> hub.baidubce.com/paddlepaddle/paddle:[Version] </td>
        <td> Install pecified version of PaddlePaddle </td>
    </tr>
    <tr>
        <td> hub.baidubce.com/paddlepaddle/paddle:latest </td>
        <td> Install development version of PaddlePaddle。Note: This release may contain features and unstable features that have not yet been released, so it is not recommended for regular users or production environments. </td>
    </tr>
    <tr>
        <td> hub.baidubce.com/paddlepaddle/paddle:latest-gpu </td>
        <td> Install development of PaddlePaddle(support GPU). Note: This release may contain features and unstable features that have not yet been released, so it is not recommended for regular users or production environments. </td>
    </tr>
        <tr>
        <td> hub.baidubce.com/paddlepaddle/paddle:latest-dev </td>
        <td> Install the latest development environment of PaddlePaddle </td>
    </tr>
   </tbody>
</table>
</p>

You can find the docker mirroring of the published versions of PaddlePaddle in [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/).
### Note

* Python version in the image is 2.7
* In order to reduce the size, `vim` is not installed in PaddlePaddle Docker image by default. You can edit the code in the container after executing `apt-get install -y vim` in the container.

### 补充说明

* When you need to enter the docker container for the second time, use the following command:

    Container created before startup
    ```
    docker start [Name of container]
    ```

    Enter the starting container
    ```
    docker attach [Name of container]
    ```

* If you are a newcomer to Docker, you can refer to the materials on the Internet for learning, such as [Docker tutorial](http://www.runoob.com/docker/docker-hello-world.html)

## How to uninstall

After entering the Docker container, execute the following command:

* **CPU version of PaddlePaddle**:
    ```
    pip uninstall paddlepaddle
    ```

* **GPU version of PaddlePaddle**:
    ```
    pip uninstall paddlepaddle-gpu
    ```

Or delete the docker container directly through `docker rm [Name of container]`
