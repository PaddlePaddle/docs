# **Install on macOS via Docker**

[Docker](https://docs.docker.com/install/) is an open source application container engine. Using docker, you can not only isolate the installation and use of paddlepaddle from the system environment, but also share GPU, network and other resources with the host

## Environment preparation

- macOS version 10.11/10.12/10.13/10.14 (64 bit)(not support GPU version)

- On the local host [Install Docker](https://hub.docker.com/search/?type=edition&offering=community)

## Installation steps

1. Pull PaddlePaddle image

    * CPU version of PaddlePaddle：
        ```
        docker pull registry.baidubce.com/paddlepaddle/paddle:[version number]
        ```

    * CPU version of PaddlePaddle, and the image is pre-installed with jupyter：
        ```
        docker pull registry.baidubce.com/paddlepaddle/paddle:[version number]-jupyter
        ```

    If your machine is not in mainland China, you can pull the image directly from DockerHub:

    * CPU version of PaddlePaddle：
        ```
        docker pull paddlepaddle/paddle:[version number]
        ```

    * CPU version of PaddlePaddle, and the image is pre-installed with jupyter：
        ```
        docker pull paddlepaddle/paddle:[version number]-jupyter

    After `:`please fill in the PaddlePaddle version number, you can see [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) to get the image that matches your machine.

2. Build and enter Docker container

    * Use CPU version of PaddlePaddle：



        ```
        docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash
        ```

        > --name [Name of container] set name of Docker;


        > -it The parameter indicates that the container has been operated interactively with the local machine;


        > -v $PWD:/paddle specifies to mount the current path of the host (PWD variable will expand to the absolute path of the current path) to the /paddle directory inside the container;

        > `<imagename>` Specify the name of the image to be used. You can view it through the 'docker images' command. /bin/Bash is the command to be executed in Docker

    * Use CPU version of PaddlePaddle：


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

        > --rm Delete the container after closing it;


        > --env USER_PASSWD=[password you set] Set the login password for jupyter, [password you set] is the password you set;


        > -v $PWD:/home/paddle Specifies to mount the current path (the PWD variable will be expanded to the absolute path of the current path) to the /home/paddle directory inside the container;

        > `<imagename>` Specify the name of the image to be used, you can view it through the `docker images` command



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
        <td> registry.baidubce.com/paddlepaddle/paddle:2.1.0 </td>
        <td> CPU image with 2.1.0 version of paddle installed </td>
    </tr>
    <tr>
        <td> registry.baidubce.com/paddlepaddle/paddle:2.1.0-jupyter </td>
        <td> CPU image of paddle version 2.1.0 is installed, and jupyter is pre-installed in the image. Start the docker to run the jupyter service </td>
    </tr>
   </tbody>
</table>
</p>

You can find the docker mirroring of the published versions of PaddlePaddle in [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/).


### Note

* Python version in the image is 3.7

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

Or delete the docker container directly through `docker rm [Name of container]`
