# Install on Docker

[Docker](https://docs.docker.com/install/) is an open-source application container engine, with which you can seperate the installation and use of PaddlePaddle from the system environment, and share GPU and network resources with the localhost.

## Environment Preparations

- For system types supporting now, please refer to [Installation Instructions](./index_en.html). Using Docker on CentOS 6 is not supported.

- [Install Docker](https://hub.docker.com/search/?type=edition&offering=community) on the localhost.

- If you need to start GPU supporting on Linux, please [install nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

## Installation Steps

1. Pull PaddlePaddle image

	* CPU version PaddlePaddle： `docker pull hub.baidubce.com/paddlepaddle/paddle:[version number]`

	* GPU version PaddlePaddle： `docker pull hub.baidubce.com/paddlepaddle/paddle:[version number]-gpu-cuda9.0-cudnn7`

    Please fill in the version number of PaddlePaddle such as 1.2 after `:`. For more details, please refer to [Image Instructions](#dockers).

2. Construct and enter Docker container

	`docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash`

	> Parameters in the commands above mean: --name [Name of container]  set the name ofDocker；-it  the parameter means the container has been working interactively with the localhost； -v $PWD:/paddle  appoints to pull the present path(which will unfold to absolute path by PWD variable on Linux) to /paddle catalogue inside the container； `<imagename>` appoints the name of the image needed, and you can check by `docker images` command；/bin/bash  is the command to execute in Docker

Until now, you have installed PaddlePaddle with Docker successfully, for more usage details please refer to [Docker Official Documents](https://docs.docker.com)

<a name="dockers"></a>
</br></br>
### **Image Instructions**
<p align="center">
<table>
	<thead>
	<tr>
		<th> version name </th>
		<th> version instructions </th>
	</tr>
	</thead>
	<tbody>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest </td>
		<td> newest image having installed CPU version PaddlePaddle </td>
	</tr>
		<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:[Version] </td>
		<td> old version image having installed PaddlePaddleversion </td>
	</tr>
	<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest-gpu </td>
		<td> newest image having installed GPU version PaddlePaddle </td>
	</tr>
		<tr>
		<td> hub.baidubce.com/paddlepaddle/paddle:latest-dev </td>
		<td> newest development environment of PaddlePaddle </td>
	</tr>
   </tbody>
</table>
</p>

You can find docker images of all versions of PaddlePaddle in [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/).

### Anouncements

* Python version is 2.7 in the image
* PaddlePaddle Docker doesn't install `vim` by default in order to decrease the volume, and you can execute `apt-get install -y vim` command in the container for compling codes.

### Supplementary Instructions

* When you need to enter Docker container the second time, please use the following commands:
```
	#start the previously constructed container
	docker start [Name of container]

	#enter the started container
	docker attach [Name of container]
```
* If you are new with Docker, please refer ro information on the Internet, such as [Docker Tutorial](http://www.runoob.com/docker/docker-hello-world.html)

## How to Unload

After entering Docker, execute the following commands:

* ***CPU version PaddlePaddle***: `pip uninstall paddlepaddle`

* ***GPU version PaddlePaddle***: `pip uninstall paddlepaddle-gpu`

or use `docker rm [Name of container]` to delete Docker container directly.

