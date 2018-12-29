***
# **Install under MacOS**

This instruction will show you how to install PaddlePaddle on a *64-bit desktop or laptop* and MacOS system. The MacOS system we support must meet the following requirements.

Please note: Attempts on other systems may cause the installation to fail.

* MacOS 10.11/10.12/10.13/10.14

## Determine which version to install

* Only PaddlePaddle for CPU is supported.



## Choose how to install

Under the MacOS system we offer 3 installation methods:

* Pip installation (not supported for GPU version) (distributed is not supported under python3)
* Docker installation (the GPU version is not supported) (the version of python in the image is 2.7)
* Docker source compilation and installation (not supported for GPU version) (Python version 2.7, 3.5, 3.6, 3.7 in image)

**With pip installation** (the easiest way to install), we offer you a pip installation method, but it depends more on your local environment and may have some issues related to your local environment.

**Use Docker for installation** (the safest way to install), because we are installing the tools and configuration in a Docker image so that if something goes wrong, others can reproduce the problem for help. In addition, for developers accustomed to using Windows and MacOS, there is no need to configure a cross-compilation environment using Docker. It should be emphasized that Docker does not virtualize any hardware. The compiler tools running in the Docker container are actually run directly on the local CPU and operating system. The performance is the same as installing the compiler on the machine.


<br/><br/>
### ***Install using pip***

Due to the large difference in Python situation in MacOS, we do not provide quick installation commands. Please follow the steps below to install.

First, **check that your computer and operating system** meet our supported compilation standards or not: `uname -m` and view the system version `about this machine`.

Second, your computer needs to meet the following requirements:

> **Please do not use Python with MacOS**. For **Python 2**, we recommend Python2.7.15 provided by [Homebrew](https://brew.sh/) or [Python.org](https://www.python.org/ftp/python/2.7.15/python-2.7.15-macosx10.9.pkg). For Python3, please use python3.5.x, Python3.6.x or python3.7.x provided by [Python.org](https://www.python.org/downloads/mac-osx/).



		For python2: brew install python@2 or use Python officially downloaded python2.7.15
		For python3: Use Python officially downloaded python3.5.x, python3.6.x or python3.7.x


* Python2.7.x, Pip >= 9.0.1

* Python3.5.x, Pip3 >= 9.0.1

* Python3.6.x, Pip3 >= 9.0.1

* Python3.7.x, Pip3 >= 9.0.1

	> Note: You may have installed pip on your MacOS. Please use pip -V to confirm that we recommend using pip 9.0.1 or higher to install.

Here's how to install PaddlePaddle:

1. Use pip install to install PaddlePaddle:

	* For users who need **the CPU version PaddlePaddle**: `pip install paddlepaddle` or `pip3 install paddlepaddle`

	* For users with **other requirements**: `pip install paddlepaddle==[version number]` or `pip3 install paddlepaddle==[version number]`

	> For `the version number`, please refer to [the latest Release installation package list](./Tables.html/#ciwhls-release). If you need to obtain and install **the latest PaddlePaddle development branch**, you can download the latest whl installation package and c-api development package from [the CI system](https://paddleci.ngrok.io/project.html?projectId=Manylinux1&tab=projectOverview) and install it. To log in, click on "Log in as guest".



Now you have completed the process of installing PaddlePaddle via `pip install`.


<br/><br/>
### ***Install using Docker***

In order to better use Docker and avoid problems, we recommend using **the highest version of Docker**. For details on **installing and using Docker**, please refer to [the official Docker documentation](https://docs.docker.com/install/).

> Please note that logging in to docker on MacOS requires logging in with your dockerID, otherwise an `Authenticate Failed` error will occur.

If Docker is **properly installed**, you can start **using Docker to install PaddlePaddle**.

1. Use the following command to pull the image we pre-installed for PaddlePaddle:

	* For users who need **the CPU version of PaddlePaddle**, use the following command to pull the image we pre-installed for your *PaddlePaddle For CPU*:

		`Docker pull hub.baidubce.com/paddlepaddle/paddle: 1.2`

	* You can also pull any of our Docker images by following the instructions below:

		`Docker pull hub.baidubce.com/paddlepaddle/paddle:[tag]`

		> (Please replace [tag] with the contents of [the mirror table](./Tables.html/#dockers))

2. Use the following command to build from the already pulled image and enter the Docker container:

	`Docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash`

	> In the above command, --name [Name of container] sets the name of the Docker; the -it parameter indicates that the container is running interactively with the machine; -v $PWD:/paddle specifies the current path (the PWD variable in Linux will expand to [The absolute path](https://baike.baidu.com/item/绝对路径/481185) of the current path is mounted to the /paddle directory inside the container; `<imagename>` specifies the name of the image to use, if you need to use our image please use `hub.baidubce.com/paddlepaddle/paddle:[tag]`. Note: The meaning of tag is the same as the second step; /bin/bash is the command to be executed in Docker.

3. (Optional: When you need to enter the Docker container a second time) Use PaddlePaddle with the following command:

	`Docker start [Name of container]`

	> The container created before starting.

	`Docker attach [Name of container]`

	> Enter the boot container.

Now that you have successfully installed PaddlePaddle using Docker, you only need to run PaddlePaddle after entering the Docker container. For more Docker usage, please refer to [the official Docker documentation](https://docs.docker.com/).

> Note: PaddlePaddle Docker image In order to reduce the size, `vim` is not installed by default. You can edit the code in the container after executing `apt-get install -y vim` in the container.


<br/><br/>
## ***Verify installation***

After the installation is complete, you can use `python` or `python3` to enter the python interpreter and then use `import paddle.fluid` to verify that the installation was successful.

<br/><br/>
## ***How to uninstall***

Please use the following command to uninstall PaddlePaddle (users who use PakerPaddle to install PaddlePaddle should use the following command in the container containing PaddlePaddle, please use the corresponding version of pip):

* ***CPU version of PaddlePaddle***: `pip uninstall paddlepaddle` or `pip3 uninstall paddlepaddle`
