***

# **Install under Ubuntu**

This instruction describes how to install PaddlePaddle on a *64-bit desktop or laptop and Ubuntu system. The Ubuntu systems we support must meet the following requirements:

Please note: Attempts on other systems may cause the installation to fail. Please ensure that your environment meets the above conditions. The installation we provide by default requires your computer processor to support the AVX instruction set. Otherwise, please select the version of `no_avx` in the [latest Release installation package list](./Tables.html/#ciwhls-release).

Under Ubuntu, you can use `cat /proc/cpuinfo | grep avx` to check if your processor supports the AVX instruction set.

* Ubuntu 14.04 /16.04 /18.04

## Determine which version to install

* Only PaddlePaddle for CPU is supported. If your computer does not have an NVIDIA® GPU, you can only install this version. If your computer has a GPU, it is also recommended that you install the CPU version of PaddlePaddle first to check if your local environment is suitable.

* Support for GPU PaddlePaddle. In order to make the PaddlePaddle program run more quickly, we accelerate the PaddlePaddle program through the GPU, but the GPU version of the PaddlePaddle needs to have the NVIDIA? GPU that meets the following conditions (see the NVIDIA official documentation for the specific installation process and configuration: [For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), [For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/))

	* *CUDA Toolkit 9.0 with cuDNN v7*
	* *CUDA Toolkit 8.0 with cuDNN v7*
	* *Hardware devices with GPU computing power exceeding 1.0*



## Choose how to install

Under the Ubuntu system, we offer 4 installation methods:

* Pip installation
* Docker installation (the version of python in the image is 2.7)
* Source code compilation and installation
* Docker source code compilation and installation (the python version in the image is 2.7, 3.5, 3.6, 3.7)


**With pip installation** (the easiest way to install), we offer you a pip installation method, but it depends more on your native environment and may have some issues related to your local environment.

**Use Docker for installation** (the safest way to install), because we are installing the tools and configuration in a Docker image so that if something goes wrong, others can reproduce the problem for help. In addition, for developers accustomed to using Windows and MacOS, there is no need to configure a cross-compilation environment using Docker. It should be emphasized that Docker does not virtualize any hardware. The compiler tools running in the Docker container are actually run directly on the native CPU and operating system. The performance is the same as installing the compiler on the machine.

Compile and install from [**source**](#ubt_source) and [**use Docker**](#ubt_docker). This is a process of compiling the PaddlePaddle source code into a binary file and then installing the binary file, compared to the binary that has been tested for you. The form of PaddlePaddle, manual compilation is more complicated, we will answer you in detail at the end of the description.

<br/><br/>
### ***Install using pip***


First, we use the following commands to **check if the environment of this machine** is suitable for installing PaddlePaddle:

    Uname -m && cat /etc/*release

>The above command will display the operating system and digits of the machine. Please make sure your computer is consistent with the 	requirements of this tutorial.

Second, your computer needs to meet any of the following requirements:

*	Python2.7.x (dev), Pip >= 9.0.1
*	Python3.5+.x (dev), Pip3 >= 9.0.1

>You may have installed pip on your Ubuntu. Please use pip -V or pip3 -V to confirm that we recommend using pip 9.0.1 or higher to install.

	Update apt source: `apt update`

Use the following command to install or upgrade Python and pip to the required version: (python3.6, python3.7 install pip and dev differ greatly in different Ubuntu versions, not described one by one)


	- For python2: `sudo apt install python-dev python-pip`

	- For python3.5: `sudo apt install python3.5-dev and curl https://bootstrap.pypa.io/get-pip.py -o - | python3.5 && easy_install pip`

	- For python3.6, python3.7: We default to python3.6 (3.7) and the corresponding versions of dev and pip3


>Even if you already have Python 2 or Python 3 in your environment, you need to install Python-dev or Python 3.5 (3.6, 3.7) -dev.

Now let's install PaddlePaddle:



1. Use pip install to install PaddlePaddle


	* For users who need **the CPU version PaddlePaddle**: `pip install paddlepaddle` or `pip3 install paddlepaddle`

	* For users who need **the GPU version PaddlePaddle**: `pip install paddlepaddle-gpu` or `pip3 install paddlepaddle-gpu`

	> 1. In order to prevent problems that cannot be found by nccl.h, please first install nccl2 according to the following command (here is ubuntu 16.04, CUDA9, ncDNN v7 nccl2 installation instructions), for more information about the installation information, please refer to [the NVIDIA official website](https://developer.nvidia.com/nccl/nccl-download):
		i. `Wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb`
		ii. `dpkg -i nvidia-machine- Learning-repo-ubuntu1604_1.0.0-1_amd64.deb` 
		iii. `sudo apt-get install -y libnccl2=2.2.13-1+cuda9.0 libnccl-dev=2.2.13-1+cuda9.0`

	> 2. If you do not specify the pypi package version number, we will by default provide you with a version of PaddlePaddle that supports Cuda 9/cuDNN v7.

	* For users with `Cannot uninstall 'six'`. problems, due to Python installation issues already in your system, please use `pip install paddlepaddle --ignore-installed six`(CPU) or `pip install paddlepaddle --ignore-installed Six` (GPU) solution.

	* For users with **other requirements**: `pip install paddlepaddle==[version number]` or `pip3 install paddlepaddle==[version number]`

	> For `the version number`, please refer to [the latest Release installation package list](./Tables.html/#whls). If you need to obtain and install **the latest PaddlePaddle development branch**, you can download and install the latest whl installation package and c-api development package from [the latest dev installation package list](./Tables.html/#ciwhls) or our [CI system](https://paddleci.ngrok.io/project.html?projectId=Manylinux1&tab=projectOverview). To log in, click on "Log in as guest".


Now you have completed the process of installing PaddlePaddle using `pip install`.


<br/><br/>
### ***Install using Docker***

In order to better use Docker and avoid problems, we recommend using **the highest version of Docker**. For details on **installing and using Docker**, please refer to [the official Docker documentation](https://docs.docker.com/install/).

> Please note that to install and use the PaddlePaddle version that supports GPU, you must first install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

If Docker is **properly installed**, you can start **installing PaddlePaddle using Docker**.

1. Use the following command to pull the image we pre-installed for PaddlePaddle:

	* For users who need **a CPU version of PaddlePaddle**, use the following command to pull the image we pre-installed for your *PaddlePaddle For CPU*:

		`Docker pull hub.baidubce.com/paddlepaddle/paddle: 1.2`

	* For users who need **a GPU version of PaddlePaddle**, use the following command to pull the image we pre-installed for your *PaddlePaddle For GPU*:

		`Docker pull hub.baidubce.com/paddlepaddle/paddle:1.2-gpu-cuda9.0-cudnn7`

	* You can also pull any of our Docker images by following the instructions below:

		`Docker pull hub.baidubce.com/paddlepaddle/paddle:[tag]`

		> (Please replace [tag] with the contents of [the mirror table](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/beginners_guide/install/Tables.html/#dockers))

2. Use the following command to build from the already pulled image and enter the Docker container:

	`Docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash`

	> In the above command, --name [Name of container] sets the name of the Docker; the `-it` parameter indicates that the container is running interactively with the machine; -v $PWD:/paddle specifies the current path (the PWD variable in Linux will expand to The absolute path of the current path is mounted to the /paddle directory inside the container; `<imagename>` specifies the name of the image to use, if you need to use our image please use `hub.baidubce.com/paddlepaddle/paddle:[tag]`. Note: The meaning of tag is the same as the second step; /bin/bash is the command to be executed in Docker.

3. (Optional: When you need to enter the Docker container a second time) Use PaddlePaddle with the following command:

	`Docker start [Name of container]`

	> The container created before starting.

	`Docker attach [Name of container]`

	> Enter the boot container.

Now that you have successfully installed PaddlePaddle using Docker, you only need to run PaddlePaddle after entering the Docker container. For more Docker usage, please refer to [the official Docker documentation](https://docs.docker.com/).

>Note: PaddlePaddle Docker image In order to reduce the size, `vim` is not installed by default. You can edit the code in the container after executing `apt-get install -y vim` in the container.


<br/><br/>
## ***Verify installation***

After the installation is complete, you can use `python` or `python3` to enter the python interpreter and then use `import paddle.fluid` to verify that the installation was successful.



<br/><br/>
## ***How to uninstall***

Please use the following command to uninstall PaddlePaddle (users who use PakerPaddle to install PaddlePaddle should use the following command in the container containing PaddlePaddle, please use the corresponding version of pip):

* ***CPU version of PaddlePaddle***: `pip uninstall paddlepaddle` or `pip3 uninstall paddlepaddle`

- ***GPU version of PaddlePaddle***: `pip uninstall paddlepaddle-gpu` or `pip3 uninstall paddlepaddle-gpu`
