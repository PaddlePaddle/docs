***

# **Install under CentOS**

This note will show you how to install PaddlePaddle on a *64-bit desktop or laptop* and CentOS. The CentOS system we support needs to meet the following requirements:

Please note: Attempts on other systems may cause the installation to fail. Please ensure that your environment meets the conditions. The installation we provide by default requires your computer processor to support the AVX instruction set. Otherwise, please select the version of `no_avx` in [the latest Release installation package list](./Tables.html/#ciwhls-release).

Under CentOS you can use `cat /proc/cpuinfo | grep avx` to check if your processor supports the AVX instruction set.

* CentOS 6 / 7

## Determine which version to install

* Only PaddlePaddle for CPU is supported. If your computer does not have an NVIDIA® GPU, you can only install this version. If your computer has a GPU, it is recommended that you install the CPU version of PaddlePaddle first to check if your local environment is suitable.

* PaddlePaddle with GPU support, in order to make the PaddlePaddle program run more quickly, we accelerate the PaddlePaddle program through the GPU, but the GPU version of PaddlePaddle needs to have the NVIDIA® GPU that meets the following conditions (see the NVIDIA official for the specific installation process and configuration). Documentation: [For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), [For cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/))

	* *CUDA Toolkit 9.0 with cuDNN v7*
	* *CUDA Toolkit 8.0 with cuDNN v7*
	* *Hardware devices with GPU compute capability exceeding 1.0*


## Choose an installation method

We offer 4 installation methods under the CentOS system:

* Pip installation
* Docker installation (the GPU version is not supported) (the version of python in the image is 2.7)
* Source code compilation and installation (all versions of CentOS 6 and GPU version of CentOS 7 are not supported)
* Docker source compilation and installation (not supported for GPU version) (Python version 2.7, 3.5, 3.6, 3.7 in image)

**With pip installation** (the easiest way to install), we offer you a pip installation method, but it depends more on your native environment and may have some issues related to your local environment.

**Use Docker for installation** (the safest way to install), because we are installing the tools and configuration in a Docker image so that if something goes wrong, others can reproduce the problem for help. In addition, for developers accustomed to using Windows and MacOS, there is no need to configure a cross-compilation environment using Docker. It should be emphasized that Docker does not virtualize any hardware. The compiler tools running in the Docker container are actually run directly on the native CPU and operating system. The performance is the same as installing the compiler on the machine.

Compile and install from [**source**](#ct_source) and [**use Docker**](#ct_docker). This is a process of compiling the PaddlePaddle source code into a binary file and then installing the binary file. Compared with the binary form of PaddlePaddle that has been successfully tested and compiled for you, this manual compilation is more complicated, and we will answer you in detail at the end of this tutorial.


<br/><br/>
## ***Install PaddlePaddle using pip***

First, we use the following commands to check if **the environment of this machine** is suitable for installing PaddlePaddle:

`Uname -m && cat /etc/*release`

> The above command will display the operating system and processing bits of the machine. Please make sure your computer is consistent with the requirements of this tutorial.

Second, your computer needs to meet the following requirements:

*	Python2.7.x (devel), Pip >= 9.0.1

    > CentOS6 needs to compile Python 2.7 into a [shared library](./FAQ.html/#FAQ).

*	Python3.5+.x (devel), Pip3 >= 9.0.1

	> You may have installed pip on your CentOS. Please use pip -V to confirm that we recommend using pip 9.0.1 or higher to install.

	Update the source of yum: `yum update` and install the extension source to install pip: `yum install -y epel-release`

	Use the following command to install or upgrade Python and pip to the required version:


	    - For Python2: `sudo yum install python-devel python-pip`
	    - For Python3: (Please refer to the official Python installation, and pay attention to whether the python3 version is consistent with the python version corresponding to the pip3 command. If there are multiple python3 versions, please specify the pip version such as pip3.7, or add soft link from pip3 to the python version you use. )




	> Even if you already have `Python` in your environment, you need to install the `python develop` package.

Here's how to install PaddlePaddle:

1. Use pip install to install PaddlePaddle:

	* For users who need **the CPU version PaddlePaddle**: `pip install paddlepaddle` or `pip3 install paddlepaddle`

	* For users who need **the GPU version PaddlePaddle**: `pip install paddlepaddle-gpu` or `pip3 install paddlepaddle-gpu`

	> 1 . In order to prevent problem "nccl.h cannot be found", please first install nccl2 according to the instructions of [NVIDIA official website](https://developer.nvidia.com/nccl/nccl-download).

	> 2 . If you do not specify the pypi package version number, we will by default provide you with a version of PaddlePaddle that supports Cuda 9/cuDNN v7.

	* For users with `Cannot uninstall 'six'.` problems, the probable reason is the existing Python installation issues in your system. In this case, use  `pip install paddlepaddle --ignore-installed six`(CPU) or `pip install paddlepaddle-gpu -- Ignore-installed six` (GPU)  to resolve.

	* For users with **other requirements**: `pip install paddlepaddle==[version number]` or `pip3 install paddlepaddle==[version number]`

	> For `the version number`, please refer to [the latest Release installation package list](./Tables.html/#whls). If you need to obtain and install **the latest PaddlePaddle development branch**, you can download and install the latest whl installation package and c-api development package from [the latest dev installation package list](./Tables.html/#ciwhls) or our [CI system](https://paddleci.ngrok.io/project.html?projectId=Manylinux1&tab=projectOverview). To log in, click on "Log in as guest".

Now you have completed the process of installing PaddlePaddle via `pip install`.


<br/><br/>
## *Install using Docker*

In order to better use Docker and avoid problems, we recommend using **the highest version of Docker**. For details on installing and using Docker, please refer to [the official Docker documentation](https://docs.docker.com/install/).

> Please note that to install and use the PaddlePaddle version that supports GPU, you must first install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Once you have **properly installed Docker**, you can start **installing PaddlePaddle with Docker**.

1. Use the following command to pull the image we pre-installed for PaddlePaddle:

	* For users who need a **CPU version of PaddlePaddle**, use the following command to pull the image we pre-installed for your *PaddlePaddle For CPU*:

		`Docker pull hub.baidubce.com/paddlepaddle/paddle: 1.2`

	* You can also pull any of our Docker images by following the instructions below:

		`Docker pull hub.baidubce.com/paddlepaddle/paddle:[tag]`

		> (Please replace [tag] with the contents of [the mirror table](./Tables.html/#dockers))


2. Use the following command to build from the already pulled image and enter the Docker container:

	`Docker run --name [Name of container] -it -v $PWD:/paddle <imagename> /bin/bash`

		> In the above command, --name [Name of container] sets the name of the Docker; the -it parameter indicates that the container is running interactively with the host machine; -v $PWD:/paddle specifies the current path (the PWD variable in Linux will expand to [The absolute path](https://baike.baidu.com/item/%E7%BB%9D%E5%AF%B9%E8%B7%AF%E5%BE%84/481185) of the current path ) which is mounted to the /paddle directory inside the container; `<imagename>` specifies the name of the image to use, if you need to use our image please use `hub.baidubce.com/paddlepaddle/paddle:[tag]`. Note: The meaning of the tag is the same as the second step. /bin/bash is the command to be executed in Docker.

3. (Optional: When you need to enter the Docker container a second time) re-enter the PaddlePaddle container with the following command:

	`Docker start [Name of container]`

	> start the container created previously

	`Docker attach [Name of container]`

	> Enter the started container in the last step.

Now that you have successfully installed PaddlePaddle using Docker, you only need to run PaddlePaddle after entering the Docker container. For more Docker usage, please refer to [the official Docker documentation](https://docs.docker.com/).

> Note: In order to reduce the size, `vim` is not installed in PaddlePaddle Docker image by default. You can edit the code in the container after executing `apt-get install -y vim` in the container.


<br/><br/>
## ***Verify installation***

After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle.fluid` to verify that the installation was successful.

<br/><br/>
## ***How to uninstall***

Please use the following command to uninstall PaddlePaddle (users who use Docker to install PaddlePaddle should use the following command in the container containing PaddlePaddle. Please use the corresponding version of pip):

* ***CPU version of PaddlePaddle***: `pip uninstall paddlepaddle` or `pip3 uninstall paddlepaddle`

* ***GPU version of PaddlePaddle***: `pip uninstall paddlepaddle-gpu` or `pip3 uninstall paddlepaddle-gpu`
