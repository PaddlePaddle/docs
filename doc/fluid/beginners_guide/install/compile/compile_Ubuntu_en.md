***
# **Compile on Ubuntu from Source Code**

This instruction describes how to compile PaddlePaddle on *64-bit desktops or laptops* and Ubuntu systems. The Ubuntu systems we support must meet the following requirements:

* Ubuntu 14.04/16.04/18.04 (this involves whether the related tools can be installed successfully)

## Determine which version to compile

* **CPU version of PaddlePaddle**, if your system does not have an NVIDIA® GPU, you must install this version. This version is easier than the GPU version. So even if you have a GPU on your computer, we recommend that you first install the CPU version of PaddlePaddle to check if your local environment is suitable.

* **GPU version of PaddlePaddle**, in order to make the PaddlePaddle program run more quickly, we usually use the GPU to accelerate the PaddlePaddle program, but the GPU version of PaddlePaddle needs to have the NVIDIA® GPU that meets the following conditions (see NVIDIA for the specific installation process and configuration). Official documentation: [For CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/), For [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/))

	* *CUDA Toolkit 9.0 with cuDNN v7*
	* *CUDA Toolkit 8.0 with cuDNN v7*
	* *Hardware devices with GPU compute capability exceeding 1.0*

## Choose a compilation method

Under Ubuntu's system we offer 2 ways to compile:

* Compile with Docker (this image already contains python2.7, python3.6, python3.7 environment)

* Local compilation (does not support GPU version under ubuntu18.04)

We recommend using **Docker for compilation** because we are installing both the tools and the configuration in a Docker image. This way, if you encounter problems, others can reproduce the problem to help. In addition, for developers accustomed to using Windows and MacOS, there is no need to configure a cross-compilation environment using Docker. Someone uses a virtual machine to analogize to Docker. It should be emphasized that Docker does not virtualize any hardware. The compiler tools running in the Docker container are actually running directly on the native CPU and operating system. The performance is the same as installing the compiler on the machine.

We also provide methods that can be **compiled from local source code**, but since the situation on host machine is more complicated, we only provide support for specific systems.

<br/><br/>
## ***Compile with Docker***

In order to better use Docker and avoid problems, we recommend using **the highest version of Docker**. For details on **installing and using Docker**, please refer to [the official Docker documentation](https://docs.docker.com/install/).

> Please note that to install and use the PaddlePaddle version that supports GPU, you must first install nvidia-docker

Once you have **properly installed Docker**, you can start **compiling PaddlePaddle with Docker**:

1. First select the path where you want to store PaddlePaddle, then use the following command to clone PaddlePaddle's source code from github to a folder named Paddle in the local current directory:

	`git clone https://github.com/PaddlePaddle/Paddle.git`

2. Go to the Paddle directory: `cd Paddle`

3. Take advantage of the image we provided (with this command you don't have to download the image in advance):

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`

	> --name paddle-test names the Docker container you created as paddle-test, -v $PWD:/paddle mounts the current directory to the /paddle directory in the Docker container (the PWD variable in Linux will expand to the current path's [absolute path](https://baike.baidu.com/item/%E7%BB%9D%E5%AF%B9%E8%B7%AF%E5%BE%84/481185)), -it keeps interacting with the host, `hub.baidubce.com/paddlepaddle/paddle:latest-dev` creates a Docker container with a mirror named `hub.baidubce.com/paddlepaddle/paddle:latest-dev`, /bin /bash Starts the /bin/bash command after entering the container.

4. After entering Docker, go to the paddle directory: `cd paddle`

5. Switch to a more stable release branch to compile: (Note that python 3.6, python 3.7 version are supported from the 1.2 branch)

	`git checkout release/1.2`

6. Create and enter the /paddle/build path:

	`mkdir -p /paddle/build && cd /paddle/build`

7. Use the following command to install the dependencies: (For Python3: Please select the pip for the python version you wish to use, such as pip3.5, pip3.6)

		For Python2: pip install protobuf==3.1.0
		For Python3: pip3.5 install protobuf==3.1.0


	> Install protobuf 3.1.0.

	`apt install patchelf`

	> Installing patchelf, PatchELF is a small and useful program for modifying the dynamic linker and RPATH of ELF executables.

8. Execute cmake:

	> For the meaning of the specific compiling options, [compilation options table](../Tables.html/#Compile) is your resort. Please note that the parameter `-DPY_VERSION` is the python version used in your current environment.

	* For users who need to compile the **CPU version PaddlePaddle**:

		`cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

	* For users who need to compile the **GPU version PaddlePaddle**:

		`cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

9. Execute compilation:

	`make -j$(nproc)`

	> Use multicore compilation

10. After compiling successfully, go to the `/paddle/build/python/dist` directory and find the generated `.whl` package: `cd /paddle/build/python/dist`

11. Install the compiled `.whl` package on the current machine or target machine: (For Python3: Please select the pip corresponding to the python version you wish to use, such as pip3.5, pip3.6)

		For Python2: pip install (whl package name)
		For Python3: pip3.5 install (whl package name)


Now that you have successfully installed PaddlePaddle using Docker, you only need to run PaddlePaddle after entering the Docker container. For more Docker usage, please refer to the [official Docker documentation](https://docs.docker.com/).

> Note: In order to reduce the size, `vim` is not installed in PaddlePaddle Docker image by default. You can edit the code in the container after executing `apt-get install -y vim` in the container.

Congratulations, you have now completed the process of compiling PaddlePaddle using Docker.


<br/><br/>
### ***Local compilation***

**Please strictly follow the following instructions step by step**

1. Check that your computer and operating system meet the compilation standards we support: `uname -m && cat /etc/*release`

2. Update the source of `apt`: `apt update`, and install openCV in advance.

3. We support compiling and installing with virtualenv. First, create a virtual environment called `paddle-venv` with the following command:

	* a. Install Python-dev: (Please install python3.x-dev that matches the current environment python version)

			For Python2: apt install python-dev
			For Python3: apt install python3.5-dev

	* b. Install pip: (Please ensure that pip version is 9.0.1 and above ): (Please note that the version corresponding to python3 is modified)

			For Python2: apt install python-pip
			For Python3: apt-get udpate && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt install curl && curl https://bootstrap.pypa.io/get-pip. Py -o - | python3.5 && easy_install pip


	* c. Install the virtual environment `virtualenv` and `virtualenvwrapper` and create a virtual environment called `paddle-venv` (please note the python version) : 

		1. `apt install virtualenv` or `pip install virtualenv` or `pip3 install virtualenv`
		2. `apt install virtualenvwrapper` or `pip install virtualenvwrapper` or `pip3 install virtualenvwrapper`
		3. Find `virtualenvwrapper.sh`: `find / -name virtualenvwrapper.sh`
		4. (Only for Python3) Set the interpreter path for the virtual environment: `export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.5`
		5. See the installation method in `virtualenvwrapper.sh`: `cat virtualenvwrapper.sh`
		6. Install `virtualwrapper` according to the installation method in `virtualenvwrapper.sh`
		7. Create a virtual environment called `paddle-venv`: `mkvirtualenv paddle-venv`

4. Enter the virtual environment: `workon paddle-venv`

5. Before **executing the compilation**, please confirm that the related dependencies mentioned in [the compile dependency table](../Tables.html/#third_party) are installed in the virtual environment:

	* Here is the installation method for `patchELF`. Other dependencies can be installed using `apt install` or `pip install` followed by the name and version:

		`apt install patchelf`

		> Users who can't use apt installation can refer to patchElF [github official documentation](https://gist.github.com/ruario/80fefd174b3395d34c14).

6. Clone the PaddlePaddle source code in the Paddle folder in the current directory and go to the Paddle directory:

	- `git clone https://github.com/PaddlePaddle/Paddle.git`

	- `cd Paddle`

7. Switch to a more stable release branch to compile, replacing the brackets and their contents with **the target branch name**:

	- `git checkout [name of target branch]`

8. And please create and enter a directory called build:

	`mkdir build && cd build`

9. Execute cmake:

	> For details on the compilation options, see [the compilation options table](../Tables.html/#Compile).

	* For users who need to compile the **CPU version of PaddlePaddle**: (For Python3: Please configure the correct python version for the PY_VERSION parameter)

			For Python2: cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release	
			For Python3: cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release


	* For users who need to compile **GPU version of PaddlePaddle**: (*only support ubuntu16.04/14.04*)

		1. Please make sure that you have installed nccl2 correctly, or install nccl2 according to the following instructions (here is ubuntu 16.04, CUDA9, ncDNN7 nccl2 installation instructions), for more information on the installation information please refer to the [NVIDIA official website](https://developer.nvidia.com/nccl/nccl-download):

			i. `wget http: / /developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb `

			ii. `dpkg -i nvidia-machine-learning-repo-ubuntu1604_1 .0.0-1_amd64.deb`

			iii. `sudo apt-get install -y libnccl2=2.2.13-1+cuda9.0 libnccl-dev=2.2.13-1+cuda9.0`

		2. If you have already installed `nccl2` correctly, you can start cmake: *(For Python3: Please configure the correct python version for the PY_VERSION parameter)*

				For Python2: cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
				For Python3: cmake .. -DPY_VERSION=3.5 -DWITH_FLUID_ONLY=ON -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release


			> `-DPY_VERSION=3.5` Please change to the Python version of the installation environment

10. Compile with the following command:

	`make -j$(nproc)`

11. After compiling successfully, go to the `/paddle/build/python/dist `directory and find the generated `.whl` package: `cd /paddle/build/python/dist`

12. Install the compiled `.whl` package on the current machine or target machine:

	`Pip install (whl package name)` or `pip3 install (whl package name)`

Congratulations, now you have completed the process of compiling PaddlePaddle natively.

<br/><br/>
### ***Verify installation***

After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle.fluid` to verify that the installation was successful.

<br/><br/>
### ***How to uninstall***
Please use the following command to uninstall PaddlePaddle (users who use Docker to install PaddlePaddle should use the following command in the container containing PaddlePaddle. Please use the corresponding version of pip):

- ***CPU version of PaddlePaddle***: `pip uninstall paddlepaddle` or `pip3 uninstall paddlepaddle`

- ***GPU version of PaddlePaddle***: `pip uninstall paddlepaddle-gpu` or `pip3 uninstall paddlepaddle-gpu`
