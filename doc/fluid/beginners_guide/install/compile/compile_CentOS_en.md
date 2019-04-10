***
# **Compile on CentOS from Source Code**

This instruction will show you how to compile PaddlePaddle on a 64-bit desktop or laptop and CentOS. The Centos systems we support must meet the following requirements:

* CentOS 7 / 6 (this involves whether the related tools can be installed normally)

## Determine which version to compile

* **Only PaddlePaddle for CPU is supported.**

## Choose a compilation method

We provide two compilation methods under the CentOS system:

* Compile with Docker (the CentOS 6 / 7 GPU version is not supported) (this image already contains python2.7, python3.6, python3.7 environment)
* Local compilation (does not support all versions of CentOS 6 and GPU versions of CentOS 7)

We recommend using **Docker for compilation** because we are installing both the tools and the configuration in a Docker image. This way, if you encounter problems, others can reproduce the problem to help. In addition, for developers accustomed to using Windows and MacOS, there is no need to configure a cross-compilation environment using Docker. It should be emphasized that Docker does not virtualize any hardware. The compiler tools running in the Docker container are actually running directly on the native CPU and operating system. The performance is the same as installing the compiler on the machine.

Also for those who can't install Docker for a variety of reasons, we also provide a way to **compile directly from sources**, but since the situation on host machine is more complicated, we only support specific systems.


### ***Compile with Docker***

In order to better use Docker and avoid problems, we recommend using **the highest version of Docker**. For details on **installing and using Docker**, please refer to the [official Docker documentation](https://docs.docker.com/install/).

Once you have **properly installed Docker**, you can start **compiling PaddlePaddle with Docker**:

1. First select the path where you want to store PaddlePaddle, then use the following command to clone PaddlePaddle's source code from github to a folder named Paddle in the local current directory:

	`git clone https://github.com/PaddlePaddle/Paddle.git`

2. Go to the Paddle directory: `cd Paddle`

3. Take advantage of the image we provided (with this command you don't have to download the image in advance):

	`docker run --name paddle-test -v $PWD:/paddle --network=host -it` `hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`

	> `--name paddle-test` names the Docker container you created as paddle-test, `-v $PWD:/paddle` mounts the current directory to the /paddle directory in the Docker container (the PWD variable in Linux will expand to the current [Absolute path](https://baike.baidu.com/item/%E7%BB%9D%E5%AF%B9%E8%B7%AF%E5%BE%84/481185)), `-it` keeps interacting with the host, `hub.baidubce.com/paddlepaddle/paddle` creates a Docker container with an image called `hub.baidubce.com/paddlepaddle/paddle:latest-dev`, /bin/bash enters the container After starting the `/bin/bash` command.

4. After entering Docker, go to the paddle directory: `cd paddle`

5. Switch to a more stable version to compile:

	`git checkout v1.1`

6. Create and enter the /paddle/build path:

	`mkdir -p /paddle/build && cd /paddle/build`

7. Use the following command to install the dependencies: (For Python3: Please select the pip for the python version you wish to use, such as pip3.5, pip3.6)


		For Python2: pip install protobuf==3.1.0
		For Python3: pip3.5 install protobuf==3.1.0


	> Install protobuf 3.1.0

	`apt install patchelf`

	> Installing patchelf, PatchELF is a small and useful program for modifying the dynamic linker and RPATH of ELF executables.

8. Execute cmake:

	> For details on the compilation options, see the [compilation options table](../Tables.html/#Compile).

	* For users who need to compile the **CPU version PaddlePaddle**:

		`cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

	> We currently do not support the compilation of the GPU version PaddlePaddle under CentOS.

9. Execute compilation:

	`make -j$(nproc)`

	> Use multicore compilation

10. After compiling successfully, go to the `/paddle/build/python/dist` directory and find the generated `.whl` package: `cd /paddle/build/python/dist`

11. Install the compiled `.whl` package on the current machine or target machine: (For Python3: Please select the pip corresponding to the python version you wish to use, such as pip3.5, pip3.6)


		For Python2: pip install (whl package name)
		For Python3: pip3.5 install (whl package name)


Now that you have successfully installed PaddlePaddle using Docker, you only need to run PaddlePaddle after entering the Docker container. For more Docker usage, please refer to the [official Docker documentation](https://docs.docker.com/).

> Notes: In order to reduce the size, `vim` is not installed in PaddlePaddle Docker image by default. You can edit the code in the container after executing `apt-get install -y vim` in the container.

Congratulations, you have now completed the process of compiling PaddlePaddle using Docker.



<br/><br/>
### *Local compilation*

**Please strictly follow the order of the following instructions**


1. Check that your computer and operating system meet the compilation standards we support: `uname -m && cat /etc/*release`

2. Update the source of `yum`: `yum update`, and add the necessary yum source: `yum install -y epel-release`, and install openCV in advance

3. Install the necessary tools `bzip2` and `make`: `yum install -y bzip2 `, `yum install -y make`

4. We support compiling and installing with virtualenv. First, create a virtual environment called `paddle-venv` with the following command:

	* a. Install Python-dev:

			For Python2: yum install python-devel
			For Python3: (Please refer to the official Python installation process)


	* b. Install pip:


			For Python2: yum install python-pip (please have a pip version of 9.0.1 and above)
			For Python3: (Please refer to the official Python installation process, and ensure that the pip3 version 9.0.1 and above, please note that in python3.6 and above, pip3 does not necessarily correspond to the python version, such as python3.7 default only Pip3.7)

	* c. (Only For Python3) set Python3 related environment variables, here is python3.5 version example, please replace with the version you use (3.6, 3.7):

		1. First find the path to the Python lib using ``` find `dirname $(dirname
			$(which python3))` -name "libpython3.so"``` . If it is 3.6 or 3.7, change `python3` to `python3.6` or `python3.7`, then replace [python-lib-path] in the following steps with the file path found.

		2. Set PYTHON_LIBRARIES: `export PYTHON_LIBRARY=[python-lib-path]`.

		3. Secondly, use ```find `dirname $(dirname
			$(which python3))`/include -name "python3.5m"``` to find the path to Python Include, please pay attention to the python version, then replace the following [python-include-path] to the file path found.

		4. Set PYTHON_INCLUDE_DIR: `export PYTHON_INCLUDE_DIRS=[python-include-path]`

		5. Set the system environment variable path: `export PATH=[python-lib-path]:$PATH `(here replace the last two levels content of [python-lib-path] with /bin/)

	* d. Install the virtual environment `virtualenv` and `virtualenvwrapper` and create a virtual environment called `paddle-venv`: (please note the pip3 commands corresponding to the python version, such as pip3.6, pip3.7)

		1. `pip install virtualenv` or `pip3 install virtualenv`

		2. `Pip install virtualenvwrapper` or `pip3 install virtualenvwrapper`

		3. Find `virtualenvwrapper.sh`: `find / -name virtualenvwrapper.sh` (please find the corresponding Python version of `virtualenvwrapper.sh`)

		4. See the installation method in `virtualenvwrapper.sh`: `cat vitualenvwrapper.sh`

		5. Install `virtualwrapper`

		6. Create a virtual environment called `paddle-venv`: `mkvirtualenv paddle-venv`

5. Enter the virtual environment: `workon paddle-venv`

6. Before **executing the compilation**, please confirm that the related dependencies mentioned in the [compile dependency table](../Tables.html/#third_party) are installed in the virtual environment:

	* Here is the installation method for `patchELF`. Other dependencies can be installed using `yum install` or `pip install`/`pip3 install` followed by the name and version:

	`yum install patchelf`
	> Users who can't use apt installation can refer to patchElF [github official documentation](https://gist.github.com/ruario/80fefd174b3395d34c14).

7. Put the PaddlePaddle source cloned in the Paddle folder in the current directory and go to the Paddle directory:

	- `git clone https://github.com/PaddlePaddle/Paddle.git`

	- `cd Paddle`

8. Switch to a more stable release branch for compilation (support for Python 3.6 and 3.7 is added from the 1.2 branch):

	- `git checkout release/1.2`

9. And please create and enter a directory called build:

	- `mkdir build && cd build`

10. Execute cmake:

	> For details on the compilation options, see the [compilation options table](../Tables.html/#Compile).

	* For users who need to compile the **CPU version PaddlePaddle**:


			For Python2: cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
			For Python3: cmake .. -DPY_VERSION=3.5 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
			-DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release


		> If you encounter `Could NOT find PROTOBUF (missing: PROTOBUF_LIBRARY PROTOBUF_INCLUDE_DIR)`, you can re-execute the cmake command. 
		> Please note that the PY_VERSION parameter is replaced with the python version you need.

11. Compile with the following command:

	`make -j$(nproc)`

12. After compiling successfully, go to the `/paddle/build/python/dist `directory and find the generated `.whl` package: `cd /paddle/build/python/dist`

13. Install the compiled `.whl` package on the current machine or target machine:

	`Pip install (whl package name) `or `pip3 install (whl package name)`

Congratulations, now you have completed the process of compiling PaddlePaddle natively.

<br/><br/>
### ***Verify installation***

After the installation is complete, you can use `python` to enter the Python interpreter and then use `import paddle.fluid` to verify that the installation was successful.

<br/><br/>
### ***How to uninstall***

Please use the following command to uninstall PaddlePaddle (users who use Docker to install PaddlePaddle should use the following command in the container containing PaddlePaddle. Please use the corresponding version of pip):

* ***CPU version of PaddlePaddle***: `pip uninstall paddlepaddle` or `pip3 uninstall paddlepaddle`
