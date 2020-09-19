# **Compile on CentOS from Source Code**

## Environment preparation

* **CentOS version (64 bit)**
    * **CentOS 6 (not recommended, no official support for compilation problems)**
    * **CentOS 7 (GPU version supports CUDA 9.0/9.1/9.2/10.0/10.1 CUDA 9.1, only support single-card mode)**
* **Python version 2.7.15+/3.5.1+/3.6/3.7 (64 bit)**
* **pip or pip3 version 9.0.1+ (64 bit)**

## Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install CPU version of PaddlePaddle

* If your computer has NVIDIA® GPU, and the following conditions are met，GPU version of PaddlePaddle is recommended.

    * **CUDA toolkit 10.0 with cuDNN v7.3+(for multi card support, NCCL2.3.7 or higher)**
    * **CUDA toolkit 9.0 with cuDNN v7.3+(for multi card support, NCCL2.3.7 or higher)**
    * **Hardware devices with GPU computing power over 1.0**

        You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

* * If you need to use multi card environment, please make sure that you have installed nccl2 correctly, or install nccl2 according to the following instructions (here is the installation instructions of nccl2 under ubuntu 16.04, CUDA9 and cuDNN7). For more version of installation information, please refer to NVIDIA[official website](https://developer.nvidia.com/nccl):


        wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        yum update -y
        yum install -y libnccl-2.3.7-2+cuda9.0 libnccl-devel-2.3.7-2+cuda9.0 libnccl-static-2.3.7-2+cuda9.0


## Installation steps

There are two compilation methods under CentOS system:

* Compile with Docker(GPU version only supports CentOS 7)
* Local compilation (no official support for compilation problems under CentOS 6)

<a name="ct_docker"></a>
### **Compile with Docker**

[Docker](https://docs.docker.com/install/) is an open source application container engine. Using docker, you can not only isolate the installation and use of paddlepaddle from the system environment, but also share GPU, network and other resources with the host

Compiling PaddlePaddle with Docker，you need:

- On the local host [Install Docker](https://hub.docker.com/search/?type=edition&offering=community)

- To enable GPU support on Linux, please [Install nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Please follow the steps below to install:

1. First select the path where you want to store PaddlePaddle, then use the following command to clone PaddlePaddle's source code from github to a folder named Paddle in the local current directory:

    `git clone https://github.com/PaddlePaddle/Paddle.git`

2. Go to the Paddle directory: `cd Paddle`

3. Create and enter a Docker container that meets the compilation environment:

    * Compile CPU version of PaddlePaddle：



        `docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`

        > --name paddle-test names the Docker container you created as paddle-test;


        > -v $PWD:/paddle mount the current directory to the /paddle directory in the docker container (PWD variable in Linux will be expanded to [absolute path](https://baike.baidu.com/item/绝对路径/481185) of the current path);


        > -it keeps interaction with the host，`hub.baidubce.com/paddlepaddle/paddle:latest-dev` use the image named `hub.baidubce.com/paddlepaddle/paddle:latest-dev` to create Docker container, /bin/bash start the /bin/bash command after entering the container.



    * Compile GPU version of PaddlePaddle (only supports CentOS 7):



        `nvidia-docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`

        > --name paddle-test names the Docker container you created as paddle-test;


        > -v $PWD:/paddle mount the current directory to the /paddle directory in the docker container (PWD variable in Linux will be expanded to [absolute path](https://baike.baidu.com/item/绝对路径/481185) of the current path);



        > -it keeps interaction with the host，`hub.baidubce.com/paddlepaddle/paddle:latest-dev` use the image named `hub.baidubce.com/paddlepaddle/paddle:latest-dev` to create Docker container, /bin/bash start the /bin/bash command after entering the container.


        > Note: hub.baidubce.com/paddlepaddle/paddle:latest-dev internally install CUDA 10.0.


4. After entering Docker, go to the paddle directory: `cd paddle`

5. Switch to a more stable version to compile:

    `git checkout [name of the branch]`

    For example：

    `git checkout release/1.5`

    Note: python3.6、python3.7 version started supporting from release/1.2 branch

6. Create and enter the /paddle/build path:

    `mkdir -p /paddle/build && cd /paddle/build`

7. Use the following command to install the dependencies:


        For Python2: pip install protobuf
        For Python3: pip3.5 install protobuf

    Note: We used Python3.5 command as an example above, if the version of your Python is 3.6/3.7, please change Python3.5 in the commands to Python3.6/Python3.7

    > Install protobuf 3.1.0

    `yum install patchelf`

    > Installing patchelf, PatchELF is a small and useful program for modifying the dynamic linker and RPATH of ELF executables.

8. Execute cmake:

    > For details on the compilation options, see the [compilation options table](../Tables.html/#Compile).
    > Please attention to modify parameters `-DPY_VERSION` for the version of Python you want to compile with, for example `-DPY_VERSION=3.5` means the version of python is 3.5.x

    * For users who need to compile the **CPU version PaddlePaddle**:

        `cmake .. -DPY_VERSION=3.5 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

    * For users who need to compile the **GPU version PaddlePaddle**:
        `cmake .. -DPY_VERSION=3.5 -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

    > We currently do not support the compilation of the GPU version PaddlePaddle under CentOS 6.

9. Execute compilation:

    `make -j$(nproc)`

    > Use multicore compilation

10. After compiling successfully, go to the `/paddle/build/python/dist` directory and find the generated `.whl` package: `cd /paddle/build/python/dist`

11. Install the compiled `.whl` package on the current machine or target machine:

        For Python2: pip install -U (whl package name)
        For Python3: pip3.5 install -U (whl package name)

    Note: For the command involving Python 3, we use Python 3.5 as an example above, if the version of your Python is 3.6/3.7, please change Python3.5 in the commands to Python3.6/Python3.7

Congratulations, now that you have successfully installed PaddlePaddle using Docker, you only need to run PaddlePaddle after entering the Docker container. For more Docker usage, please refer to the [official Docker documentation](https://docs.docker.com/).

> Note: In order to reduce the size, `vim` is not installed in PaddlePaddle Docker image by default. You can edit the code in the container after executing `yum install -y vim` in the container.

<a name="ct_source"></a>
### **Local compilation**

1. Check that your computer and operating system meet the compilation standards we support: `uname -m && cat /etc/*release`

2. Update the source of `yum`: `yum update`, and add the necessary yum source: `yum install -y epel-release`, and install [OpenCV](https://opencv.org/releases.html) in advance

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

        4. See the installation method in `virtualenvwrapper.sh`: `cat vitualenvwrapper.sh`, this shell file describes the steps and commands

        5. Install `virtualwrapper` as described in `virtualenvwrapper.sh`

        6. Set VIRTUALENVWRAPPER_PYTHON：`export VIRTUALENVWRAPPER_PYTHON=[python-lib-path]:$PATH` (here replace the last two levels content of [python-lib-path] with /bin/)
        7. Create virtual environment named `paddle-venv`: `mkvirtualenv paddle-venv`

5. Enter the virtual environment: `workon paddle-venv`

6. Before **executing the compilation**, please confirm that the related dependencies mentioned in the [compile dependency table](../Tables.html/#third_party) are installed in the virtual environment:

    * Here is the installation method for `patchELF`. Other dependencies can be installed using `yum install` or `pip install`/`pip3 install` followed by the name and version:

    `yum install patchelf`
    > Users who can't use yum installation can refer to patchElF github [official documentation](https://gist.github.com/ruario/80fefd174b3395d34c14).

7. Put the PaddlePaddle source cloned in the Paddle folder in the current directory and go to the Paddle directory:

    - `git clone https://github.com/PaddlePaddle/Paddle.git`

    - `cd Paddle`

8. Switch to a more stable release branch for compilation (support for Python 3.6 and 3.7 is added from the 1.2 branch):

    - `git checkout [name of target branch]`

    For example:

    `git checkout release/1.5`

9. And please create and enter a directory called build:

    - `mkdir build && cd build`

10. Execute cmake:

    > For details on the compilation options, see the [compilation options table](../Tables.html/#Compile).

    * For users who need to compile the **CPU version PaddlePaddle**:


            For Python2: cmake .. -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
            For Python3: cmake .. -DPY_VERSION=3.5 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
            -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release


        > If you encounter `Could NOT find PROTOBUF (missing: PROTOBUF_LIBRARY PROTOBUF_INCLUDE_DIR)`, you can re-execute the cmake command.
        > Please note that the PY_VERSION parameter is replaced with the python version you need.


    * For users who need to compile the **GPU version PaddlePaddle**:

        1. Please make sure that you have installed nccl2 correctly, or install nccl2 according to the following instructions (here is ubuntu 16.04, CUDA9, ncDNN7 nccl2 installation instructions), for more information on the installation information please refer to the [NVIDIA official website](https://developer.nvidia.com/nccl/nccl-download):

            i.  `wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm`

            ii. `rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm`

            iii. `yum install -y libnccl-2.3.7-2+cuda9.0 libnccl-devel-2.3.7-2+cuda9.0 libnccl-static-2.3.7-2+cuda9.0`

        2. If you have already installed `nccl2` correctly, you can start cmake: *(For Python3: Please configure the correct python version for the PY_VERSION parameter)*

                 For Python2: cmake .. -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
                For Python3: cmake .. -DPYTHON_EXECUTABLE:FILEPATH=[您可执行的Python3的路径] -DPYTHON_INCLUDE_DIR:PATH=[之前的PYTHON_INCLUDE_DIRS] -DPYTHON_LIBRARY:FILEPATH=[之前的PYTHON_LIBRARY] -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release


    Note: For the command involving Python 3, we use Python 3.5 as an example above, if the version of your Python is 3.6/3.7, please change Python3.5 in the commands to Python3.6/Python3.7



11. Compile with the following command:

    `make -j$(nproc)`

    > Use multicore compilation

    > If “Too many open files” error is displayed during compilation, please use the instruction ulimit -n 8192  to increase the number of files allowed to be opened by the current process. Generally speaking, 8192 can ensure the completion of compilation.

12. After compiling successfully, go to the `/paddle/build/python/dist `directory and find the generated `.whl` package: `cd /paddle/build/python/dist`

13. Install the compiled `.whl` package on the current machine or target machine:

    `Pip install -U (whl package name) `or `pip3 install -U (whl package name)`

Congratulations, now you have completed the process of compiling PaddlePaddle natively.

<br/><br/>
### ***Verify installation***

After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle.fluid as fluid` and then  `fluid.install_check.run_check()` to verify that the installation was successful.

If `Your Paddle Fluid is installed succesfully!` appears, it means the installation was successful.

<br/><br/>
### ***How to uninstall***

Please use the following command to uninstall PaddlePaddle (users who use Docker to install PaddlePaddle should use the following command in the container containing PaddlePaddle. Please use the corresponding version of pip):

* ***CPU version of PaddlePaddle***: `pip uninstall paddlepaddle` or `pip3 uninstall paddlepaddle`
* ***GPU version of PaddlePaddle***: `pip uninstall paddlepaddle-gpu` or `pip3 uninstall paddlepaddle-gpu`

Users installing PaddlePaddle with Docker, please use above commands in the container involved PaddlePaddle and attention to use the corresponding version of Pip
