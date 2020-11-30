# **Compile on Ubuntu from Source Code**

## Environment preparation

* **Ubuntu version (64 bit)**
    * **Ubuntu 14.04 (GPU version supports CUDA 10.0/10.1)**
    * **Ubuntu 16.04 (GPU version supports CUDA 9.0/9.1/9.2/10.0/10.1)**
    * **Ubuntu 18.04 (GPU version supports CUDA 10.0/10.1)**
* **Python version 2.7.15+/3.5.1+/3.6/3.7 (64 bit)**
* **pip or pip3 version 9.0.1+ (64 bit)**

## Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install CPU version of PaddlePaddle

* If your computer has NVIDIA® GPU, and the following conditions are met，GPU version of PaddlePaddle is recommended.
    * **CUDA toolkit 10.0 with cuDNN v7.3+(for multi card support, NCCL2.3.7 or higher)**
    * **CUDA toolkit 9.0 with cuDNN v7.3+(for multi card support, NCCL2.3.7 or higher)**
    * **CUDA toolkit 8.0 with cuDNN v7.1+(for multi card support, NCCL2.1.15-2.2.13）**
    * **Hardware devices with GPU computing power over 1.0**

        You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)

* * If you need to use multi card environment, please make sure that you have installed nccl2 correctly, or install nccl2 according to the following instructions (here is the installation instructions of nccl2 under ubuntu 16.04, CUDA9 and cuDNN7). For more version of installation information, please refer to NVIDIA[official website](https://developer.nvidia.com/nccl):


        wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
        dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
        sudo apt-get install -y libnccl2=2.3.7-1+cuda9.0 libnccl-dev=2.3.7-1+cuda9.0

## Installation steps

There are two compilation methods in Ubuntu system:

* Compile with docker (GPU version under Ubuntu 18.04 is not supported temporarily)
* Local compilation

<a name="ubt_docker"></a>
### **Compile with docker**

[Docker](https://docs.docker.com/install/) is an open source application container engine. Using docker, you can not only isolate the installation and use of paddlepaddle from the system environment, but also share GPU, network and other resources with the host

Compiling PaddlePaddle with Docker，you need:

- On the local host [Install Docker](https://hub.docker.com/search/?type=edition&offering=community)

- To enable GPU support on Linux, please [Install nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Please follow the steps below to install:

1. First, select the path where you want to store PaddlePaddle, and then use the following command to clone the source code of PaddlePaddle from GitHub to the folder named Paddle under the current local directory:

    `git clone https://github.com/PaddlePaddle/Paddle.git`

2. Enter the Paddle Directory: `cd Paddle`

3. Create and enter a Docker container that meets the compilation environment:

    * Compile CPU version of PaddlePaddle:



        `docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`

        > --name paddle-test names the Docker container you created as paddle-test;


        > -v $PWD:/paddle mount the current directory to the /paddle directory in the docker container (PWD variable in Linux will be expanded to [absolute path](https://baike.baidu.com/item/绝对路径/481185) of the current path);


        > -it keeps interaction with the host，`hub.baidubce.com/paddlepaddle/paddle:latest-dev` use the image named `hub.baidubce.com/paddlepaddle/paddle:latest-dev` to create Docker container, /bin/bash start the /bin/bash command after entering the container.


    * Compile GPU version of PaddlePaddle:



        `nvidia-docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash`

        > --name paddle-test names the Docker container you created as paddle-test;


        > -v $PWD:/paddle mount the current directory to the /paddle directory in the docker container (PWD variable in Linux will be expanded to [absolute path](https://baike.baidu.com/item/绝对路径/481185) of the current path);


        > -it keeps interaction with the host，`hub.baidubce.com/paddlepaddle/paddle:latest-dev` use the image named `hub.baidubce.com/paddlepaddle/paddle:latest-dev` to create Docker container, /bin/bash start the /bin/bash command after entering the container.


        > Note: hub.baidubce.com/paddlepaddle/paddle:latest-dev internally install CUDA 10.0.

4. After entering Docker, enter the Paddle Directory:

    `cd paddle`

5. Switch to a more stable release branch for compilation:

    `git checkout [name of the branch]`

    For example：

    `git checkout release/1.5`

    Note: python3.6、python3.7 version started supporting from release/1.2 branch

6. Create and enter /paddle/build Directory:

    `mkdir -p /paddle/build && cd /paddle/build`

7. Use the following command to install dependencies:

        For Python2: pip install protobuf
        For Python3: pip3.5 install protobuf

    Note: We used Python3.5 command as an example above, if the version of your Python is 3.6/3.7, please change Python3.5 in the commands to Python3.6/Python3.7

    > Install protobuf

    `apt install patchelf`

    > Install patchelf
    This is a small but useful program, it can be used to modify dynamic linker and RPATH of ELF executable

8. Execute cmake：

    > For the specific meaning of compilation options, you can read [Compile options table](../Tables.html#Compile)

    > Please attention to modify parameters `-DPY_VERSION` for the version of Python you want to compile with, for example `-DPY_VERSION=3.5` means the version of python is 3.5.x

    *  Compile**CPU version of PaddlePaddle**:

        `cmake .. -DPY_VERSION=3.5 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

    *  Compile**GPU version of PaddlePaddle**：

        `cmake .. -DPY_VERSION=3.5 -DWITH_GPU=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release`

9. Execute compiling：

    `make -j$(nproc)`

    > Use multicore compilation

10. after compiling successful, enter `/paddle/build/python/dist` Directory and find generated `.whl` package: `cd /paddle/build/python/dist`

11. Install the compiled `.whl` package on the current machine or target machine:

        For Python2: pip install -U（whl package name）
        For Python3: pip3.5 install -U（whl package name）

    Note: For the command involving Python 3, we use Python 3.5 as an example above, if the version of your Python is 3.6/3.7, please change Python3.5 in the commands to Python3.6/Python3.7

Congratulations, now you have completed the compilation and installation of PaddlePaddle. You only need to enter the Docker container and run PaddlePaddle to start using. For more Docker usage, please refer to [official docker documentation](https://docs.docker.com)

> Note: In order to reduce the size, `vim` is not installed in PaddlePaddle Docker image by default. You can edit the code in the container after executing `apt-get install -y vim` in the container.

<a name="ubt_source"></a>
### ***Local compilation***

**Please strictly follow the following instructions step by step**

1. Check that your computer and operating system meet the compilation standards we support: `uname -m && cat /etc/*release`

2. Update the source of `apt`: `apt update`, and install openCV in advance.

3. We support compiling and installing with virtualenv. First, create a virtual environment called `paddle-venv` with the following command:

    * a. Install Python-dev: (Please note that gcc4.8 is not supported in python2.7 under Ubuntu 16.04, please use gcc5.4 to compile Paddle)

            For Python2: apt install python-dev
            For Python3: apt install python3.5-dev

    * b. Install pip: (Please ensure that pip version is 9.0.1 and above ):

            For Python2: apt install python-pip
            For Python3: apt-get udpate && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt install curl && curl https://bootstrap.pypa.io/get-pip. Py -o - | python3.5 && easy_install pip


    * c. Install the virtual environment `virtualenv` and `virtualenvwrapper` and create a virtual environment called `paddle-venv` :

        1.  `apt install virtualenv` or `pip install virtualenv` or `pip3 install virtualenv`
        2. `apt install virtualenvwrapper` or `pip install virtualenvwrapper` or `pip3 install virtualenvwrapper`
        3. Find `virtualenvwrapper.sh`: `find / -name virtualenvwrapper.sh`
        4. (Only for Python3) Set the interpreter path for the virtual environment: `export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.5`
        5. See the installation method in `virtualenvwrapper.sh`: `cat virtualenvwrapper.sh`, this shell file describes the steps and commands
        6. Install `virtualwrapper` according to the installation method in `virtualenvwrapper.sh`
        7. Set VIRTUALENVWRAPPER_PYTHON：`export VIRTUALENVWRAPPER_PYTHON=[python-lib-path]:$PATH` （Here, replace the last two directories of [python-lib-path] with /bin/)
        8. Create a virtual environment called `paddle-venv`: `mkvirtualenv paddle-venv`

    Note: for the above commands involving Python 3, we use Python 3.5 as an example. If your Python version is 3.6 / 3.7, please change Python 3.5 in the above commands to Python 3.6 / Python 3.7

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

    For example:

    `git checkout release/1.5`

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


    Note: We used Python3.5 command as an example above, if the version of your Python is 3.6/3.7, please change Python3.5 in the commands to Python3.6/Python3.7

10. Compile with the following command:

    `make -j$(nproc)`

    > compile using multi-core

    > If “Too many open files” error is displayed during compilation, please use the instruction ulimit -n 8192  to increase the number of files allowed to be opened by the current process. Generally speaking, 8192 can ensure the completion of compilation.

11. After compiling successfully, go to the `/paddle/build/python/dist `directory and find the generated `.whl` package: `cd /paddle/build/python/dist`

12. Install the compiled `.whl` package on the current machine or target machine:

    `Pip install (whl package name)` or `pip3 install (whl package name)`

Congratulations, now you have completed the process of compiling PaddlePaddle natively.

<br/><br/>
### ***Verify installation***

After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use `import paddle.fluid as fluid` and then  `fluid.install_check.run_check()` to verify that the installation was successful.

If `Your Paddle Fluid is installed succesfully!` appears, it means the installation was successful.

<br/><br/>
### ***How to uninstall***
Please use the following command to uninstall PaddlePaddle:

- ***CPU version of PaddlePaddle***: `pip uninstall paddlepaddle` or `pip3 uninstall paddlepaddle`

- ***GPU version of PaddlePaddle***: `pip uninstall paddlepaddle-gpu` or `pip3 uninstall paddlepaddle-gpu`

Users installing PaddlePaddle with Docker, please use above commands in the container involved PaddlePaddle and attention to use the corresponding version of Pip
