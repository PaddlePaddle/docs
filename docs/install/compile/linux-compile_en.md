# **Compile on Linux from Source Code**

## Environment preparation

* **Linux version (64 bit)**
    * **CentOS 6 (not recommended, no official support for compilation problems)**
    * **CentOS 7 (GPU version supports CUDA 10.1/10.2/11.0/11.1/11.2**
    * **Ubuntu 14.04 (not recommended, no official support for compilation problems)**
    * **Ubuntu 16.04 (GPU version supports CUDA 10.1/10.2/11.0/11.1/11.2)**
    * **Ubuntu 18.04 (GPU version supports CUDA 10.1/10.2/11.0/11.1/11.2)**
* **Python version 3.7/3.8/3.9/3.10 (64 bit)**

## Choose CPU/GPU

* If your computer doesn't have NVIDIA® GPU, please install CPU version of PaddlePaddle

* If your computer has NVIDIA® GPU, and the following conditions are met，GPU version of PaddlePaddle is recommended.

    * **CUDA toolkit 10.1/10.2 with cuDNN 7 (cuDNN version>=7.6.5, for multi card support, NCCL2.7 or higher)**
    * **CUDA toolkit 11.0 with cuDNN v8.0.4(for multi card support, NCCL2.7 or higher)**
    * **CUDA toolkit 11.1 with cuDNN v8.1.1(for multi card support, NCCL2.7 or higher)**
    * **CUDA toolkit 11.2 with cuDNN v8.1.1(for multi card support, NCCL2.7 or higher)**
    * **Hardware devices with GPU computing power over 3.5**

        You can refer to NVIDIA official documents for installation process and configuration method of CUDA and cudnn. Please refer to[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)


## Installation steps

There are two compilation methods under Linux system. It's recommended to use Docker to compile.
The dependencies required for compiling Paddle are pre-installed in the Docker environment, which is simpler than the native compiling environment.

* [Compile with Docker](#compile_from_docker) (no official support for compilation problems under CentOS 6)
* [Local compilation](#compile_from_host) (no official support for compilation problems under CentOS 6)

<a name="ct_docker"></a>
### <span id="compile_from_docker">**Compile with Docker**</span>

[Docker](https://docs.docker.com/install/) is an open source application container engine. Using docker, you can not only isolate the installation and use of paddlepaddle from the system environment, but also share GPU, network and other resources with the host

Compiling PaddlePaddle with Docker，you need:

- On the local host [Install Docker](https://docs.docker.com/engine/install/)

- To enable GPU support on Linux, please [Install nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

Please follow the steps below to install:

#### 1. First select the path where you want to store PaddlePaddle, then use the following command to clone PaddlePaddle's source code from github to a folder named Paddle in the local current directory:

```
git clone https://github.com/PaddlePaddle/Paddle.git
```

#### 2. Go to the Paddle directory:
```
cd Paddle
```

#### 3. Pull PaddlePaddle image:

For domestic users, when downloading docker is slow due to network problems, you can use the mirror provided by Baidu:

* CPU version of PaddlePaddle：
    ```
    docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev
    ```

* GPU version of PaddlePaddle：
    ```
    docker pull registry.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7-dev
    ```

If your machine is not in mainland China, you can pull the image directly from DockerHub:

* CPU version of PaddlePaddle：
    ```
    docker pull paddlepaddle/paddle:latest-dev
    ```

* GPU version of PaddlePaddle：
    ```
    docker pull paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7-dev
    ```

In the above example, `latest-gpu-cuda10.2-cudnn7-dev` is only for illustration, indicating that the GPU version of the image is installed. If you want to install another `cuda/cudnn` version of the image, you can replace it with `latest-dev-cuda11.2-cudnn8-gcc82`, `latest-gpu-cuda10.1-cudnn7-gcc82-dev`, `latest-gpu-cuda10.1-cudnn7-gcc54-dev` etc.

You can see [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) to get the image that matches your machine.



#### 4. Create and enter a Docker container that meets the compilation environment:

* Compile CPU version of PaddlePaddle：

    ```
    docker run --name paddle-test -v $PWD:/paddle --network=host -it registry.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
    ```

    - `--name paddle-test`: names the Docker container you created as paddle-test;


    - `-v $PWD:/paddle`: mount the current directory to the /paddle directory in the docker container (PWD variable in Linux will be expanded to [absolute path](https://baike.baidu.com/item/绝对路径/481185) of the current path);


    - `-it`: keeps interaction with the host;

    - `registry.baidubce.com/paddlepaddle/paddle:latest-dev`: use the image named `registry.baidubce.com/paddlepaddle/paddle:latest-dev` to create Docker container, /bin/bash start the /bin/bash command after entering the container.



* Compile GPU version of PaddlePaddle:

    ```
    nvidia-docker run --name paddle-test -v $PWD:/paddle --network=host -it registry.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7-dev /bin/bash
    ```

    - `--name paddle-test`: names the Docker container you created as paddle-test;


    - `-v $PWD:/paddle`: mount the current directory to the /paddle directory in the docker container (PWD variable in Linux will be expanded to [absolute path](https://baike.baidu.com/item/绝对路径/481185) of the current path);



    - `-it`: keeps interaction with the host;

    - `registry.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7-dev`: use the image named `registry.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7-dev` to create Docker container, /bin/bash start the /bin/bash command after entering the container.


Note:
Please make sure to allocate at least 4g of memory for docker, otherwise the compilation process may fail due to insufficient memory.

#### 5. After entering Docker, go to the paddle directory:
```
cd /paddle
```

#### 6. Switch to develop version to compile:

```
git checkout develop
```

Note: python3.6、python3.7 version started supporting from release/1.2 branch, python3.8 version started supporting from release/1.8 branch, python3.9 version started supporting from release/2.1 branch, python3.10 version started supporting from release/2.3 branch

#### 7. Create and enter the /paddle/build path:

```
mkdir -p /paddle/build && cd /paddle/build
```

#### 8. Use the following command to install the dependencies:

- Install protobuf 3.1.0

```
pip3.7 install protobuf
```

Note: We used Python3.7 command as an example above, if the version of your Python is 3.6/3.8/3.9, please change pip3.7 in the commands to pip3.6/pip3.8/pip3.9

- Installing patchelf, PatchELF is a small and useful program for modifying the dynamic linker and RPATH of ELF executables.

```
apt install patchelf
```

#### 9. Execute cmake:

* For users who need to compile the **CPU version PaddlePaddle**:

    ```
    cmake .. -DPY_VERSION=3.7 -DWITH_GPU=OFF
    ```

* For users who need to compile the **GPU version PaddlePaddle**:
    ```
    cmake .. -DPY_VERSION=3.7 -DWITH_GPU=ON
    ```

- For details on the compilation options, see the [compilation options table](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/Tables.html#Compile).

- Please attention to modify parameters `-DPY_VERSION` for the version of Python you want to compile with, for example `-DPY_VERSION=3.7` means the version of python is 3.7

- We currently do not support the compilation of the GPU version PaddlePaddle under CentOS 6.

#### 10. Execute compilation:

Use multicore compilation

```
make -j$(nproc)
```

Note:
During the compilation process, you need to download dependencies from github. Please make sure that your compilation environment can download the code from github normally.

#### 11. After compiling successfully, go to the `/paddle/build/python/dist` directory and find the generated `.whl` package:
```
cd /paddle/build/python/dist
```

#### 12. Install the compiled `.whl` package on the current machine or target machine:

For Python3:
```
pip3.7 install -U [whl package name]
```

Note:
We used Python3.7 command as an example above, if the version of your Python is 3.6/3.8/3.9, please change pip3.7 in the commands to pip3.6/pip3.8/pip3.9.

#### Congratulations, now that you have successfully installed PaddlePaddle using Docker, you only need to run PaddlePaddle after entering the Docker container. For more Docker usage, please refer to the [official Docker documentation](https://docs.docker.com/).


<a name="ct_source"></a>
### <span id="compile_from_host">**Local compilation**</span>

#### 1. Check that your computer and operating system meet the compilation standards we support:
```
uname -m && cat /etc/*release
```

#### 2. Update the system source

* CentOS system

    Update the source of `yum`: `yum update`, and add the necessary yum source:
    ```
    yum install -y epel-release
    ```

* Ubuntu system

    Update the source of `apt`:
    ```
    apt update
    ```

#### 3. Install NCCL (optional)

* If you need to use multi card environment, please make sure that you have installed nccl2 correctly, or install nccl2 according to the following instructions (here is the installation instructions of nccl2 under CUDA10.2 and cuDNN7. For more version of installation information, please refer to NVIDIA[official website](https://developer.nvidia.com/nccl)):


    * **CentOS system can refer to the following commands**

        ```
        wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        ```
        ```
        rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        ```
        ```
        yum update -y
        ```
        ```
        yum install -y libnccl-2.7.8-1+cuda10.2 libnccl-devel-2.7.8-1+cuda10.2 libnccl-static-2.7.8-1+cuda10.2
        ```

    * **Ubuntu system can refer to the following commands**

        ```
        wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
        ```
        ```
        dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
        ```
        ```
        sudo apt install -y libnccl2=2.7.8-1+cuda10.2 libnccl-dev=2.7.8-1+cuda10.2
        ```

#### 4. Install the necessary tools

* CentOS system

    `bzip2` and `make`:
    ```
    yum install -y bzip2
    ```
    ```
    yum install -y make
    ```

    cmake requires version 3.15, we recommend that you use 3.16.0 version:

    ```
    wget -q https://cmake.org/files/v3.16/cmake-3.16.0-Linux-x86_64.tar.gz
    ```
    ```
    tar -zxvf cmake-3.16.0-Linux-x86_64.tar.gz
    ```
    ```
    rm cmake-3.16.0-Linux-x86_64.tar.gz
    ```
    ```
    PATH=/home/cmake-3.16.0-Linux-x86_64/bin:$PATH
    ```

    gcc requires version 5.4, we recommend that you use 8.2.0 version:

    ```
    wget -q https://paddle-docker-tar.bj.bcebos.com/home/users/tianshuo/bce-python-sdk-0.8.27/gcc-8.2.0.tar.xz && \
    tar -xvf gcc-8.2.0.tar.xz && \
    cd gcc-8.2.0 && \
    sed -i 's#ftp://gcc.gnu.org/pub/gcc/infrastructure/#https://paddle-ci.gz.bcebos.com/#g' ./contrib/download_prerequisites && \
    unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
    ./contrib/download_prerequisites && \
    cd .. && mkdir temp_gcc82 && cd temp_gcc82 && \
    ../gcc-8.2.0/configure --prefix=/usr/local/gcc-8.2 --enable-threads=posix --disable-checking --disable-multilib && \
    make -j8 && make install
    ```

* Ubuntu system

    `bzip2` and `make`:
    ```
    apt install -y bzip2
    ```
    ```
    apt install -y make
    ```

    cmake requires version 3.15, we recommend that you use 3.16.0 version:

    ```
    wget -q https://cmake.org/files/v3.16/cmake-3.16.0-Linux-x86_64.tar.gz
    ```
    ```
    tar -zxvf cmake-3.16.0-Linux-x86_64.tar.gz
    ```
    ```
    rm cmake-3.16.0-Linux-x86_64.tar.gz
    ```
    ```
    PATH=/home/cmake-3.16.0-Linux-x86_64/bin:$PATH
    ```

    gcc requires version 5.4, we recommend that you use 8.2.0 version:

    ```
    wget -q https://paddle-docker-tar.bj.bcebos.com/home/users/tianshuo/bce-python-sdk-0.8.27/gcc-8.2.0.tar.xz && \
    tar -xvf gcc-8.2.0.tar.xz && \
    cd gcc-8.2.0 && \
    sed -i 's#ftp://gcc.gnu.org/pub/gcc/infrastructure/#https://paddle-ci.gz.bcebos.com/#g' ./contrib/download_prerequisites && \
    unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE && \
    ./contrib/download_prerequisites && \
    cd .. && mkdir temp_gcc82 && cd temp_gcc82 && \
    ../gcc-8.2.0/configure --prefix=/usr/local/gcc-8.2 --enable-threads=posix --disable-checking --disable-multilib && \
    make -j8 && make install
    ```

#### 5. We support compiling and installing with virtualenv. First, create a virtual environment called `paddle-venv` with the following command:

* a. Install Python-dev:

    (Please refer to the official Python installation process)


* b. Install pip:


    (Please refer to the official Python installation process, and ensure that the pip3 version 20.2.2 and above, please note that in python3.6 and above, pip3 does not necessarily correspond to the python version, such as python3.7 default only Pip3.7)

* c. (Only For Python3) set Python3 related environment variables, here is python3.7 version example, please replace with the version you use (3.6, 3.8, 3.9):

    1. First find the path to the Python lib using
        ```
        find `dirname $(dirname $(which python3))` -name "libpython3.so"
        ```
        If it is 3.7/3.8/3.9/3.10, change `python3` to `python3.7`, `python3.8`, `python3.9`, `python3.10`, then replace [python-lib-path] in the following steps with the file path found.

    2. Set PYTHON_LIBRARIES:
        ```
        export PYTHON_LIBRARY=[python-lib-path]
        ```

    3. Secondly, use
        ```
        find `dirname $(dirname $(which python3))`/include -name "python3.7m"
        ```
        to find the path to Python Include, please pay attention to the python version, then replace the following [python-include-path] to the file path found.

    4. Set PYTHON_INCLUDE_DIR:
        ```
        export PYTHON_INCLUDE_DIRS=[python-include-path]
        ```

    5. Set the system environment variable path:
        ```
        export PATH=[python-lib-path]:$PATH
        ```
        (here replace the last two levels content of [python-lib-path] with /bin/)

* d. Install the virtual environment `virtualenv` and `virtualenvwrapper` and create a virtual environment called `paddle-venv`: (please note the pip3 commands corresponding to the python version, such as pip3.6, pip3.7, pip3.8, pip3.9)

    1. Install `virtualenv`:
        ```
        pip install virtualenv
        ```
        or
        ```
        pip3 install virtualenv
        ```

    2. Install `virtualenvwrapper`
        ```
        Pip install virtualenvwrapper
        ```
        or
        ```
        pip3 install virtualenvwrapper
        ```

    3. Find `virtualenvwrapper.sh`:
        ```
        find / -name virtualenvwrapper.sh
        ```
        (please find the corresponding Python version of `virtualenvwrapper.sh`)

    4. See the installation method in `virtualenvwrapper.sh`:
        ```
        cat vitualenvwrapper.sh
        ```
        this shell file describes the steps and commands

    5. Install `virtualwrapper` as described in `virtualenvwrapper.sh`

    6. Set VIRTUALENVWRAPPER_PYTHON：
        ```
        export VIRTUALENVWRAPPER_PYTHON=[python-lib-path]:$PATH
        ```
        (here replace the last two levels content of [python-lib-path] with /bin/)
    7. Create virtual environment named `paddle-venv`:
        ```
        mkvirtualenv paddle-venv
        ```

#### 6. Enter the virtual environment:
```
workon paddle-venv
```

#### 7. Before **executing the compilation**, please confirm that the related dependencies mentioned in the [compile dependency table](/documentation/docs/en/install/Tables_en.html/#third_party) are installed in the virtual environment:

* Here is the installation method for `patchELF`. Other dependencies can be installed using `yum install` or `apt install`, `pip install`/`pip3 install` followed by the name and version:

    ```
    yum install patchelf
    ```
    > Users who can't use yum installation can refer to patchElF github [official documentation](https://gist.github.com/ruario/80fefd174b3395d34c14).

#### 8. Put the PaddlePaddle source cloned in the Paddle folder in the current directory and go to the Paddle directory:

```
git clone https://github.com/PaddlePaddle/Paddle.git
```

```
cd Paddle
```

#### 9. Switch to develop branch for compilation (support for Python 3.6 and 3.7 is added from the 1.2 branch, support for Python 3.8 is added from the 1.8 branch, support for Python 3.9 is added from the 2.1 branch,):

```
git checkout develop
```

#### 10. And please create and enter a directory called build:

```
mkdir build && cd build
```

#### 11. Execute cmake:

> For details on the compilation options, see the [compilation options table](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/Tables.html#Compile).

* For users who need to compile the **CPU version PaddlePaddle**:

    ```
    cmake .. -DPY_VERSION=3.7 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
    -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF
    ```


    > If you encounter `Could NOT find PROTOBUF (missing: PROTOBUF_LIBRARY PROTOBUF_INCLUDE_DIR)`, you can re-execute the cmake command.
    > Please note that the PY_VERSION parameter is replaced with the python version you need.


* For users who need to compile the **GPU version PaddlePaddle**:

    1. Please make sure that you have installed nccl2 correctly, or install nccl2 according to the following instructions (here is ubuntu 16.04, CUDA10.2, cuDNN7 nccl2 installation instructions, for more information on the installation information please refer to the [NVIDIA official website](https://developer.nvidia.com/nccl/nccl-download)):

        ```
        wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        ```

        ```
        rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        ```

        ```
        yum install -y libnccl-2.7.8-1+cuda10.2 libnccl-devel-2.7.8-1+cuda10.2 libnccl-static-2.7.8-1+cuda10.2
        ```

    2. If you have already installed `nccl2` correctly, you can start cmake: *(For Python3: Please configure the correct python version for the PY_VERSION parameter)*


        ```
        cmake .. -DPYTHON_EXECUTABLE:FILEPATH=[您可执行的 Python3 的路径] -DPYTHON_INCLUDE_DIR:PATH=[之前的 PYTHON_INCLUDE_DIRS] -DPYTHON_LIBRARY:FILEPATH=[之前的 PYTHON_LIBRARY] -DWITH_GPU=ON
        ```


Note: For the command involving Python 3, we use Python 3.7 as an example above, if the version of your Python is 3.8/3.9, please change Python3.7 in the commands to Python3.8/Python3.9



#### 12. Compile with the following command:

```
make -j$(nproc)
```

> Use multicore compilation

> If “Too many open files” error is displayed during compilation, please use the instruction ulimit -n 8192  to increase the number of files allowed to be opened by the current process. Generally speaking, 8192 can ensure the completion of compilation.

#### 13. After compiling successfully, go to the `/paddle/build/python/dist `directory and find the generated `.whl` package:
```
cd /paddle/build/python/dist
```

#### 14. Install the compiled `.whl` package on the current machine or target machine:

```
Pip install -U (whl package name)
```
or
```
pip3 install -U (whl package name)
```

#### Congratulations, now you have completed the process of compiling PaddlePaddle natively.

<br/><br/>
### ***Verify installation***

After the installation is complete, you can use `python` or `python3` to enter the Python interpreter and then use
```
import paddle
```
and then
```
paddle.utils.run_check()
```
to verify that the installation was successful.

If `PaddlePaddle is installed successfully!` appears, it means the installation was successful.

<br/><br/>
### ***How to uninstall***

Please use the following command to uninstall PaddlePaddle (users who use Docker to install PaddlePaddle should use the following command in the container containing PaddlePaddle. Please use the corresponding version of pip):

* ***CPU version of PaddlePaddle***:
    ```
    pip uninstall paddlepaddle
    ```
    or
    ```
    pip3 uninstall paddlepaddle
    ```
* ***GPU version of PaddlePaddle***:
    ```
    pip uninstall paddlepaddle-gpu
    ```
    or
    ```
    pip3 uninstall paddlepaddle-gpu
    ```

Users installing PaddlePaddle with Docker, please use above commands in the container involved PaddlePaddle and attention to use the corresponding version of Pip
