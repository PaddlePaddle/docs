# **Linux 下使用ninja从源码编译**

## 环境准备

* **Linux 版本 (64 bit)**
    * **CentOS 6 (不推荐，不提供编译出现问题时的官方支持)**
    * **CentOS 7 (GPU 版本支持 CUDA 10.1/10.2/11.0/11.1/11.2)**
    * **Ubuntu 14.04 (不推荐，不提供编译出现问题时的官方支持)**
    * **Ubuntu 16.04 (GPU 版本支持 CUDA 10.1/10.2/11.0/11.1/11.2)**
    * **Ubuntu 18.04 (GPU 版本支持 CUDA 10.1/10.2/11.0/11.1/11.2)**
* **Python 版本 3.7/3.8/3.9/3.10 (64 bit)**

## 选择 CPU/GPU

* 如果您的计算机没有 NVIDIA® GPU，请安装 CPU 版本的 PaddlePaddle

* 如果您的计算机有 NVIDIA® GPU，请确保满足以下条件以编译 GPU 版 PaddlePaddle

    * **CUDA 工具包 10.1/10.2 配合 cuDNN 7 (cuDNN 版本>=7.6.5, 如需多卡支持，需配合 NCCL2.7 及更高)**
    * **CUDA 工具包 11.0 配合 cuDNN v8.0.4(如需多卡支持，需配合 NCCL2.7 及更高)**
    * **CUDA 工具包 11.1 配合 cuDNN v8.1.1(如需多卡支持，需配合 NCCL2.7 及更高)**
    * **CUDA 工具包 11.2 配合 cuDNN v8.1.1(如需多卡支持，需配合 NCCL2.7 及更高)**
    * **GPU 运算能力超过 3.5 的硬件设备**

        您可参考 NVIDIA 官方文档了解 CUDA 和 CUDNN 的安装流程和配置方法，请见[CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)，[cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)


## 安装步骤

在 Linux 的系统下有 2 种编译方式，推荐使用 Docker 编译。
Docker 环境中已预装好编译 Paddle 需要的各种依赖，相较本机编译环境更简单。

* [使用 Docker 编译](#compile_from_docker)（不提供在 CentOS 6 下编译中遇到问题的支持）
* [本机编译](#compile_from_host)（不提供在 CentOS 6 下编译中遇到问题的支持）

<a name="ct_docker"></a>
### <span id="compile_from_docker">使用 Docker 编译</span>

[Docker](https://docs.docker.com/install/)是一个开源的应用容器引擎。使用 Docker，既可以将 PaddlePaddle 的安装&使用与系统环境隔离，也可以与主机共享 GPU、网络等资源

使用 Docker 编译 PaddlePaddle，您需要：

- 在本地主机上[安装 Docker](https://docs.docker.com/engine/install/)

- 如需在 Linux 开启 GPU 支持，请[安装 nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

请您按照以下步骤安装：

#### 1. 请首先选择您希望储存 PaddlePaddle 的路径，然后在该路径下使用以下命令将 PaddlePaddle 的源码从 github 克隆到本地当前目录下名为 Paddle 的文件夹中：

```
git clone https://github.com/PaddlePaddle/Paddle.git
```

#### 2. 进入 Paddle 目录下：

```
cd Paddle
```

#### 3. 拉取 PaddlePaddle 镜像

对于国内用户，因为网络问题下载 docker 比较慢时，可使用百度提供的镜像：

* CPU 版的 PaddlePaddle：
    ```
    docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev
    ```

* GPU 版的 PaddlePaddle：
    ```
    nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.2-cudnn7.6-trt7.0-gcc8.2
    ```

如果您的机器不在中国大陆地区，可以直接从 DockerHub 拉取镜像：

* CPU 版的 PaddlePaddle：
    ```
    docker pull paddlepaddle/paddle:latest-dev
    ```

* GPU 版的 PaddlePaddle：
    ```
    nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.2-cudnn7.6-trt7.0-gcc8.2
    ```

上例中，`latest-dev-cuda10.2-cudnn7.6-trt7.0-gcc8.2` 仅作示意用，表示安装 GPU 版的镜像。如果您还想安装其他 cuda/cudnn 版本的镜像，可以将其替换成`latest-dev-cuda11.2-cudnn8.2-trt8.0-gcc82`等。
您可以访问[DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/)获取与您机器适配的镜像。


#### 4. 创建并进入已配置好编译环境的 Docker 容器：

* 编译 CPU 版本的 PaddlePaddle：

    ```
    docker run --name paddle-test -v $PWD:/paddle --network=host -it registry.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
    ```

    - `--name paddle-test`：为您创建的 Docker 容器命名为 paddle-test;

    - `-v $PWD:/paddle`： 将当前目录挂载到 Docker 容器中的/paddle 目录下（Linux 中 PWD 变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185));

    - `-it`： 与宿主机保持交互状态;

    - `registry.baidubce.com/paddlepaddle/paddle:latest-dev`：使用名为`registry.baidubce.com/paddlepaddle/paddle:latest-dev`的镜像创建 Docker 容器，/bin/bash 进入容器后启动/bin/bash 命令。


* 编译 GPU 版本的 PaddlePaddle：

    ```
    nvidia-docker run --name paddle-test -v $PWD:/paddle --network=host -it registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.2-cudnn7.6-trt7.0-gcc8.2 /bin/bash
    ```

    - `--name paddle-test`：为您创建的 Docker 容器命名为 paddle-test;

    - `-v $PWD:/paddle`： 将当前目录挂载到 Docker 容器中的/paddle 目录下（Linux 中 PWD 变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185));

    - `-it`： 与宿主机保持交互状态;

    - `registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.2-cudnn7.6-trt7.0-gcc8.2`：使用名为`registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda10.2-cudnn7.6-trt7.0-gcc8.2`的镜像创建 Docker 容器，/bin/bash 进入容器后启动/bin/bash 命令。


注意：
请确保至少为 docker 分配 4g 以上的内存，否则编译过程可能因内存不足导致失败。

#### 5. 进入 Docker 后进入 paddle 目录下：

```
cd /paddle
```

#### 6. 切换到 develop 版本进行编译：

```
git checkout develop
```

注意：python3.6、python3.7 版本从 release/1.2 分支开始支持, python3.8 版本从 release/1.8 分支开始支持, python3.9 版本从 release/2.1 分支开始支持, python3.10 版本从 release/2.3 分支开始支持

#### 7. 创建并进入/paddle/build 路径下：

```
mkdir -p /paddle/build && cd /paddle/build
```

#### 8. 使用以下命令安装相关依赖：

- 安装 protobuf和ninja。

```
pip3.7 install protobuf ninja
```

注意：以上用 Python3.7 命令来举例，如您的 Python 版本为 3.6/3.8/3.9，请将上述命令中的 pip3.7 改成 pip3.6/pip3.8/pip3.9

- 安装 patchelf，PatchELF 是一个小而实用的程序，用于修改 ELF 可执行文件的动态链接器和 RPATH。

```
apt install patchelf
```


#### 9. 执行 cmake：

* 对于需要编译**CPU 版本 PaddlePaddle**的用户：
    ```
    cmake .. -GNinja -DPY_VERSION=3.7 -DWITH_GPU=OFF
    ```

* 对于需要编译**GPU 版本 PaddlePaddle**的用户：
    ```
    cmake .. -GNinja -DPY_VERSION=3.7 -DWITH_GPU=ON
    ```
- 具体编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

- 请注意修改参数`-DPY_VERSION`为您希望编译使用的 python 版本,  例如`-DPY_VERSION=3.7`表示 python 版本为 3.7

- 我们目前不支持 CentOS 6 下使用 Docker 编译 GPU 版本的 PaddlePaddle

#### 10. 执行编译：

使用多核编译

```
ninja -j$(nproc)
```

注意：
编译过程中需要从 github 上下载依赖，请确保您的编译环境能正常从 github 下载代码。

#### 11. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包：

```
cd /paddle/build/python/dist
```

#### 12. 在当前机器或目标机器安装编译好的`.whl`包：


For Python3:
```
pip3.7 install -U [whl 包的名字]
```

注意：
以上用 Python3.7 命令来举例，如您的 Python 版本为 3.6/3.8/3.9，请将上述命令中的 pip3.7 改成 pip3.6/pip3.8/pip3.9。

#### 恭喜，至此您已完成 PaddlePaddle 的编译安装。您只需要进入 Docker 容器后运行 PaddlePaddle，即可开始使用。更多 Docker 使用请参见[Docker 官方文档](https://docs.docker.com)


<a name="ct_source"></a>
### <span id="compile_from_host">本机编译</span>

#### 1. 检查您的计算机和操作系统是否符合我们支持的编译标准：

```
uname -m && cat /etc/*release
```

#### 2. 更新系统源

* CentOS 环境

    更新`yum`的源：

    ```
    yum update
    ```

    并添加必要的 yum 源：

    ```
    yum install -y epel-release
    ```

* Ubuntu 环境

    更新`apt`的源：

    ```
    apt update
    ```


#### 3. 安装 NCCL（可选）

* 如果您需要使用 GPU 多卡，请确保您已经正确安装 nccl2，或者按照以下指令安装 nccl2（这里提供的是 CUDA10.2，cuDNN7 下 nccl2 的安装指令，更多版本的安装信息请参考 NVIDIA[官方网站](https://developer.nvidia.com/nccl)）:

    * **CentOS 系统可以参考以下命令**

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

    * **Ubuntu 系统可以参考以下命令**

        ```
        wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
        ```

        ```
        dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
        ```

        ```
        sudo apt install -y libnccl2=2.7.8-1+cuda10.2 libnccl-dev=2.7.8-1+cuda10.2
        ```

#### 4. 安装必要的工具

* CentOS 环境

    `bzip2`以及`make`：

    ```
    yum install -y bzip2
    ```

    ```
    yum install -y make
    ```

    cmake 需要 3.15 以上，建议使用 3.16.0:

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

    gcc 需要 5.4 以上，建议使用 8.2.0:

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

* Ubuntu 环境

    `bzip2`以及`make`：

    ```
    apt install -y bzip2
    ```
    ```
    apt install -y make
    ```

    cmake 需要 3.15 以上，建议使用 3.16.0:

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

    gcc 需要 5.4 以上，建议使用 8.2.0:

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

#### 5. 我们支持使用 virtualenv 进行编译安装，首先请使用以下命令创建一个名为`paddle-venv`的虚环境：

* a. 安装 Python-dev:

    (请参照 Python 官方流程安装）


* b. 安装 pip:

    (请参照 Python 官方流程安装, 并保证拥有 20.2.2 及以上的 pip3 版本，请注意，python3.6 及以上版本环境下，pip3 并不一定对应 python 版本，如 python3.7 下默认只有 pip3.7）

* c.（Only For Python3）设置 Python3 相关的环境变量，这里以 python3.7 版本示例，请替换成您使用的版本（3.8、3.9）：

    1. 首先使用
        ```
        find `dirname $(dirname $(which python3))` -name "libpython3.so"
        ```
        找到 Python lib 的路径，如果是 3.7、3.8、3.9、3.10，请将`python3`改成`python3.7`、`python3.8`、`python3.9`，`python3.10`，然后将下面[python-lib-path]替换为找到文件路径

    2. 设置 PYTHON_LIBRARIES：
        ```
        export PYTHON_LIBRARY=[python-lib-path]
        ```

    3. 其次使用
        ```
        find `dirname $(dirname $(which python3))`/include -name "python3.7m"
        ```
        找到 Python Include 的路径，请注意 python 版本，然后将下面[python-include-path]替换为找到文件路径

    4. 设置 PYTHON_INCLUDE_DIR:
        ```
        export PYTHON_INCLUDE_DIRS=[python-include-path]
        ```

    5. 设置系统环境变量路径：
        ```
        export PATH=[python-lib-path]:$PATH
        ```
        （这里将[python-lib-path]的最后两级目录替换为/bin/)

* d. 安装虚环境`virtualenv`以及`virtualenvwrapper`并创建名为`paddle-venv`的虚环境：(请注意对应 python 版本的 pip3 的命令，如 pip3.6、pip3.7、pip3.8、pip3.9)

    1. 安装`virtualenv`
        ```
        pip install virtualenv
        ```
        或
        ```
        pip3 install virtualenv
        ```
    2. 安装`virtualenvwrapper`
        ```
        pip install virtualenvwrapper
        ```
        或
        ```
        pip3 install virtualenvwrapper
        ```
    3.  找到`virtualenvwrapper.sh`：
        ```
        find / -name virtualenvwrapper.sh
        ```
        （请找到对应 Python 版本的`virtualenvwrapper.sh`）
    4.  查看`virtualenvwrapper.sh`中的安装方法：
        ```
        cat vitualenvwrapper.sh
        ```
        该 shell 文件中描述了步骤及命令
    5.  按照`virtualenvwrapper.sh`中的描述，安装`virtualwrapper`
    6.  设置 VIRTUALENVWRAPPER_PYTHON：
        ```
        export VIRTUALENVWRAPPER_PYTHON=[python-lib-path]:$PATH
        ```
        （这里将[python-lib-path]的最后两级目录替换为/bin/)
    7.  创建名为`paddle-venv`的虚环境：
        ```
        mkvirtualenv paddle-venv
        ```

#### 6. 进入虚环境并安装ninja：

```
workon paddle-venv
```

```
pip install ninja
```
#### 7. **执行编译前**请您确认在虚环境中安装有[编译依赖表](/documentation/docs/zh/install/Tables.html#third_party)中提到的相关依赖：

* 这里特别提供`patchELF`的安装方法，其他的依赖可以使用`yum install`或者`pip install`/`pip3 install` 后跟依赖名称和版本安装:

    ```
    yum install patchelf
    ```
    > 不能使用 yum 安装的用户请参见 patchElF github[官方文档](https://gist.github.com/ruario/80fefd174b3395d34c14)

#### 8. 将 PaddlePaddle 的源码 clone 在当下目录下的 Paddle 的文件夹中，并进入 Padde 目录下：

```
git clone https://github.com/PaddlePaddle/Paddle.git
```

```
cd Paddle
```

#### 9. 切换到 develop 分支进行编译：

```
git checkout develop
```

#### 10. 并且请创建并进入一个叫 build 的目录下：

```
mkdir build && cd build
```

#### 11. 执行 cmake：

>具体编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

*  对于需要编译**CPU 版本 PaddlePaddle**的用户：


    ```
    cmake .. -GNinja -DPY_VERSION=3.7 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
    -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF
    ```

    > 如果遇到`Could NOT find PROTOBUF (missing:  PROTOBUF_LIBRARY PROTOBUF_INCLUDE_DIR)`可以重新执行一次 cmake 指令。
    > 请注意 PY_VERSION 参数更换为您需要的 python 版本


* 对于需要编译**GPU 版本 PaddlePaddle**的用户：(**仅支持 CentOS7（CUDA11.2/CUDA11.0/CUDA10.2/CUDA10.1)**)

    1. 请确保您已经正确安装 nccl2，或者按照以下指令安装 nccl2（这里提供的是 CUDA10.2，cuDNN7 下 nccl2 的安装指令，更多版本的安装信息请参考 NVIDIA[官方网站](https://developer.nvidia.com/nccl)）:

    * **CentOS 系统可以参考以下命令**

        ```
        wget http://developer.download.nvidia.com/compute/machine-learning/repos/rhel7/x86_64/nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        ```

        ```
        rpm -i nvidia-machine-learning-repo-rhel7-1.0.0-1.x86_64.rpm
        ```

        ```
        yum install -y libnccl-2.7.8-1+cuda10.2 libnccl-devel-2.7.8-1+cuda10.2 libnccl-static-2.7.8-1+cuda10.2
        ```

    * **Ubuntu 系统可以参考以下命令**

        ```
        wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
        ```

        ```
        dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
        ```

        ```
        sudo apt install -y libnccl2=2.7.8-1+cuda10.2 libnccl-dev=2.7.8-1+cuda10.2
        ```

    2. 如果您已经正确安装了`nccl2`，就可以开始 cmake 了：(*For Python3: 请给 PY_VERSION 参数配置正确的 python 版本*)

        ```
        cmake .. -GNinja -DPYTHON_EXECUTABLE:FILEPATH=[您可执行的 Python3 的路径] -DPYTHON_INCLUDE_DIR:PATH=[之前的 PYTHON_INCLUDE_DIRS] -DPYTHON_LIBRARY:FILEPATH=[之前的 PYTHON_LIBRARY] -DWITH_GPU=ON
        ```

注意：以上涉及 Python3 的命令，用 Python3.7 来举例，如您的 Python 版本为 3.8/3.9，请将上述命令中的 Python3.7 改成 Python3.8/Python3.9




#### 12. 使用以下命令来编译：

```
ninja -j$(nproc)
```

> 使用多核编译

> 如果编译过程中显示“Too many open files”错误时，请使用指令 ulimit -n 8192 来增大当前进程允许打开的文件数，一般来说 8192 可以保证编译完成。

#### 13. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包：
```
cd /paddle/build/python/dist
```

#### 14. 在当前机器或目标机器安装编译好的`.whl`包：

```
pip install -U（whl 包的名字）
```
或
```
pip3 install -U（whl 包的名字）
```

#### 恭喜，至此您已完成 PaddlePaddle 的编译安装

## **验证安装**
安装完成后您可以使用 `python` 或 `python3` 进入 python 解释器，输入

```
import paddle
```

再输入

```
paddle.utils.run_check()
```

如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

## **如何卸载**
请使用以下命令卸载 PaddlePaddle：

* **CPU 版本的 PaddlePaddle**:
    ```
    pip uninstall paddlepaddle
    ```
    或
    ```
    pip3 uninstall paddlepaddle
    ```

* **GPU 版本的 PaddlePaddle**:
    ```
    pip uninstall paddlepaddle-gpu
    ```
    或
    ```
    pip3 uninstall paddlepaddle-gpu
    ```

使用 Docker 安装 PaddlePaddle 的用户，请进入包含 PaddlePaddle 的容器中使用上述命令，注意使用对应版本的 pip
