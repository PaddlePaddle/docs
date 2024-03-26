# **Compile on macOS from Source Code**

## Environment preparation

* **macOS version 10.x/11.x/12.x/13.x/14.x (64 bit) (not support GPU version)**
* **Python version 3.8/3.9/3.10/3.11/3.12 (64 bit)**

## Choose CPU/GPU

* Currently, only PaddlePaddle for CPU is supported.

## Installation steps
There are two compilation methods in macOS system. It's recommended to use Docker to compile.
The dependencies required for compiling Paddle are pre-installed in the Docker environment, which is simpler than the native compiling environment.

* [Compile with Docker](#compile_from_docker)
* [Local compilation](#compile_from_host)

<a name="mac_docker"></a>
### <span id="compile_from_docker">**Compile with Docker**</span>

[Docker](https://docs.docker.com/install/) is an open source application container engine. Using docker, you can not only isolate the installation and use of paddlepaddle from the system environment, but also share GPU, network and other resources with the host

Compiling PaddlePaddle with Docker，you need:

- On the local host [Install Docker](https://docs.docker.com/engine/install/)

- Log in to Docker with Docker ID to avoid `Authenticate Failed` error

Please follow the steps below to install:

#### 1. Enter the terminal of the Mac

#### 2. Please select the path where you want to store PaddlePaddle, and then use the following command to clone PaddlePaddle's source code from github to a folder named Paddle in the local current directory:

```
git clone https://github.com/PaddlePaddle/Paddle.git
```

#### 3. Go to the Paddle directory:
```
cd Paddle
```


#### 4. Pull PaddlePaddle image

For domestic users, when downloading docker is slow due to network problems, you can use the mirror provided by Baidu:

* CPU version of PaddlePaddle：
```
docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev
```

If your machine is not in mainland China, you can pull the image directly from DockerHub:

* CPU version of PaddlePaddle：
```
docker pull paddlepaddle/paddle:latest-dev
```

You can see [DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/) to get the image that matches your machine.


#### 5. Create and enter a Docker container that meets the compilation environment:

```
docker run --name paddle-test -v $PWD:/paddle --network=host -it registry.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
```

- `--name paddle-test`: name the Docker container you created as paddle-test,

- `-v $PWD:/paddle`: mount the current directory to the /paddle directory in the Docker container (the PWD variable in Linux will expand to the current path's [Absolute path](https://baike.baidu.com/item/绝对路径/481185)),

- `-it`: keeps interacting with the host;

- `registry.baidubce.com/paddlepaddle/paddle:latest-dev`: creates a Docker container with a mirror named `registry.baidubce.com/paddlepaddle/paddle:latest-dev`, /bin /bash starts the /bin/bash command after entering the container.


Note:
Please make sure to allocate at least 4g of memory for docker, otherwise the compilation process may fail due to insufficient memory. You can set a container's memory allocation cap in "Preferences-Resources" in the docker UI.


#### 6. After entering Docker, go to the paddle directory:

```
cd /paddle
```

#### 7. Switch to develop version to compile:

```
git checkout develop
```

Paddle supports Python version 3.8 and above

#### 8. Create and enter the /paddle/build path:

```
mkdir -p /paddle/build && cd /paddle/build
```

#### 9. Use the following command to install the dependencies:

> Install protobuf 3.20.2.

```
pip3.10 install protobuf==3.20.2
```

Note: We used Python3.10 command as an example above, if the version of your Python is 3.8/3.9/3.11/3.12, please change pip3.10 in the commands to pip3.8/pip3.9/3.11/3.12

> Installing patchelf, PatchELF is a small and useful program for modifying the dynamic linker and RPATH of ELF executables.

```
apt install patchelf
```

#### 10. Execute cmake:

* For users who need to compile the **CPU version PaddlePaddle** (We currently do not support the compilation of the GPU version PaddlePaddle under macOS):

    ```
    cmake .. -DPY_VERSION=3.10 -DWITH_GPU=OFF
    ```
> For details on the compilation options, see the [compilation options table](/documentation/docs/en/install/Tables_en.html/#Compile).

> Please attention to modify parameters `-DPY_VERSION` for the version of Python you want to compile with, for example `-DPY_VERSION=3.10` means the version of python is 3.10

#### 11. Execute compilation:

> Use multicore compilation

```
make -j$(nproc)
```

Note:
During the compilation process, you need to download dependencies from github. Please make sure that your compilation environment can download the code from github normally.

#### 12. After compiling successfully, go to the `/paddle/build/python/dist `directory and find the generated `.whl` package:
```
cd /paddle/build/python/dist
```

#### 13. Install the compiled `.whl` package on the current machine or target machine: (For Python3: Please select the pip corresponding to the python version you wish to use, such as pip3.10)


For Python3:
```
pip3.10 install -U [whl package name]
```

Note:
We used Python3.10 command as an example above, if the version of your Python is 3.8/3.9/3.11/3.12, please change pip3.10 in the commands to pip3.8/pip3.9/pip3.11/pip3.12.

#### Congratulations, now that you have successfully installed PaddlePaddle using Docker, you only need to run PaddlePaddle after entering the Docker container. For more Docker usage, please refer to the [official Docker documentation](https://docs.docker.com/).


<a name="mac_source"></a>
<br/><br/>
### <span id="compile_from_host">**Local compilation**</span>

**Please strictly follow the order of the following instructions**

#### 1. Check that your computer and operating system meet our supported compilation standards: `uname -m` and view the system version `about this Mac`. And install [OpenCV](https://opencv.org/releases.html) in advance.

#### 2. Install python and pip:

> **Please do not use the Python initially given by macOS**, we strongly recommend that you use [Homebrew](https://brew.sh/) to install python (for Python3 please use python [official download](https://www.python.org/downloads/mac-osx/) python3.8, python3.9, python3.10, python3.11, python3.12), pip and other dependencies, This will greatly reduce the difficulty of installing and compiling.

Install using Python official website


> Please note that when you have multiple pythons installed on your mac, make sure that the python you are using is the python you wish to use.


#### 3. (Only For Python3) Set Python-related environment variables:

- a. First use
    ```
    find `dirname $(dirname $(which python3))` -name "libpython3.*.dylib"
    ```
    to find the path to Pythonlib (the first one it prompts is the dylib path for the python you need to use), then (below [python-lib-path] is replaced by finding the file path)

- b. Set PYTHON_LIBRARIES:
    ```
    export PYTHON_LIBRARY=[python-lib-path]
    ```

- c. Secondly use the path to find PythonInclude (usually find the above directory of [python-lib-path] as the include of the same directory, then find the path of python3.x in the directory), then (the [python-include-path] in the following commands should be replaced by the path found here)

- d. Set PYTHON_INCLUDE_DIR:
    ```
    export PYTHON_INCLUDE_DIRS=[python-include-path]
    ```

- e. Set the system environment variable path:
    ```
    export PATH=[python-bin-path]:$PATH
    ```
    (here [python-bin-path] is the result of replacing the last two levels of [python-lib-path] with the path after /bin/)

- f. Set the dynamic library link:
    ```
    export LD_LIBRARY_PATH=[python-ld-path]
    ```
    and
    ```
    export DYLD_LIBRARY_PATH=[python-ld-path]
    ```
    (here [python-ld-path] is the [python-bin-path]'s parent directory )

- g. (Optional) If you are compiling PaddlePaddle on macOS 10.14, make sure you have the [appropriate version](http://developer.apple.com/download) of Xcode installed.


#### 4. Before **compilation**, please confirm that the relevant dependencies mentioned in the [compilation dependency table](/documentation/docs/en/install/Tables_en.html/#third_party) are installed in your environment, otherwise we strongly recommend using `Homebrew` to install related dependencies.

> Under macOS, if you have not modified or installed the dependencies mentioned in the "Compile Dependency Table", you only need to use `pip` to install `numpy`, `protobuf`, `wheel`, use `Homebrew` to install `wget`, `swig`,then install `cmake`.

- a. Here is a special description of the installation of **CMake**:

    We support CMake version 3.18 and above, CMake 3.18 is recommended, please follow the steps below to install:

    1. Download the CMake image from the [official CMake website](https://cmake.org/files/v3.18/cmake-3.18.0-Darwin-x86_64.dmg) and install it.

    2. Enter
        ```
        sudo "/Applications/CMake.app/Contents/bin/cmake-gui" –install
        ```
        in the console

- b. If you do not want to use the system default blas and want to use your own installed OPENBLAS please read [FAQ](../FAQ.html/#OPENBLAS)

#### 5. Put the PaddlePaddle source cloned in the Paddle folder in the current directory and go to the Paddle directory:

```
git clone https://github.com/PaddlePaddle/Paddle.git
```

```
cd Paddle
```

#### 6. Switch to develop branch to compile: (Paddle supports Python version 3.8 and above)

```
git checkout develop
```

#### 7. And please create and enter a directory called build:

```
mkdir build && cd build
```

#### 8. Execute cmake:

> For details on the compilation options, see the [compilation options table](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/Tables.html#Compile).

* For users who need to compile the **CPU version PaddlePaddle**:

    ```
    cmake .. -DPY_VERSION=3.10 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
    -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF
    ```

- ``-DPY_VERSION=3.10`` Please change to the Python version of the installation environment.

#### 9. Compile with the following command:

```
make -j$(sysctl -n hw.ncpu)
```

#### 10. After compiling successfully, go to the `/paddle/build/python/dist `directory and find the generated `.whl` package:
```
cd /paddle/build/python/dist
```

#### 11. Install the compiled `.whl` package on the current machine or target machine:

```
pip install -U (whl package name)
```
or
```
pip3 install -U (whl package name)
```


#### Congratulations, now you have completed the process of compiling PaddlePaddle using this machine.

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

Users installing PaddlePaddle with Docker, please use above commands in the container involved PaddlePaddle and attention to use the corresponding version of Pip
