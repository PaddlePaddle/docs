# **Compile on MacOS from Source Code**

## Environment preparation

* **MacOS version 10.11/10.12/10.13/10.14 (64 bit) (not support GPU version)**
* **Python version 2.7.15+/3.5.1+/3.6/3.7/3.8 (64 bit)**
* **pip or pip3 version 20.2.2+ (64 bit)**

## Choose CPU/GPU

* Currently, only PaddlePaddle for CPU is supported.

## Installation steps
There are two compilation methods in MacOS system:

* [Compile with Docker](#compile_from_docker)
* [Local compilation](#compile_from_host)

<a name="mac_docker"></a>
### <span id="compile_from_docker">**Compile with Docker**</span>

[Docker](https://docs.docker.com/install/) is an open source application container engine. Using docker, you can not only isolate the installation and use of paddlepaddle from the system environment, but also share GPU, network and other resources with the host

Compiling PaddlePaddle with Docker，you need:

- On the local host [Install Docker](https://hub.docker.com/search/?type=edition&offering=community)

- Log in to Docker with Docker ID to avoid `Authenticate Failed` error

Please follow the steps below to install:

1. Enter the terminal of the Mac

2. Please select the path where you want to store PaddlePaddle, and then use the following command to clone PaddlePaddle's source code from github to a folder named Paddle in the local current directory:

    ```
    git clone https://github.com/PaddlePaddle/Paddle.git
    ```

3. Go to the Paddle directory:
    ```
    cd Paddle
    ```

4. Create and enter a Docker container that meets the compilation environment:

    ```
    docker run --name paddle-test -v $PWD:/paddle --network=host -it hub.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
    ```

    > --name paddle-test name the Docker container you created as paddle-test,

    > -v $PWD:/paddle mount the current directory to the /paddle directory in the Docker container (the PWD variable in Linux will expand to the current path's [Absolute path](https://baike.baidu.com/item/绝对路径/481185)),

    > -it keeps interacting with the host, `hub.baidubce.com/paddlepaddle/paddle:latest-dev` creates a Docker container with a mirror named `hub.baidubce.com/paddlepaddle/paddle:latest-dev`, /bin /bash starts the /bin/bash command after entering the container.

5. After entering Docker, go to the paddle directory:

    ```
    cd paddle
    ```

6. Switch to `develop` version to compile:

    ```
    git checkout develop
    ```

    Note: python3.6、python3.7 version started supporting from release/1.2 branch, python3.8 version started supporting from release/1.8 branch

7. Create and enter the /paddle/build path:

    ```
    mkdir -p /paddle/build && cd /paddle/build
    ```

8. Use the following command to install the dependencies:

    For Python2:
    ```
    pip install protobuf==3.1.0
    ```
    For Python3:
    ```
    pip3.5 install protobuf==3.1.0
    ```

    Note: We used Python3.5 command as an example above, if the version of your Python is 3.6/3.7/3.8, please change Python3.5 in the commands to Python3.6/Python3.7/Python3.8

    > Install protobuf 3.1.0.

    ```
    apt install patchelf
    ```

    > Installing patchelf, PatchELF is a small and useful program for modifying the dynamic linker and RPATH of ELF executables.

9. Execute cmake:

    > For details on the compilation options, see the [compilation options table](../Tables_en.html/#Compile).
    > Please attention to modify parameters `-DPY_VERSION` for the version of Python you want to compile with, for example `-DPY_VERSION=3.5` means the version of python is 3.5.x

    * For users who need to compile the **CPU version PaddlePaddle**:

        ```
        cmake .. -DPY_VERSION=3.5 -DWITH_GPU=OFF -DWITH_TESTING=OFF -DWITH_AVX=OFF -DCMAKE_BUILD_TYPE=Release
        ```

        > We currently do not support the compilation of the GPU version PaddlePaddle under CentOS.

10. Execute compilation:

    ```
    make -j$(nproc)
    ```

    > Use multicore compilation

11. After compiling successfully, go to the `/paddle/build/python/dist `directory and find the generated `.whl` package:
    ```
    cd /paddle/build/python/dist
    ```

12. Install the compiled `.whl` package on the current machine or target machine: (For Python3: Please select the pip corresponding to the python version you wish to use, such as pip3.5, pip3.6)


    For Python2:
    ```
    pip install -U (whl package name)
    ```
    For Python3:
    ```
    pip3.5 install -U (whl package name)
    ```

    Note: We used Python3.5 command as an example above, if the version of your Python is 3.6/3.7/3.8, please change Python3.5 in the commands to Python3.6/Python3.7/Python3.8

Congratulations, now that you have successfully installed PaddlePaddle using Docker, you only need to run PaddlePaddle after entering the Docker container. For more Docker usage, please refer to the [official Docker documentation](https://docs.docker.com/).

> Note: In order to reduce the size, `vim` is not installed in PaddlePaddle Docker image by default. You can edit the code in the container after executing `apt-get install -y vim` in the container.

<a name="mac_source"></a>
<br/><br/>
### <span id="compile_from_host">**Local compilation**</span>

**Please strictly follow the order of the following instructions**

1. Check that your computer and operating system meet our supported compilation standards: `uname -m` and view the system version `about this Mac`. And install [OpenCV](https://opencv.org/releases.html) in advance.

2. Install python and pip:

    > **Please do not use the Python initially given by MacOS**, we strongly recommend that you use [Homebrew](https://brew.sh/) to install python (for Python3 please use python [official download](https://www.python.org/downloads/mac-osx/) python3.5.x, python3.6.x, python3.7.x, python3.8), pip and other dependencies, This will greatly reduce the difficulty of installing and compiling.

    For python2:
    ```
    brew install python@2
    ```
    For python3: Install using Python official website


    > Please note that when you have multiple pythons installed on your mac, make sure that the python you are using is the python you wish to use.

3. (Only For Python2) Set Python-related environment variables:

    - Use
        ```
        find / -name libpython2.7.dylib
        ```
        to find your current python `libpython2.7.dylib` path and use
        ```
        export LD_LIBRARY_PATH=[libpython2.7.dylib path] && export DYLD_LIBRARY_PATH=[libpython2.7.dylib  to the top two directories of the directory]
        ```

4. (Only For Python3) Set Python-related environment variables:

    - a. First use
        ```
        find `dirname $(dirname $(which python3))` -name "libpython3.*.dylib"
        ```
        to find the path to Pythonlib (the first one it prompts is the dylib path for the python you need to use), then (below [python-lib-path] is replaced by finding the file path)

    - b. Set PYTHON_LIBRARIES:
        ```
        export PYTHON_LIBRARY=[python-lib-path]
        ```

    - c. Secondly use the path to find PythonInclude (usually find the above directory of [python-lib-path] as the include of the same directory, then find the path of python3.x or python2.x in the directory), then (the [python-include-path] in the following commands should be replaced by the path found here)

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

    - g. (Optional) If you are compiling PaddlePaddle on MacOS 10.14, make sure you have the [appropriate version](http://developer.apple.com/download) of Xcode installed.


5. Before **compilation**, please confirm that the relevant dependencies mentioned in the [compilation dependency table](h../Tables.html/#third_party) are installed in your environment, otherwise we strongly recommend using `Homebrew` to install related dependencies.

    > Under MacOS, if you have not modified or installed the dependencies mentioned in the "Compile Dependency Table", you only need to use `pip` to install `numpy`, `protobuf`, `wheel`, use `homebrew` to install `wget`, `swig`,then install `cmake`.

    - a. Here is a special description of the installation of **CMake**:

        We support CMake version 3.15 and above, CMake 3.16 is recommended, please follow the steps below to install:

        1. Download the CMake image from the [official CMake website](https://cmake.org/files/v3.16/cmake-3.16.0-Darwin-x86_64.dmg) and install it.

        2. Enter
            ```
            sudo "/Applications/CMake.app/Contents/bin/cmake-gui" –install
            ```
            in the console

    - b. If you do not want to use the system default blas and want to use your own installed OPENBLAS please read [FAQ](../FAQ.html/#OPENBLAS)

6. Put the PaddlePaddle source cloned in the Paddle folder in the current directory and go to the Paddle directory:

    ```
    git clone https://github.com/PaddlePaddle/Paddle.git
    ```

    ```
    cd Paddle
    ```

7. Switch to `develop` branch to compile: (Note that python 3.6, python 3.7 version are supported from the 1.2 branch, python3.8 version started supporting from release/1.8 branch)

    ```
    git checkout develop
    ```

    Note: python3.6、python3.7 version started supporting from release/1.2 branch, python3.8 version started supporting from release/1.8 branch

8. And please create and enter a directory called build:

    ```
    mkdir build && cd build
    ```

9. Execute cmake:

    > For details on the compilation options, see the [compilation options table](https://www.paddlepaddle.org.cn/documentation/docs/en/develop/install/Tables.html#Compile).

    * For users who need to compile the **CPU version PaddlePaddle**:


        For Python2:
        ```
        cmake .. -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
        ```
        For Python3:
        ```
        cmake .. -DPY_VERSION=3.5 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
        -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_FLUID_ONLY=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
        ```

    > ``-DPY_VERSION=3.5`` Please change to the Python version of the installation environment.

10. Compile with the following command:

    ```
    make -j4
    ```

11. After compiling successfully, go to the `/paddle/build/python/dist `directory and find the generated `.whl` package:
    ```
    cd /paddle/build/python/dist
    ```

12. Install the compiled `.whl` package on the current machine or target machine:

    ```
    pip install -U (whl package name)
    ```
    or
    ```
    pip3 install -U (whl package name)
    ```

    > If you have multiple python environments and pips installed on your computer, please see the [FAQ](../Tables.html/#MACPRO).

Congratulations, now you have completed the process of compiling PaddlePaddle using this machine.

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
