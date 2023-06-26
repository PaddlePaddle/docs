# **macOS 下使用ninja从源码编译**

## 环境准备

* **macOS 版本 10.x/11.x (64 bit) (不支持 GPU 版本)**
* **Python 版本 3.7/3.8/3.9/3.10 (64 bit)**

## 选择 CPU/GPU

* 目前仅支持在 macOS 环境下编译安装 CPU 版本的 PaddlePaddle

## 安装步骤
在 macOS 系统下有 2 种编译方式，推荐使用 Docker 编译。
Docker 环境中已预装好编译 Paddle 需要的各种依赖，相较本机编译环境更简单。

* [Docker 源码编译](#compile_from_docker)
* [本机源码编译](#compile_from_host)

<a name="mac_docker"></a>
### <span id="compile_from_docker">**使用 Docker 编译**</span>

[Docker](https://docs.docker.com/install/)是一个开源的应用容器引擎。使用 Docker，既可以将 PaddlePaddle 的安装&使用与系统环境隔离，也可以与主机共享 GPU、网络等资源

使用 Docker 编译 PaddlePaddle，您需要：

- 在本地主机上[安装 Docker](https://docs.docker.com/engine/install/)

- 使用 Docker ID 登陆 Docker，以避免出现`Authenticate Failed`错误

请您按照以下步骤安装：

#### 1. 进入 Mac 的终端

#### 2. 请选择您希望储存 PaddlePaddle 的路径，然后在该路径下使用以下命令将 PaddlePaddle 的源码从 github 克隆到本地当前目录下名为 Paddle 的文件夹中：

```
git clone https://github.com/PaddlePaddle/Paddle.git
```

#### 3. 进入 Paddle 目录下：
```
cd Paddle
```

#### 4. 拉取 PaddlePaddle 镜像

对于国内用户，因为网络问题下载 docker 比较慢时，可使用百度提供的镜像：

* CPU 版的 PaddlePaddle：
    ```
    docker pull registry.baidubce.com/paddlepaddle/paddle:latest-dev
    ```

如果您的机器不在中国大陆地区，可以直接从 DockerHub 拉取镜像：

* CPU 版的 PaddlePaddle：
    ```
    docker pull paddlepaddle/paddle:latest-dev
    ```

您可以访问[DockerHub](https://hub.docker.com/r/paddlepaddle/paddle/tags/)获取与您机器适配的镜像。


#### 5. 创建并进入满足编译环境的 Docker 容器：

```
docker run --name paddle-test -v $PWD:/paddle --network=host -it registry.baidubce.com/paddlepaddle/paddle:latest-dev /bin/bash
```

- `--name paddle-test`：为您创建的 Docker 容器命名为 paddle-test

- `-v：$PWD:/paddle`：将当前目录挂载到 Docker 容器中的/paddle 目录下（Linux 中 PWD 变量会展开为当前路径的[绝对路径](https://baike.baidu.com/item/绝对路径/481185)）

- `-it`：与宿主机保持交互状态

- `registry.baidubce.com/paddlepaddle/paddle:latest-dev`：使用名为`registry.baidubce.com/paddlepaddle/paddle:latest-dev`的镜像创建 Docker 容器，/bin/bash 进入容器后启动/bin/bash 命令

注意：
请确保至少为 docker 分配 4g 以上的内存，否则编译过程可能因内存不足导致失败。您可以在 docker 用户界面的“Preferences-Resources”中设置容器的内存分配上限。

#### 6. 进入 Docker 后进入 paddle 目录下：

```
cd /paddle
```

#### 7. 切换到 develop 版本进行编译：

```
git checkout develop
```

注意：python3.6、python3.7 版本从 release/1.2 分支开始支持, python3.8 版本从 release/1.8 分支开始支持, python3.9 版本从 release/2.1 分支开始支持, python3.10 版本从 release/2.3 分支开始支持

#### 8. 创建并进入/paddle/build 路径下：

```
mkdir -p /paddle/build && cd /paddle/build
```

#### 9. 使用以下命令安装相关依赖：

- 安装 protobuf 3.1.0和ninja。

```
pip3.7 install protobuf==3.1.0 ninja
```

注意：以上用 Python3.7 命令来举例，如您的 Python 版本为 3.6/3.8/3.9，请将上述命令中的 pip3.7 改成 pip3.6/pip3.8/pip3.9

- 安装 patchelf，PatchELF 是一个小而实用的程序，用于修改 ELF 可执行文件的动态链接器和 RPATH。

```
apt install patchelf
```

#### 10. 执行 cmake：

*  对于需要编译**CPU 版本 PaddlePaddle**的用户（我们目前不支持 macOS 下 GPU 版本 PaddlePaddle 的编译）：

    ```
    cmake .. -GNinja -DPY_VERSION=3.7 -DWITH_GPU=OFF
    ```
- 具体编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

- 请注意修改参数`-DPY_VERSION`为您希望编译使用的 python 版本,  例如`-DPY_VERSION=3.7`表示 python 版本为 3.7

#### 11. 执行编译：

使用多核编译

```
ninja -j$(nproc)
```

注意：
编译过程中需要从 github 上下载依赖，请确保您的编译环境能正常从 github 下载代码。

#### 12. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包：
```
cd /paddle/build/python/dist
```

#### 13. 在当前机器或目标机器安装编译好的`.whl`包：

```
pip3.7 install -U [whl 包的名字]
```

注意：
以上用 Python3.7 命令来举例，如您的 Python 版本为 3.6/3.8/3.9，请将上述命令中的 pip3.7 改成 pip3.6/pip3.8/pip3.9。

#### 恭喜，至此您已完成 PaddlePaddle 的编译安装。您只需要进入 Docker 容器后运行 PaddlePaddle，即可开始使用。更多 Docker 使用请参见[Docker 官方文档](https://docs.docker.com)


<a name="mac_source"></a>
<br/><br/>
### <span id="compile_from_host">**本机编译**</span>

**请严格按照以下指令顺序执行**

#### 1. 检查您的计算机和操作系统是否符合我们支持的编译标准：
```
uname -m
```
并且在`关于本机`中查看系统版本。并提前安装[OpenCV](https://opencv.org/releases.html)

#### 2. 安装 Python 以及 pip：

> **请不要使用 macOS 中自带 Python**，我们强烈建议您使用[Homebrew](https://brew.sh)安装 python(对于**Python3**请使用 python[官方下载](https://www.python.org/downloads/mac-osx/)python3.7.x、python3.8、python3.9、python3.10), pip 以及其他的依赖，这将会使您高效编译。

使用 Python 官网安装

> 请注意，当您的 mac 上安装有多个 python 时请保证您正在使用的 python 是您希望使用的 python。


#### 3. (Only For Python3)设置 Python 相关的环境变量：

- a. 首先使用
    ```
    find `dirname $(dirname $(which python3))` -name "libpython3.*.dylib"
    ```
    找到 Pythonlib 的路径（弹出的第一个对应您需要使用的 python 的 dylib 路径），然后（下面[python-lib-path]替换为找到文件路径）

- b. 设置 PYTHON_LIBRARIES：
    ```
    export PYTHON_LIBRARY=[python-lib-path]
    ```

- c. 其次使用找到 PythonInclude 的路径（通常是找到[python-lib-path]的上一级目录为同级目录的 include,然后找到该目录下 python3.x 的路径），然后（下面[python-include-path]替换为找到路径）
- d. 设置 PYTHON_INCLUDE_DIR:
    ```
    export PYTHON_INCLUDE_DIRS=[python-include-path]
    ```

- e. 设置系统环境变量路径：
    ```
    export PATH=[python-bin-path]:$PATH
    ```
    （这里[python-bin-path]为将[python-lib-path]的最后两级目录替换为/bin/后的目录)

- f. 设置动态库链接：
    ```
    export LD_LIBRARY_PATH=[python-ld-path]
    ```
    以及
    ```
    export DYLD_LIBRARY_PATH=[python-ld-path]
    ```
    （这里[python-ld-path]为[python-bin-path]的上一级目录)

- g. (可选）如果您是在 macOS 10.14 上编译 PaddlePaddle，请保证您已经安装了[对应版本](http://developer.apple.com/download)的 Xcode。

#### 4. **执行编译前**请您确认您的环境中安装有[编译依赖表](/documentation/docs/zh/install/Tables.html#third_party)中提到的相关依赖，否则我们强烈推荐使用`Homebrew`安装相关依赖。

> macOS 下如果您未自行修改或安装过“编译依赖表”中提到的依赖，则仅需要使用`pip`安装`numpy，protobuf，wheel ninja`，使用`Homebrew`安装`wget，swig, unrar`，另外安装`cmake`即可

- a. 这里特别说明一下**CMake**的安装：

    CMake 我们支持 3.15 以上版本,推荐使用 CMake3.16,请根据以下步骤安装：

    1. 从 CMake[官方网站](https://cmake.org/files/v3.16/cmake-3.16.0-Darwin-x86_64.dmg)下载 CMake 镜像并安装
    2. 在控制台输入
        ```
        sudo "/Applications/CMake.app/Contents/bin/cmake-gui" –install
        ```

- b. 如果您不想使用系统默认的 blas 而希望使用自己安装的 OPENBLAS 请参见[FAQ](../FAQ.html/#OPENBLAS)

#### 5. 将 PaddlePaddle 的源码 clone 在当下目录下的 Paddle 的文件夹中，并进入 Padde 目录下：

```
git clone https://github.com/PaddlePaddle/Paddle.git
```

```
cd Paddle
```

#### 6. 切换到 develop 分支进行编译：

```
git checkout develop
```

注意：python3.7 版本从 release/1.2 分支开始支持, python3.8 版本从 release/1.8 分支开始支持, python3.9 版本从 release/2.1 分支开始支持, python3.10 版本从 release/2.3 分支开始支持

#### 7. 并且请创建并进入一个叫 build 的目录下：

```
mkdir build && cd build
```

#### 8. 执行 cmake：

>具体编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

*  对于需要编译**CPU 版本 PaddlePaddle**的用户：

    ```
    cmake .. -GNinja -DPY_VERSION=3.7 -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIRS} \
    -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DWITH_GPU=OFF
    ```

>`-DPY_VERSION=3.7`请修改为安装环境的 Python 版本

#### 9. 使用以下命令来编译：

```
ninja -j$(sysctl -n hw.ncpu)
```

#### 10. 编译成功后进入`/paddle/build/python/dist`目录下找到生成的`.whl`包：
```
cd /paddle/build/python/dist
```

#### 11. 在当前机器或目标机器安装编译好的`.whl`包：

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
请使用以下命令卸载 PaddlePaddle

* **CPU 版本的 PaddlePaddle**:
    ```
    pip uninstall paddlepaddle
    ```
    或
    ```
    pip3 uninstall paddlepaddle
    ```

使用 Docker 安装 PaddlePaddle 的用户，请进入包含 PaddlePaddle 的容器中使用上述命令，注意使用对应版本的 pip
