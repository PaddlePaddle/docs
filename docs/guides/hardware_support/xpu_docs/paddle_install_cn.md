# 飞桨框架昆仑 XPU 版安装说明

飞桨框架支持基于 python 的训练和原生预测，当前最新版本为 2.1，提供两种安装方式：

**1. 预编译的支持昆仑 XPU 的 wheel 包**

目前此 wheel 包只支持两种环境：

英特尔 CPU+昆仑 XPU+CentOS 系统

飞腾 CPU+昆仑 XPU+麒麟 V10 系统

**2. 源码编译安装**

其他环境请选择源码编译安装。

## 安装方式一：通过 Wheel 包安装

### 下载安装包

**环境 1：英特尔 CPU+昆仑 XPU+CentOS 系统**

Linux 发行版建议选择 CentOS 7 系统

Python3.7

```
wget https://paddle-wheel.bj.bcebos.com/kunlun/paddlepaddle-2.1.0-cp37-cp37m-linux_x86_64.whl
```

```
python3.7 -m pip install -U paddlepaddle-2.1.0-cp37-cp37m-linux_x86_64.whl
```

Python3.6

```
wget https://paddle-wheel.bj.bcebos.com/kunlun/paddlepaddle-2.1.0-cp36-cp36m-linux_x86_64.whl
```

```
python3.6 -m pip install -U ``paddlepaddle-2.1.0-cp36-cp36m-linux_x86_64.whl
```

**环境 2：飞腾 CPU+昆仑 XPU+麒麟 V10 系统**

如果您想使用预编译的支持昆仑 XPU 的 wheel 包，请联系飞桨官方邮件组：Paddle-better@baidu.com

### 验证安装

安装完成后您可以使用 python 或 python3 进入 python 解释器，输入

```
import paddle
```

再输入

```
paddle.utils.run_check()
```

如果出现 PaddlePaddle is installed successfully!，说明您已成功安装。

## 安装方式二：从源码编译支持昆仑 XPU 的包

### 环境准备

**英特尔 CPU+昆仑 XPU+CentOS 系统**

- **处理器：Intel(R) Xeon(R) Gold 6148 CPU @2.40GHz**
- **操作系统：CentOS 7.8.2003（建议使用 CentOS 7）**
- **Python 版本： 3.6/3.7 (64 bit)**
- **pip 或 pip3 版本：9.0.1+ (64 bit)**
- **cmake 版本：3.15+**
- **gcc/g++版本：8.2+**

**飞腾 CPU+昆仑 XPU+麒麟 V10 系统**

- **处理器：Phytium,FT-2000+/64**
- **操作系统：Kylin release V10 (SP1)/(Tercel)-aarch64-Build04/20200711**
- **Python 版本：3.6/3.7 (64 bit)**
- **pip 或 pip3 版本： 9.0.1+ (64 bit)**
- **cmake 版本：3.15+**
- **gcc/g++版本：8.2+**

### 源码编译安装步骤：


（1）Paddle 依赖 cmake 进行编译构建，需要 cmake 版本>=3.15，如果操作系统提供的源包括了合适版本的 cmake，直接安装即可，否则需要

```
wget https://github.com/Kitware/CMake/releases/download/v3.16.8/cmake-3.16.8.tar.gz
tar -xzf cmake-3.16.8.tar.gz && cd cmake-3.16.8
./bootstrap && make && sudo make install
```

（2）Paddle 内部使用 patchelf 来修改动态库的 rpath，如果操作系统提供的源包括了 patchelf，直接安装即可，否则需要源码安装，请参考

```
./bootstrap.sh
./configure
make
make check
sudo make install
```

（3）根据[requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装 Python 依赖库

（4）将 Paddle 的源代码克隆到当下目录下的 Paddle 文件夹中，并进入 Paddle 目录

```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
```

使用较稳定的版本编译，建议切换到 release2.1 分支下：

```
git checkout release/2.1
```

（5）进行 Wheel 包的编译，请创建并进入一个叫 build 的目录下

```
mkdir build && cd build
```

具体编译选项含义可参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

**英特尔 CPU+昆仑 XPU+CentOS 系统**

链接过程中打开文件数较多，可能超过系统默认限制导致编译出错，设置进程允许打开的最大文件数：

```
ulimit -n 2048
```

执行 cmake，完成编译

Python3

```
cmake .. -DPY_VERSION=3.6 \
         -DCMAKE_BUILD_TYPE=Release \
         -DWITH_GPU=OFF \
         -DWITH_XPU=ON \
         -DON_INFER=ON \
         -DWITH_PYTHON=ON \
         -DWITH_AVX=ON \
         -DWITH_MKL=ON \
         -DWITH_MKLDNN=ON \
         -DWITH_XPU_BKCL=ON \
         -DWITH_DISTRIBUTE=ON \
         -DWITH_NCCL=OFF

make -j20
```


**飞腾 CPU+昆仑 XPU+麒麟 V10 系统**

在该环境下，编译前需要手动拉取 XPU SDK，可使用以下命令：

```
wget https://paddle-wheel.bj.bcebos.com/kunlun/xpu_sdk_v2.0.0.61.tar.gz
tar xvf xpu_sdk_v2.0.0.61.tar.gz
mv output xpu_sdk_v2.0.0.61 xpu_sdk
```

执行 cmake，完成编译

```
ulimit -n 4096
python_exe="/usr/bin/python3.7"
export XPU_SDK_ROOT=$PWD/xpu_sdk

cmake .. -DPY_VERSION=3.7 \
         -DPYTHON_EXECUTABLE=$python_exe \
         -DWITH_ARM=ON \
         -DWITH_AARCH64=ON \
         -DWITH_TESTING=OFF \
         -DCMAKE_BUILD_TYPE=Release \
         -DON_INFER=ON \
         -DWITH_XBYAK=OFF \
         -DWITH_XPU=ON \
         -DWITH_GPU=OFF \
         -DWITH_LITE=ON \
         -DLITE_GIT_TAG=release/v2.9 \
         -DXPU_SDK_ROOT=${XPU_SDK_ROOT}

make VERBOSE=1 TARGET=ARMV8 -j32
```

（6）编译成功后进入 Paddle/build/python/dist 目录下找到生成的.whl 包 。

（7）将生成的.whl 包 copy 至带有昆仑 XPU 的目标机器上，并在目标机器上根据[requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装 Python 依赖库。（如果编译机器同时为带有昆仑 XPU 的目标机器，略过此步）

（8）在带有昆仑 XPU 的目标机器安装编译好的.whl 包：pip install -U（whl 包的名字）或 pip3 install -U（whl 包的名字）。恭喜，至此您已完成昆仑 XPU 机器上 PaddlePaddle 的编译安装。

**验证安装**

安装完成后您可以使用 python 或 python3 进入 python 解释器，输入

```
import paddle
```

再输入

```
paddle.utils.run_check()
```

如果出现 PaddlePaddle is installed successfully!，说明您已成功安装。

### 如何卸载

使用以下命令卸载 PaddlePaddle：

```
pip uninstall paddlepaddle
```

或

```
pip3 uninstall paddlepaddle
```
