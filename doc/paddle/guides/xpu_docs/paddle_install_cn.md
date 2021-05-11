# 飞桨框架昆仑XPU版安装说明

飞桨框架支持基于python的训练和原生预测，当前最新版本为2.1，提供两种安装方式：

**1. 预编译的支持昆仑XPU的wheel包**

目前此wheel包只支持两种环境：

英特尔CPU+昆仑XPU+Ubuntu系统

飞腾CPU+昆仑XPU+麒麟V10系统

**2. 源码编译安装**

其他环境请选择源码编译安装。

## 安装方式一：通过Wheel包安装

### 下载安装包

**环境1：英特尔CPU+昆仑XPU+CentOS系统**

Linux发行版建议选择CentOS 7系统

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

**环境2：飞腾CPU+昆仑XPU+麒麟V10系统**

如需该环境下的wheel包，请邮件联系Paddle-better@baidu.com获取。

###验证安装
安装完成后您可以使用 python 或 python3 进入python解释器，输入

```
import paddle
```

再输入

```
paddle.utils.run_check()
```

如果出现PaddlePaddle is installed successfully!，说明您已成功安装。

## 安装方式二：从源码编译支持昆仑XPU的包

### 环境准备

**英特尔CPU+昆仑XPU+Ubuntu系统**

- **处理器：Intel(R) Xeon(R) Gold 6148 CPU @2.40GHz**
- **操作系统：Ubuntu 16.04.6 LTS**
- **Python版本： 2.7/3.6/3.7 (64 bit)**
- **pip或pip3版本：9.0.1+ (64 bit)**
- **cmake版本：3.15+**
- **gcc/g++版本：8.2+**

**飞腾CPU+昆仑XPU+麒麟V10系统**

- **处理器：Phytium,FT-2000+/64**
- **操作系统：Kylin release V10 (SP1)/(Tercel)-aarch64-Build04/20200711**
- **Python版本：3.6/3.7 (64 bit)**
- **pip或pip3版本： 9.0.1+ (64 bit)**
- **cmake版本：3.15+**
- **gcc/g++版本：8.2+**

### 源码编译安装步骤：

（1）Paddle依赖cmake进行编译构建，需要cmake版本>=3.15，如果操作系统提供的源包括了合适版本的cmake，直接安装即可，否则需要

```
wget https://github.com/Kitware/CMake/releases/download/v3.16.8/cmake-3.16.8.tar.gz
tar -xzf cmake-3.16.8.tar.gz && cd cmake-3.16.8
./bootstrap && make && sudo make install
```

（2）Paddle内部使用patchelf来修改动态库的rpath，如果操作系统提供的源包括了patchelf，直接安装即可，否则需要源码安装，请参考

```
./bootstrap.sh
./configure
make
make check
sudo make install
```

（3）根据[requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装Python依赖库

（4）将Paddle的源代码克隆到当下目录下的Paddle文件夹中，并进入Paddle目录

```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
```

（5）建议切换到release2.1分支下进行编译：

```
git checkout [``分支名``]
```

例如：

```
git checkout release/2.1
```

（6）并且请创建并进入一个叫build的目录下

```
mkdir build && cd build
```

（7）链接过程中打开文件数较多，可能超过系统默认限制导致编译出错，设置进程允许打开的最大文件数：

```
ulimit -n 4096
```

（8）执行cmake

（9）具体编译选项含义请参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

**英特尔CPU+昆仑XPU+Ubuntu系统**


Python3

```
cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_MKL=OFF -DWITH_XPU=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_XPU_BKCL=ON
```

Python2

```
cmake .. -DPY_VERSION=2 -DPYTHON_EXECUTABLE=`which python2` -DWITH_MKL=OFF -DWITH_XPU=ON -DWITH_GPU=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_XPU_BKCL=ON
```

**飞腾CPU+昆仑XPU+麒麟V10系统**

Python3

```
cmake .. -DPY_VERSION=3 -DPYTHON_EXECUTABLE=`which python3` -DWITH_ARM=ON -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_XBYAK=OFF -DWITH_XPU=ON -DWITH_GPU=OFF -DWITH_LITE=ON -DLITE_GIT_TAG=develop -DWITH_AARCH64=ON
```

（10）使用以下命令来编译

```
make -j$(nproc)
```

（11）编译成功后进入Paddle/build/python/dist目录下找到生成的.whl包 。

（12）将生成的.whl包copy至带有昆仑XPU的目标机器上，并在目标机器上根据[requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装Python依赖库。（如果编译机器同时为带有昆仑XPU的目标机器，略过此步）

（13）在带有昆仑XPU的目标机器安装编译好的.whl包：pip install -U（whl包的名字）或pip3 install -U（whl包的名字）。恭喜，至此您已完成昆仑XPU机器上PaddlePaddle的编译安装。

**验证安装**

安装完成后您可以使用 python 或 python3 进入python解释器，输入

```
import paddle
```

再输入

```
paddle.utils.run_check()
```

如果出现PaddlePaddle is installed successfully!，说明您已成功安装。

### 如何卸载

使用以下命令卸载PaddlePaddle：

```
pip uninstall paddlepaddle
```

或

```
pip3 uninstall paddlepaddle
```



```
pip3 uninstall paddlepaddle
```
