# 飞桨框架昆仑2代芯片安装说明

飞桨框架支持基于python的训练和原生预测，当前最新版本为2.3rc，提供两种安装方式：

**1. 预编译的支持昆仑2代芯片的wheel包**

目前此wheel包只支持一种环境：

英特尔CPU+昆仑2代芯片+Linux操作系统

**2. 源码编译安装**

其他环境请选择源码编译安装。

## 安装方式一：通过Wheel包安装

### 下载安装包

**环境1：英特尔CPU+昆仑2代芯片+Linux操作系统**

Linux发行版建议选择CentOS 7系统

Python3.7

```
wget https://paddle-inference-lib.bj.bcebos.com/2.3.0-rc0/python/Linux/XPU2/x86-64_gcc8.2_py36_avx_mkl/paddlepaddle_xpu-2.3.0rc0-cp37-cp37m-linux_x86_64.whl
```

```
python3.7 -m pip install -U paddlepaddle_xpu-2.3.0rc0-cp37-cp37m-linux_x86_64.whl
```


### 验证安装

安装完成后您可以使用 python 或 python3 进入python解释器，输入

```
import paddle
```

再输入

```
paddle.utils.run_check()
```

如果出现PaddlePaddle is installed successfully!，说明您已成功安装。

* 注：支持基于Kermel Primitive算子的昆仑2代芯片编译whl包，[点击这里查看](https://www.kunlunxin.com.cn)。

## 安装方式二：从源码编译支持昆仑XPU的包

### 环境准备

**英特尔CPU+昆仑2代芯片+CentOS系统**

- **处理器：Intel(R) Xeon(R) Gold 6148 CPU @2.40GHz**
- **操作系统：CentOS 7.8.2003（建议使用CentOS 7）**
- **Python版本： 3.7 (64 bit)**
- **pip或pip3版本：9.0.1+ (64 bit)**
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

使用较稳定的版本编译，建议切换到release2.3分支下：

```
git checkout release/2.3
```

（5）进行Wheel包的编译，请创建并进入一个叫build的目录下

```
mkdir build && cd build
```

具体编译选项含义可参见[编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

**英特尔CPU+昆仑2代芯+CentOS系统**

链接过程中打开文件数较多，可能超过系统默认限制导致编译出错，设置进程允许打开的最大文件数：

```
ulimit -n 4096
```

执行cmake，完成编译

Python3.7

```
cmake .. -DPY_VERSION=3.7 \
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

make -j$(nproc)
```

（6）编译成功后进入Paddle/build/python/dist目录下找到生成的.whl包 。

（7）将生成的.whl包copy至带有昆仑XPU的目标机器上，并在目标机器上根据[requirments.txt](https://github.com/PaddlePaddle/Paddle/blob/develop/python/requirements.txt)安装Python依赖库。（如果编译机器同时为带有昆仑XPU的目标机器，略过此步）

（8）在带有昆仑XPU的目标机器安装编译好的.whl包：pip install -U（whl包的名字）或pip3 install -U（whl包的名字）。恭喜，至此您已完成昆仑XPU机器上PaddlePaddle的编译安装。

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

* 注：支持基于Kermel Primitive算子的昆仑2代芯片源码编译，[点击这里查看](https://www.kunlunxin.com.cn)。
