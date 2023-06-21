# 飞桨框架寒武纪 MLU 版安装说明

飞桨框架支持基于 python 的训练和原生预测，当前最新版本为 2.5，提供两种安装方式：

- 通过预编译的 wheel 包安装
- 通过源代码编译安装

## 前置条件

### 板卡安装

寒武纪 MLU370 系列板卡安装，可以参见 [寒武纪官网板卡安装教程](https://developer.cambricon.com/index/curriculum/details/id/38/classid/7.html)。

### 驱动安装

寒武纪驱动安装，可以参见 [寒武纪官网驱动安装](https://www.cambricon.com/docs/sdk_1.12.0/driver_5.10.10/user_guide_5.10.10/index.html)。

**注意**：建议安装寒武纪驱动版本高于 `v.5.10.10`。


## 镜像准备

**注意**：当前仅提供基于 Ubuntu18.04 & CNToolkit3.4.2 的 docker 镜像环境。

首先需要准备支持寒武纪板卡运行环境的 docker 镜像，可以直接从 Paddle 的官方镜像库拉取预先装有 CNToolkit3.4.2 的 docker 镜像来准备相应的运行环境。

```bash
# 拉取镜像
docker pull registry.baidubce.com/device/paddle-mlu:cntoolkit3.4.2-1-cnnl1.17.0-1-gcc82

# 启动容器，注意这里的参数，例如 shm-size, device 等都需要配置
# 可以通过 `-v` 参数来挂载训练所需的数据集目录，例如 -v /datasets:/datasets
docker run --shm-size=128G \
           --net=host \
           --cap-add=sys_ptrace \
           -v /usr/bin/cnmon:/usr/bin/cnmon \
           -v `pwd`:/workspace
           -it --privileged \
           --name paddle_mlu_$USER \
           -w=/workspace
           registry.baidubce.com/device/paddle-mlu:cntoolkit3.4.2-1-cnnl1.17.0-1-gcc82 \
           /bin/bash

# 检查容器是否可以正确识别寒武纪 MLU 设备
cnmon

# 预期得到以下结果（如下是一台 2 卡机器的信息）：
Wed Jun 21 10:27:01 2023
+------------------------------------------------------------------------------+
| CNMON v5.10.10                                               Driver v5.10.10 |
+-------------------------------+----------------------+-----------------------+
| Card  VF  Name       Firmware |               Bus-Id | Util        Ecc-Error |
| Fan   Temp      Pwr:Usage/Cap |         Memory-Usage | SR-IOV   Compute-Mode |
|===============================+======================+=======================|
| 0     /   MLU370-X4    v1.1.6 |         0000:1B:00.0 | 0%                N/A |
|  0%   41C         27 W/ 150 W |     0 MiB/ 23308 MiB | N/A         Exclusive |
+-------------------------------+----------------------+-----------------------+
| 1     /   MLU370-X4    v1.1.6 |         0000:36:00.0 | 0%               N/A  |
|  0%   41C         27 W/ 150 W |     0 MiB/ 23308 MiB | N/A         Exclusive |
+-------------------------------+----------------------+-----------------------+

+------------------------------------------------------------------------------+
| Processes:                                                                   |
|  Card  VF  PID    Command Line                             MLU Memory Usage  |
|==============================================================================|
|  No running processes found                                                  |
+------------------------------------------------------------------------------+
```

## 安装方式一：通过 wheel 包安装

**注意**：当前仅提供 Python 3.7 的 wheel 安装包。

**第一步**：下载 Python3.7 wheel 安装包

```bash
# 安装 Paddle CPU 版本
pip install paddlepaddle==2.5.0 -f https://paddle-device.bj.bcebos.com/2.5.0/cpu/paddlepaddle-2.5.0-cp37-cp37m-linux_x86_64.whl

# 安装 PaddleCustmDevice MLU 版本
pip install paddle_custom_mlu-0.0.0-cp37-cp37m-linux_x86_64.whl
```

**第二步**：验证安装包

安装完成之后，运行如下命令。如果出现以下结果，说明已经安装成功。

```bash
# 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 期待输出以下结果
['mlu']

# 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.mlu.version()"
# 预期得到如下输出结果
version: 0.0.0
commit: 40eb1098af5a1cb52f4a72cae4c1cabb12fbd802
```

## 安装方式二：通过源码编译安装

**注意**：环境准备参见 [镜像准备](./paddle_install_cn.md#镜像准备)

**第一步**：下载 Paddle 源码并编译，CMAKE 编译选项含义请参见 [编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

```bash
# 下载源码
git clone --recursive https://github.com/PaddlePaddle/PaddleCustomDevice -b release/2.5

# 进入硬件后端(寒武纪 MLU)目录
cd PaddleCustomDevice/backends/mlu

# 创建编译目录并编译
mkdir build && cd build

# X86_64 环境编译
cmake ..
make -j8

# Aarch64 环境编译
cmake .. -DWITH_ARM=ON
make TARGET=ARMV8 -j8
```

**第二步**：安装与验证编译生成的 wheel 包

编译完成之后进入 `PaddleCustomDevice/backends/mlu/build/dist` 目录即可找到编译生成的.whl 安装包，安装与验证命令如下：

```bash
# 安装命令
python -m pip install -U paddlepaddle_mlu-0.0.0-cp37-cp37-linux_x86_64.whl

# 列出可用硬件后端
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# 期待输出以下结果
['mlu']

# 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.mlu.version()"
# 预期得到如下输出结果
version: 0.0.0
commit: 40eb1098af5a1cb52f4a72cae4c1cabb12fbd802
```

## 如何卸载

请使用以下命令卸载 Paddle:

```bash
pip uninstall paddlepaddle
pip uninstall paddle-custom-mlu
```
