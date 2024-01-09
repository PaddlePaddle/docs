# 飞桨框架寒武纪 MLU 版安装说明

飞桨框架支持基于 python 的训练和原生预测，当前最新版本为 2.3，提供两种安装方式：

- 通过预编译的 wheel 包安装
- 通过源代码编译安装

## 前置条件

### 板卡安装

寒武纪 MLU370 系列板卡安装，可以参见 [寒武纪官网板卡安装教程](https://developer.cambricon.com/index/curriculum/details/id/38/classid/7.html)。

### 驱动安装

寒武纪驱动安装，可以参见 [寒武纪官网驱动安装](https://www.cambricon.com/docs/sdk_1.6.0/driver_4.20.12/user_guide_4.20.12/index.html)。

**注意**：建议安装寒武纪驱动版本高于 `v4.20.11`。


## 镜像准备

**注意**：当前仅提供基于 Ubuntu18.04 & CNToolkit3.0 的 docker 镜像环境。

首先需要准备支持寒武纪板卡运行环境的 docker 镜像，可以直接从 Paddle 的官方镜像库拉取预先装有 CNToolkit3.0 的 docker 镜像来准备相应的运行环境。

```bash
# 拉取镜像
docker pull registry.baidubce.com/device/paddle-mlu:cntoolkit3.0.2-cnnl1.13.0

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
           registry.baidubce.com/device/paddle-mlu:cntoolkit3.0.2-cnnl1.13.0 \
           /bin/bash

# 检查容器是否可以正确识别寒武纪 MLU 设备
cnmon

# 预期得到以下结果（如下是一台 3 卡机器的信息）：
Sat Oct  8 11:22:22 2022
+------------------------------------------------------------------------------+
| CNMON v4.20.11                                                               |
+-------------------------------+----------------------+-----------------------+
| Card  VF  Name       Firmware | Inited        Driver | Util        Ecc-Error |
| Fan   Temp      Pwr:Usage/Cap |         Memory-Usage |         vMemory-Usage |
|===============================+======================+=======================|
| 0     /   MLU370-X4    v1.1.6 | On          v4.20.11 | 0%          N/A       |
|  0%   32C         30 W/ 150 W |     0 MiB/ 23308 MiB | 10240 MiB/1048576 MiB |
+-------------------------------+----------------------+-----------------------+
| 1     /   MLU370-X4    v1.1.6 | On          v4.20.11 | 0%          N/A       |
|  0%   33C         25 W/ 150 W |     0 MiB/ 23308 MiB | 10240 MiB/1048576 MiB |
+-------------------------------+----------------------+-----------------------+
| 2     /   MLU370-X4    v1.1.6 | On          v4.20.11 | 0%          N/A       |
|  0%   30C         26 W/ 150 W |     0 MiB/ 23308 MiB | 10240 MiB/1048576 MiB |
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
pip install paddlepaddle-mlu==0.0.0 -f https://paddle-device.bj.bcebos.com/develop/mlu/develop.html
```

**第二步**：验证安装包

安装完成之后，运行如下命令。如果出现 PaddlePaddle is installed successfully!，说明已经安装成功。

```bash
python -c "import paddle; paddle.utils.run_check()"
```

## 安装方式二：通过源码编译安装

**注意**：环境准备参见 [镜像准备](./paddle_install_cn.md#jingxiangzhunbei)

**第一步**：下载 Paddle 源码并编译，CMAKE 编译选项含义请参见 [编译选项表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#Compile)

```bash
# 下载源码，默认 develop 分支
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle

# 创建编译目录
mkdir build && cd build

# 执行 cmake
cmake .. -DPY_VERSION=3.7 -DWITH_MLU=ON -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Release -DWITH_DISTRIBUTE=ON -DWITH_CNCL=ON

# 使用以下命令来编译
make -j$(nproc)
```

**第二步**：安装与验证编译生成的 wheel 包

编译完成之后进入`Paddle/build/python/dist`目录即可找到编译生成的.whl 安装包，安装与验证命令如下：

```bash
# 安装命令
python -m pip install -U paddlepaddle_mlu-0.0.0-cp37-cp37-linux_x86_64.whl

# 验证命令
python -c "import paddle; paddle.utils.run_check()"
```

## 如何卸载

请使用以下命令卸载 Paddle:

```bash
pip uninstall paddlepaddle-mlu
```
