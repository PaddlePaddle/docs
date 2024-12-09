# 燧原 GCU 安装说明

飞桨框架 GCU 版支持燧原 GCU 的训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译得到 wheel 包安装

## 燧原 GCU 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 燧原 S60 |
| 操作系统 | Linux 操作系统，如 Ubuntu 等 |

## 运行环境准备

推荐使用飞桨官方发布的燧原 GCU 开发镜像，该镜像预装有[燧原基础软件开发平台（TopsRider）](https://www.enflame-tech.com/developer)。

```bash
# 拉取镜像
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:topsrider3.2.109-ubuntu20-x86_64-gcc84

# 参考如下命令启动容器
docker run --name paddle-gcu-dev -v /home:/home \
    --network=host --ipc=host -it --privileged \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-gcu:topsrider3.2.109-ubuntu20-x86_64-gcc84 /bin/bash

# 容器外安装驱动程序。可以参考飞桨自定义接入硬件后端(GCU)环境准备章节
bash TopsRider_i3x_*_deb_amd64.run --driver --no-auto-load

# 容器外检查是否可以正常识别燧原 GCU 设备
efsmi

# 预期得到类似如下的结果
----------------------------------------------------------------------------
------------------- Enflame System Management Interface --------------------
---------- Enflame Tech, All Rights Reserved. 2024 Copyright (C) -----------
----------------------------------------------------------------------------

+2024-12-06, 11:33:56 CST--------------------------------------------------+
|EFSMI    V1.2.0.301       Driver Ver: 1.2.0.301                           |
|--------------------------------------------------------------------------|
|--------------------------------------------------------------------------|
| DEV    NAME                | FW VER          | BUS-ID      ECC           |
| TEMP   Dpm   Pwr(Usage/Cap)| Mem     GCU Virt| DUsed       SN            |
|--------------------------------------------------------------------------|
| 0      S60                 | 33.6.2          | 00:01:00.0  Disable       |
| 39 C   Sleep    105W  300W | 49120MiB Disable|   0.0%      A073640510015 |
+--------------------------------------------------------------------------+
|--------------------------------------------------------------------------|
| 1      S60                 | 33.6.5          | 00:09:00.0  Enable        |
| 46 C   Active   128W  300W | 42976MiB Disable|   0.0%      C807J40510285 |
+--------------------------------------------------------------------------+
```

## 安装飞桨框架

### 安装方式一：wheel 包安装

燧原支持插件式安装，需先安装飞桨 CPU 安装包，再安装飞桨 GCU 插件包。在启动的 docker 容器中，执行以下命令：

```bash
# 先安装飞桨 CPU 安装包
python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu

# 再安装飞桨 GCU 插件包
python -m pip install paddle-custom-gcu -i https://www.paddlepaddle.org.cn/packages/nightly/gcu
```

### 安装方式二：源代码编译安装

在启动的 docker 容器中，先安装飞桨 CPU 安装包，再下载 PaddleCustomDevice 源码编译得到飞桨 GCU 插件包。

```bash
# 下载 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# 进入硬件后端(燧原 GCU)目录
cd PaddleCustomDevice/backends/gcu

# 先安装飞桨 CPU 安装包
python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu

# 执行编译命令 - submodule 在编译时会按需下载
mkdir -p build && cd build
export PADDLE_CUSTOM_PATH=`python -c "import re, paddle; print(re.compile('/__init__.py.*').sub('',paddle.__file__))"`
cmake .. -DWITH_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPY_VERSION=3.10
make -j $(nproc)

# 飞桨 GCU 插件包在 build/dist 路径下，使用 pip 安装即可
python -m pip install --force-reinstall -U build/dist/paddle_custom_gcu*.whl
```

## 基础功能检查

安装完成后，在 docker 容器中输入如下命令进行飞桨基础健康功能的检查。

```bash
# 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.gcu.version()"
# 预期得到如下输出结果
version: 3.0.0.dev20241206
commit: 7a2766768cc92aa94cc3d0ea6c23e8397f15f68a
TopsPlatform: 1.2.0.301
....

# 飞桨基础健康检查
python -c "import paddle; paddle.utils.run_check()"
# 预期得到输出如下
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 gcu.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 如何卸载

请使用以下命令卸载 Paddle:

```bash
python -m pip uninstall paddlepaddle paddle-custom-gcu
```
