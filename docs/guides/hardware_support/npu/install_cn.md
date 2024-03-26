# 昇腾 NPU 安装说明

飞桨框架 NPU 版支持昇腾 NPU 的训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译安装得到 wheel 包

## 昇腾 NPU 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 昇腾 910 系列，包括 910A、910B |
| 操作系统 | Linux 操作系统，包括 Ubuntu、KylinV10 |

## 运行环境准备

推荐使用飞桨官方发布的昇腾 NPU 开发镜像，该镜像预装有[昇腾基础软件开发平台（CANN）](https://www.hiascend.com/software/cann)。

```bash
# 昇腾 910A 芯片 - 系统环境下查看 lspci | grep d801 是否有输出
docker pull registry.baidubce.com/device/paddle-npu:cann80T2-910A-ubuntu18-x86_64 # X86 架构
docker pull registry.baidubce.com/device/paddle-npu:cann80T2-910A-ubuntu18-aarch64 # ARM 架构
# 昇腾 910B 芯片 - 系统环境下查看 lspci | grep d802 是否有输出
docker pull registry.baidubce.com/device/paddle-npu:cann80T2-910B-ubuntu18-x86_64 # X86 架构
docker pull registry.baidubce.com/device/paddle-npu:cann80T2-910B-ubuntu18-aarch64 # ARM 架构

# 考如下命令启动容器，ASCEND_RT_VISIBLE_DEVICES 可指定可见的 NPU 卡号
docker run -it --name paddle-npu-dev -v $(pwd):/work \
    --privileged --network=host --shm-size=128G -w=/work \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80T2-910B-ubuntu18-x86_64 /bin/bash

# 检查容器内是否可以正常识别昇腾 NPU 设备
npu-smi info

# 预期得到类似如下的结果
+------------------------------------------------------------------------------------------------+
| npu-smi 23.0.rc3                 Version: 23.0.rc3                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B2C              | OK            | 83.4        35                0    / 0             |
| 0                         | 0000:5A:00.0  | 0           0    / 0          4302 / 65536         |
+===========================+===============+====================================================+
| 1     910B2C              | OK            | 83.4        37                0    / 0             |
| 0                         | 0000:29:00.0  | 0           0    / 0          4302 / 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
+===========================+===============+====================================================+
| No running processes found in NPU 1                                                            |
+===========================+===============+====================================================+
```

## 安装飞桨框架

**注意**：当前飞桨仅提供 910B 芯片在 X86 架构上的 wheel 包，如果需要 910A 芯片或 ARM 架构，请参考[源代码编译安装](#安装方式二：源代码编译安装)。

### 安装方式一：wheel 包安装

昇腾支持插件式安装，需先安装飞桨 CPU 安装包，再安装飞桨 NPU 插件包。在启动的 docker 容器中，执行以下命令：

```bash
# 先安装飞桨 CPU 安装包
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

# 再安装飞桨 NPU 插件包
wget https://paddle-device.bj.bcebos.com/0.0.0/npu/paddle_custom_npu-0.0.0-cp310-cp310-linux_x86_64.whl
pip install -U paddle_custom_npu-0.0.0-cp310-cp310-linux_x86_64.whl
```

### 安装方式二：源代码编译安装

在启动的 docker 容器中，先安装飞桨 CPU 安装包，再下载 PaddleCustomDevice 源码编译得到飞桨 NPU 插件包。

```bash
# 下载 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# 进入硬件后端(昇腾 NPU)目录
cd PaddleCustomDevice/backends/npu

# 先安装飞桨 CPU 安装包
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

# 执行编译脚本 - submodule 在编译时会按需下载
bash tools/compile.sh

# 飞桨 NPU 插件包在 build/dist 路径下，使用 pip 安装即可
pip install -U paddle_custom_npu-0.0.0-cp39-cp39-linux_x86_64.whl
```

## 基础功能检查

安装完成后，在 docker 容器中输入如下命令进行飞桨基础健康功能的检查。

```bash
# 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.npu.version()"
# 预期得到如下输出结果
version: 0.0.0
commit: 7163a1e74066d86c2d4ef48e4affe1f9ad07aa95
cann: 8.0.T2
....

# 飞桨基础健康检查
python -c "import paddle; paddle.utils.run_check()"
# 预期得到输出如下
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 npu.
PaddlePaddle works well on 8 npus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 如何卸载

请使用以下命令卸载 Paddle:

```bash
pip uninstall paddlepaddle-npu
```
