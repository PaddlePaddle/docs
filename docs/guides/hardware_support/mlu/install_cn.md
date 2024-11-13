# 寒武纪 MLU 安装说明

飞桨框架 MLU 版支持寒武纪 MLU 的训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译安装得到 wheel 包

## 寒武纪 MLU 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 寒武纪思元 370 系列，包括 MLU370X8、MLU370X4、MLU370S4 |
| 操作系统 | Linux 操作系统，包括 Ubuntu、KylinV10 |

## 运行环境准备

推荐使用飞桨官方发布的寒武纪 MLU 开发镜像，该镜像预装有[寒武纪基础软件开发平台](https://developer.cambricon.com/)。

```bash
# 拉取镜像
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-mlu:ubuntu20-x86_64-gcc84-py310

# 参考如下命令，启动容器
docker run -it --name paddle-mlu-dev -v $(pwd):/work \
  -w=/work --shm-size=128G --network=host --privileged  \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /usr/bin/cnmon:/usr/bin/cnmon \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-mlu:ubuntu20-x86_64-gcc84-py310 /bin/bash

# 检查容器内是否可以正常识别寒武纪 MLU 设备
cnmon

# 预期得到输出如下
+------------------------------------------------------------------------------+
| CNMON v5.10.13                                               Driver v5.10.13 |
+-------------------------------+----------------------+-----------------------+
| Card  VF  Name       Firmware |               Bus-Id | Util        Ecc-Error |
| Fan   Temp      Pwr:Usage/Cap |         Memory-Usage | SR-IOV   Compute-Mode |
|===============================+======================+=======================|
| 0     /   MLU370-X8    v1.1.6 |         0000:39:00.0 | 0%                N/A |
|  0%   32C        106 W/ 250 W |     0 MiB/ 23374 MiB | N/A           Default |
+-------------------------------+----------------------+-----------------------+
| 1     /   MLU370-X8    v1.1.6 |         0000:3D:00.0 | 0%                N/A |
|  0%   39C        106 W/ 250 W |     0 MiB/ 23374 MiB | N/A           Default |
+-------------------------------+----------------------+-----------------------+

+------------------------------------------------------------------------------+
| Processes:                                                                   |
|  Card  MI  PID     Command Line                             MLU Memory Usage |
|==============================================================================|
|  No running processes found                                                  |
+------------------------------------------------------------------------------+
```

## 安装飞桨框架

**注意**：当前飞桨 develop 分支仅支持 X86 架构，暂不支持寒武纪 MLU 的 ARM 架构。

### 安装方式一：wheel 包安装

寒武纪支持插件式安装，需先安装飞桨 CPU 安装包，再安装飞桨 MLU 插件包。在启动的 docker 容器中，执行以下命令：

```bash
# 先安装飞桨 CPU 安装包
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu

# 再安装飞桨 MLU 插件包
pip install paddle-custom-mlu -i https://www.paddlepaddle.org.cn/packages/nightly/mlu
```

### 安装方式二：源代码编译安装

在启动的 docker 容器中，先安装飞桨 CPU 安装包，再下载 PaddleCustomDevice 源码编译得到飞桨 MLU 插件包。

```bash
# 下载 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# 进入硬件后端(寒武纪 MLU)目录
cd PaddleCustomDevice/backends/mlu

# 先安装飞桨 CPU 安装包
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu

# 执行编译脚本 - submodule 在编译时会按需下载
bash tools/compile.sh

# 飞桨 MLU 插件包在 build/dist 路径下，使用 pip 安装即可
pip install build/dist/paddle_custom_mlu*.whl
```

## 基础功能检查

安装完成后，在 docker 容器中输入如下命令进行飞桨基础健康功能的检查。

```bash
# 检查当前安装版本
python -c "import paddle_custom_device; paddle_custom_device.mlu.version()"
# 预期得到如下输出结果
version: 0.0.0
commit: 147d506b2baa1971ab47b4550f0571e1f6b201fc
cntoolkit: 3.8.2
cnnl: 1.23.2
cnnl_extra: 1.6.1
cncl: 1.14.0
mluops: 0.11.0

# 飞桨基础健康检查
python -c "import paddle; paddle.utils.run_check()"
# 预期得到输出如下
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 mlu.
PaddlePaddle works well on 8 mlus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 如何卸载

请使用以下命令卸载：

```bash
pip uninstall paddlepaddle paddle-custom-mlu
```
