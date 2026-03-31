# 天数 GPGPU 安装说明

飞桨框架 **Iluvatar** 版支持在天数 GPGPU 上进行训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译得到 wheel 包安装

## 天数 GPGPU 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 天数智芯 系列芯片，包括 BI150/BI150s |
| 操作系统 | Linux 操作系统，包括 CentOS、Ubuntu、KylinV10 等 |

## 运行环境准备

推荐使用天数官方发布的天数 IXUCA 开发镜像，该镜像预装有天数 IXUCA 基础运行环境库。

```bash
# 拉取镜像
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:3.3.0
```

在 **宿主机（host）** 上请先按下述命令安装与当前硬件匹配的驱动，完成后再启动容器。
```bash
wget https://ai-rank.bj.bcebos.com/Iluvatar/corex-driver-linux64-4.3.8-x86_64.run
bash corex-driver-linux64-4.3.8-x86_64.run
```

**重要：请确认 KMD 版本符合要求。** 可在宿主机执行：

```bash
modinfo iluvatar | grep description
# 示例（以实际输出为准）：
# description:    Iluvatar Big Island for PCI Express: a66854d130483853556e1a2c3d623cb78bcbab34
```

```bash
# 启动容器
docker run -itd --name paddle-ixuca-dev --network host -v /usr/src:/usr/src -v /lib/modules:/lib/modules -v /dev:/dev -v /usr/local/corex/bin/ixsmi:/usr/local/corex/bin/ixsmi -v /usr/local/corex/lib64/libcuda.so.1:/usr/local/corex/lib64/libcuda.so.1 -v /usr/local/corex/lib64/libixml.so:/usr/local/corex/lib64/libixml.so -v /usr/local/corex/lib64/libixthunk.so:/usr/local/corex/lib64/libixthunk.so --privileged --cap-add=ALL --pid=host ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:3.3.0
docker exec -it paddle-ixuca-dev bash
```

⚠️ **注意：** 若镜像内自带的 SDK 与宿主机 KMD 不兼容，可能导致 Paddle 无法识别 Iluvatar 设备。此时需将宿主机上与当前 **corex** 版本匹配的 `ixsmi`、`libcuda.so.1`、`libixml.so`、`libixthunk.so` 等按上例挂载进容器（路径以本机 `corex` 安装目录为准）。

### 选项说明及可调整参数

#### ① `--name paddle-ixuca-dev`

- **作用**：指定容器名称。
- **可调整**：可改为其他名称（例如 `paddle-ixuca-test`），便于区分不同实验环境。

```bash
# 检查容器内是否正常识别天数 GPGPU 设备
ixsmi
```

预期输出示例（设备数量与型号以实际环境为准）：

```text
+-----------------------------------------------------------------------------+
|  IX-ML: 4.3.8       Driver Version: 4.3.8       CUDA Version: 10.2          |
|-------------------------------+----------------------+----------------------|
| GPU  Name                     | Bus-Id               | Clock-SM  Clock-Mem  |
| Fan  Temp  Perf  Pwr:Usage/Cap|      Memory-Usage    | GPU-Util  Compute M. |
|===============================+======================+======================|
| 0    Iluvatar BI-V150         | 00000000:10:00.0     | 1500MHz   1600MHz    |
| N/A  40C   P0    N/A / N/A    | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+
| 1    Iluvatar BI-V150         | 00000000:13:00.0     | 1500MHz   1600MHz    |
| N/A  39C   P0    104W / 350W  | 64MiB / 32768MiB     | 0%        Default    |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU        PID      Process name                                Usage(MiB) |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## 安装飞桨框架

### 安装方式一：wheel 包安装

**iluvatar_gpu** 后端支持通过 wheel 包安装。在已启动的 Docker 容器中执行：

nightly 版：
```bash
python -m pip install --pre paddlepaddle-iluvatar -i https://www.paddlepaddle.org.cn/packages/nightly/ixuca/
```

### 安装方式二：源代码编译安装

在容器中克隆 **Paddle-iluvatar** 源码并编译安装：

```bash
# 下载 Paddle-iluvatar 源码
git clone --recurse-submodules https://github.com/PaddlePaddle/Paddle-iluvatar.git
cd Paddle-iluvatar

# 执行编译与安装脚本
bash build_paddle.sh
bash install_paddle.sh
```

## 基础功能检查

安装完成后，在容器内执行下列命令做基础健康检查。

```bash
# 列出可用自定义硬件类型
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
```

预期输出示例：

```text
['iluvatar_gpu']
```

```bash
# 使用 paddle.utils.run_check() 做安装自检
python -c "import paddle; paddle.utils.run_check()"
```

预期输出示例（**可见 GPU 数量与文案随机器配置变化，以下仅为示例**）：

```text
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 iluvatar_gpu.
PaddlePaddle works well on N iluvatar_gpus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```

## 如何卸载

使用 pip 卸载：

```bash
pip uninstall paddlepaddle-iluvatar
```
