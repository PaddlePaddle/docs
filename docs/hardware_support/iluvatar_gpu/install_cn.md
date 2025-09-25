# 天数 GPGPU 安装说明

飞桨框架 iluvatar_gpu 版支持天数 GPGPU 的训练和推理，提供两种安装方式：

1. 通过飞桨官网发布的 wheel 包安装
2. 通过源代码编译得到 wheel 包安装

## 天数 GPGPU 系统要求

| 要求类型 |   要求内容   |
| --------- | -------- |
| 芯片型号 | 天数智芯 系列芯片，包括 BI150 |
| 操作系统 | Linux 操作系统，包括 CentOS、Ubuntu、KylinV10 等 |

## 运行环境准备

推荐使用天数官方发布的天数 IXUCA 开发镜像，该镜像预装有天数 IXUCA 基础运行环境库。

```bash
# 拉取镜像
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
```

```bash
# 在 host 上安装 driver
wget https://ai-rank.bj.bcebos.com/Iluvatar/corex-driver-linux64-4.3.0.rc.9.20250624_x86_64_10.2.run
bash corex-driver-linux64-4.3.0.rc.9.20250624_x86_64_10.2.run
```

```bash
# 启动容器
docker run -itd --name paddle-ixuca-dev -v /usr/src:/usr/src -v /lib/modules:/lib/modules \
    -v /dev:/dev -v /home:/home --privileged --cap-add=ALL --pid=host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/device/paddle-ixuca:latest
docker exec -it paddle-ixuca-dev bash
```

#### 选项说明及可调整参数

##### ① `--name paddle-ixuca-dev`
- **作用**：指定容器名称。
- **可调整**：
  - 用户可改为其他名称，例如 `paddle-ixuca-test`，方便区分不同实验。

```bash
# 检查容器内是否正常识别天数 GPGPU 设备
ixsmi
```

```bash
# 预期输出
+-----------------------------------------------------------------------------+
|  IX-ML: 4.3.0       Driver Version: 4.3.0       CUDA Version: 10.2          |
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

iluvatar-gpu 支持插件式安装，需先安装飞桨 CPU 安装包，再安装飞桨 iluvatar-gpu 插件包。在启动的 docker 容器中，执行以下命令：

```bash
# 先安装飞桨 CPU 安装包
python -m pip install  --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# 再安装飞桨 iluvatar-gpu 插件包
python -m pip install --pre paddle-iluvatar-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/ixuca/
```
⚠️ 注意：nightly 版本为每日构建，可能存在不稳定性。如果需要更稳定的版本，建议使用 3.1 版本:https://www.paddlepaddle.org.cn/packages/stable/ixuca/
### 安装方式二：源代码编译安装

在启动的 docker 容器中，先安装飞桨 CPU 安装包，再下载 PaddleCustomDevice 源码编译得到飞桨 iluvatar-gpu 插件包。

```bash
# 下载 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# 在 PaddleCUstomDevice 根目录下执行以下指令更新子模块代码
git submodule sync
git submodule update --init --recursive

# 进入硬件后端(天数 iluvatar_gpu)目录
cd backends/iluvatar_gpu

# 先安装飞桨 CPU 安装包
python -m pip install  --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/

# 安装编译所需依赖
cd /tmp
wget https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-linux-x86_64.zip
unzip protoc-21.12-linux-x86_64.zip
mv bin/protoc /usr/local/bin/
rm -rf protoc-21.12-linux-x86_64.zip include bin
cd -

pip install --upgrade setuptools wheel

# 执行编译脚本
bash build_paddle.sh

# 编译产出在 build_pip 路径下，使用安装脚本进行安装
bash install_paddle.sh
```
⚠️ 注意：nightly 版本为每日构建，可能存在不稳定性。如果需要更稳定的版本，建议使用 3.1 版本。
## 基础功能检查

安装完成后，在 docker 容器中输入如下命令进行飞桨基础健康功能的检查。

```bash
# 列出可用硬件后端
python3 -c "import paddle; print(paddle.device.get_all_custom_device_type())"
```
```bash
# 预期得到如下输出结果
['iluvatar_gpu']
```
```bash
# 使用 paddle utils 模块的 `run_check` 功能检查 paddle_iluvatar_gpu 插件和 PaddlePaddle 主框架是否正常安装，需要指定 xccl 的后端为 iluvatar_gpu
export PADDLE_XCCL_BACKEND=iluvatar_gpu
python3 -c "import paddle; paddle.utils.run_check()"
```
```bash
# 预期得到输出如下
Running verify PaddlePaddle program ...
PaddlePaddle works well on 1 iluvatar_gpu.
PaddlePaddle works well on 16 iluvatar_gpus.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
```
## 如何卸载

请使用以下命令卸载 Paddle:

```bash
pip uninstall paddlepaddle paddle-iluvatar-gpu
```
